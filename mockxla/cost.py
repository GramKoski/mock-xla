"""A roofline cost model, deliberately shaped like XLA's.

Fusion changes FLOPs by ZERO. It changes BYTES. So a cost model is not an
optional refinement here -- without one there is no way to say why fusion is
profitable, and PriorityFusion has nothing to sort by.

Faithful to xla/service/gpu/model/gpu_performance_model_base.{h,cc}, including
the thing most people get wrong: XLA does NOT use a pure max() roofline.

    CombineComputeAndMemoryAccessTime(compute, memory) =
        compute + memory - min(compute, memory) * kMemoryComputeParallelism

with kMemoryComputeParallelism = 0.95, whose comment says it "was chosen
empirically in early 2024". Rearranged, that is

    max(compute, memory) + 0.05 * min(compute, memory)

i.e. a textbook roofline plus a 5% tax on whichever term is smaller, because
"sometimes, compute or memory accesses will be stalled waiting for the other,
but usually they won't".

XLA's other constants, for reference:
    kKernelLaunchOverhead      1 us
    kNcclKernelLaunchOverhead  5 us      <- a collective costs 5x to launch
    kL2CacheSpeedup            2.5
    kL1CacheSpeedup            8
"""

from __future__ import annotations

import dataclasses
import math

from .dtypes import DType, Shape

# ---------------------------------------------------------------- the device


@dataclasses.dataclass(frozen=True)
class Device:
    name: str = "H100-ish"
    hbm_bandwidth: float = 3.35e12          # B/s
    peak_flops: float = 989e12              # bf16 dense tensor core
    peak_flops_f32: float = 67e12           # fp32 non-tensor-core
    launch_overhead: float = 1e-6           # XLA: kKernelLaunchOverhead
    nccl_launch_overhead: float = 5e-6      # XLA: kNcclKernelLaunchOverhead
    memory_compute_parallelism: float = 0.95
    devices_per_node: int = 8
    nvlink_bandwidth: float = 450e9         # B/s per GPU, intra-node
    ib_bandwidth: float = 50e9              # B/s per GPU, inter-node

    def flops_for(self, dtype: DType) -> float:
        return self.peak_flops_f32 if dtype.bits >= 32 else self.peak_flops

    def link_for(self, group_size: int) -> tuple[float, str]:
        """Intra-node groups ride NVLink; anything wider crosses the fabric.

        This is the whole HSDP argument in two lines: the moment a shard group
        is wider than a node, its collective drops onto a ~9x slower wire.
        """
        if group_size <= self.devices_per_node:
            return self.nvlink_bandwidth, "nvlink"
        return self.ib_bandwidth, "ib"


H100 = Device()


# ---------------------------------------------------------------- FLOPs

_TRANSCENDENTAL = {"exp", "log", "sqrt", "rsqrt", "tanh", "logistic", "power"}


def flops_of(instr, comp) -> float:
    op = instr.opcode
    n = instr.shape.num_elements

    if op in ("parameter", "constant", "iota", "broadcast", "reshape",
              "transpose", "convert", "fusion"):
        return 0.0          # fusion is handled by its callers, summing its body
    if op in ("all-reduce", "all-gather", "reduce-scatter", "dynamic-slice"):
        return 0.0
    if op == "dot_general":
        lhs = comp[instr.operands[0]].shape
        contracted = math.prod(lhs.dims[i] for i in instr.attrs["lhs_contracting"])
        return 2.0 * n * contracted          # the 2 is multiply-add
    if op == "reduce":
        return float(comp[instr.operands[0]].shape.num_elements)
    if op in _TRANSCENDENTAL:
        return 8.0 * n                       # a few ops each; XLA bills these higher too
    return float(n)                          # plain elementwise


# ---------------------------------------------------------------- bytes


def bytes_accessed(instr, comp, module=None) -> float:
    """Bytes crossing the HBM boundary for this instruction.

    For a FUSION this counts only its operands and its output -- every
    intermediate stays in registers and never touches memory. THAT is the entire
    source of fusion's benefit, and the reason it shows up here and nowhere else.
    """
    if instr.opcode in ("parameter", "constant", "iota"):
        return 0.0
    out = instr.shape.size_bytes
    ins = sum(comp[o].shape.size_bytes for o in instr.operands)
    return float(ins + out)


# ---------------------------------------------------------------- time


def time_of(instr, comp, module=None, device: Device = H100) -> dict:
    op = instr.opcode

    if op in ("all-reduce", "all-gather", "reduce-scatter"):
        return _collective_time(instr, comp, device)
    if op == "dynamic-slice":
        # No communication: every device already holds the data.
        return {"compute": 0.0, "memory": bytes_accessed(instr, comp) / device.hbm_bandwidth,
                "total": bytes_accessed(instr, comp) / device.hbm_bandwidth, "kind": "slice"}

    if op == "fusion":
        body = module.computations[instr.attrs["computation"]]
        flops = sum(flops_of(i, body) for i in body.order())
        mem = bytes_accessed(instr, comp)
    else:
        flops = flops_of(instr, comp)
        mem = bytes_accessed(instr, comp)

    t_compute = flops / device.flops_for(instr.shape.dtype)
    t_memory = mem / device.hbm_bandwidth
    return {"compute": t_compute, "memory": t_memory,
            "total": combine(t_compute, t_memory, device) + device.launch_overhead,
            "kind": "kernel", "flops": flops, "bytes": mem}


def combine(t_compute: float, t_memory: float, device: Device = H100) -> float:
    """XLA's CombineComputeAndMemoryAccessTime. See the module docstring."""
    p = device.memory_compute_parallelism
    return t_compute + t_memory - min(t_compute, t_memory) * p


def _collective_time(instr, comp, device: Device) -> dict:
    """The ring formulas, with the link chosen by whether the group fits a node.

        all-gather / reduce-scatter    V (N-1)/N / beta
        all-reduce                   2V (N-1)/N / beta
    """
    groups = instr.attrs["replica_groups"]
    n = max(len(g) for g in groups)
    beta, link = device.link_for(n)
    # V is the FULL (gathered / unreduced) buffer in every case.
    if instr.opcode == "all-gather":
        v = instr.shape.size_bytes
    elif instr.opcode == "reduce-scatter":
        v = comp[instr.operands[0]].shape.size_bytes
    else:
        v = instr.shape.size_bytes
    factor = 2.0 if instr.opcode == "all-reduce" else 1.0
    t = factor * v * (n - 1) / n / beta if n > 1 else 0.0
    return {"compute": 0.0, "memory": 0.0,
            "total": t + device.nccl_launch_overhead,
            "kind": f"collective/{link}", "bytes": v, "group": n}


# ---------------------------------------------------------------- whole module


def module_cost(module, device: Device = H100) -> dict:
    comp = module.entry
    total = compute = memory = comm = 0.0
    launches = 0
    traffic = 0.0
    for instr in comp.order():
        if instr.opcode == "parameter":
            continue
        t = time_of(instr, comp, module, device)
        total += t["total"]
        compute += t["compute"]
        memory += t["memory"]
        traffic += t.get("bytes", 0.0) if t["kind"] == "kernel" else 0.0
        if t["kind"].startswith("collective"):
            comm += t["total"]
        launches += 1
    return {"total": total, "compute": compute, "memory": memory,
            "comm": comm, "launches": launches, "hbm_traffic": traffic}


def report(module, device: Device = H100, top: int = 0) -> str:
    c = module_cost(module, device)
    lines = [
        f"device            {device.name}  "
        f"HBM {device.hbm_bandwidth/1e12:.2f} TB/s  "
        f"peak {device.peak_flops/1e12:.0f} TF/s",
        f"kernel launches   {c['launches']}  "
        f"({c['launches'] * device.launch_overhead * 1e6:.0f} us of pure overhead)",
        f"HBM traffic       {c['hbm_traffic']/1e6:.2f} MB",
        f"predicted time    {c['total']*1e6:.2f} us"
        f"   (compute {c['compute']*1e6:.2f}, memory {c['memory']*1e6:.2f}"
        + (f", comm {c['comm']*1e6:.2f}" if c["comm"] else "") + ")",
    ]
    if top:
        rows = sorted(
            ((time_of(i, module.entry, module, device), i)
             for i in module.entry.order() if i.opcode != "parameter"),
            key=lambda kv: -kv[0]["total"])[:top]
        lines.append(f"  {'instruction':<26}{'time us':>10}{'kind':>20}")
        for t, i in rows:
            lines.append(f"  {i.id:<26}{t['total']*1e6:>10.3f}{t['kind']:>20}")
    return "\n".join(lines)
