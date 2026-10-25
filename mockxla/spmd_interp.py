"""Execute a partitioned module on a simulated mesh of devices.

One numpy environment per device, stepped in lockstep. Ordinary instructions run
independently per device; collectives read across environments. That is the only
place in the project where devices interact, and it is ~40 lines.

What this buys: the acceptance test for stage 3 can assert that the PARTITIONED
program agrees with the SINGLE-DEVICE one. Without that, a partitioner is only
plausible.
"""

from __future__ import annotations

import numpy as np

from .ir import Module
from .ops import lookup
from .sharding import Dist, Mesh

COLLECTIVES = {"all-reduce", "all-gather", "reduce-scatter", "dynamic-slice"}

_REDUCERS = {"add": np.add, "max": np.maximum, "min": np.minimum}


def shard(arr: np.ndarray, dist: Dist, mesh: Mesh, device: int) -> np.ndarray:
    """Global array -> the slice `device` owns."""
    idx: list = [slice(None)] * arr.ndim
    for d, axis in enumerate(dist.spec):
        if axis is None:
            continue
        n = mesh.size(axis)
        c = mesh.coord(device, axis)
        w = arr.shape[d] // n
        idx[d] = slice(c * w, (c + 1) * w)
    return arr[tuple(idx)].copy()


def global_dims_of(local_dims: tuple, dist: Dist, mesh: Mesh) -> tuple:
    """Recover the global shape from one device's shape plus its distribution.

    Derived rather than looked up: reshard-inserted instructions have no
    meaningful `global_shape` to record, so relying on that field made the root
    of a partitioned module report its LOCAL shape as global.
    """
    return tuple(d * mesh.size(a) if a is not None else d
                 for d, a in zip(local_dims, dist.spec))


def unshard(pieces: list[np.ndarray], dist: Dist, mesh: Mesh,
            global_dims: tuple) -> np.ndarray:
    """Per-device slices -> the global array."""
    out = np.zeros(global_dims, dtype=pieces[0].dtype)
    for device, piece in enumerate(pieces):
        idx: list = [slice(None)] * len(global_dims)
        for d, axis in enumerate(dist.spec):
            if axis is None:
                continue
            n = mesh.size(axis)
            c = mesh.coord(device, axis)
            w = global_dims[d] // n
            idx[d] = slice(c * w, (c + 1) * w)
        out[tuple(idx)] = piece
    return out


def _run_collective(instr, envs: list[dict], mesh: Mesh) -> None:
    op = instr.opcode
    src = instr.operands[0]
    attrs = instr.attrs

    if op == "dynamic-slice":
        # No communication. Each device keeps its own piece of a value every
        # device already holds -- which is why "None -> sharded" is free.
        d = attrs["dimensions"][0]
        axis = attrs["axis"]
        n = mesh.size(axis)
        for dev, env in enumerate(envs):
            arr = env[src]
            w = arr.shape[d] // n
            c = mesh.coord(dev, axis)
            idx: list = [slice(None)] * arr.ndim
            idx[d] = slice(c * w, (c + 1) * w)
            env[instr.id] = arr[tuple(idx)].copy()
        return

    groups = attrs["replica_groups"]

    if op == "all-reduce":
        fn = _REDUCERS[attrs["computation"]]
        for group in groups:
            acc = envs[group[0]][src].copy()
            for dev in group[1:]:
                acc = fn(acc, envs[dev][src])
            for dev in group:
                envs[dev][instr.id] = acc.copy()
        return

    if op == "all-gather":
        d = attrs["dimensions"][0]
        for group in groups:
            cat = np.concatenate([envs[dev][src] for dev in group], axis=d)
            for dev in group:
                envs[dev][instr.id] = cat.copy()
        return

    if op == "reduce-scatter":
        d = attrs["dimensions"][0]
        fn = _REDUCERS[attrs["computation"]]
        for group in groups:
            acc = envs[group[0]][src].copy()
            for dev in group[1:]:
                acc = fn(acc, envs[dev][src])
            w = acc.shape[d] // len(group)
            for k, dev in enumerate(group):
                idx: list = [slice(None)] * acc.ndim
                idx[d] = slice(k * w, (k + 1) * w)
                envs[dev][instr.id] = acc[tuple(idx)].copy()
        return

    raise NotImplementedError(op)


def execute_spmd(module: Module, mesh: Mesh, *global_args,
                 trace: bool = False) -> np.ndarray:
    """Shard the inputs, run on every device, reassemble the output."""
    comp = module.entry
    envs: list[dict] = [{} for _ in range(mesh.num_devices)]

    for k, pid in enumerate(comp.params):
        arr = np.asarray(global_args[k], dtype=comp[pid].shape.dtype.np)
        for dev, env in enumerate(envs):
            env[pid] = shard(arr, comp[pid].dist, mesh, dev)

    for instr in comp.order():
        if instr.opcode == "parameter":
            continue
        if instr.opcode in COLLECTIVES:
            _run_collective(instr, envs, mesh)
        elif instr.opcode == "fusion":
            from .interp import execute_fusion
            for env in envs:
                env[instr.id] = execute_fusion(
                    module, instr, [env[o] for o in instr.operands])
        else:
            op = lookup(instr.opcode)
            for env in envs:
                env[instr.id] = np.asarray(
                    op.impl(*[env[o] for o in instr.operands], **instr.attrs))
        if trace:
            print(f"  {instr.id:>20} {instr.opcode:<16} "
                  f"local{envs[0][instr.id].shape} {instr.dist}")

    root = comp[comp.root]
    dims = global_dims_of(root.shape.dims, root.dist, mesh)
    return unshard([env[comp.root] for env in envs], root.dist, mesh, dims)
