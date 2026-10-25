"""Mesh, PartitionSpec, and the distribution lattice the partitioner reasons over.

THE CENTRAL IDEA of stage 3: a value's distribution is not just "which dim is
split on which mesh axis". It also needs a PARTIAL state -- "this value is an
unreduced partial sum over axis A". Without that third state you cannot derive
FSDP's collectives; you have to special-case them.

    Dist(spec=('fsdp', None), partial=frozenset())      a sharded value
    Dist(spec=(None, None),   partial={'fsdp'})         a partial sum

And then every collective falls out of RESOLVING one distribution into another:

    axis -> None       on a dim      all-gather        (FSDP's forward gather)
    None -> axis       on a dim      dynamic-slice     (free! no communication)
    partial -> None                  all-reduce        (DDP's gradient reduce)
    partial -> axis    on a dim      all-reduce + dynamic-slice
                                     ... which ReduceScatterCreator then rewrites
                                     into a single reduce-scatter. That is where
                                     FSDP's reduce-scatter ACTUALLY comes from.

Nothing in the partitioner knows the words "FSDP" or "tensor parallel".
"""

from __future__ import annotations

import dataclasses
import math

from .dtypes import Shape

# A spec entry is a mesh axis name, or None meaning "not split on this dim".
AxisName = str
SpecEntry = "AxisName | None"


class Mesh:
    """A named n-dimensional grid of device ids.

    Axis ORDER is a performance decision, not a convention: the last axis varies
    fastest over device ids, so it gets consecutive (physically adjacent) devices.
    Put the axis whose collective is on the critical path -- 'tp' -- last.
    """

    def __init__(self, axis_names: tuple[AxisName, ...], axis_sizes: tuple[int, ...]):
        if len(axis_names) != len(axis_sizes):
            raise ValueError("axis_names and axis_sizes must agree in length")
        self.axis_names = tuple(axis_names)
        self.axis_sizes = tuple(int(s) for s in axis_sizes)

    @property
    def num_devices(self) -> int:
        return math.prod(self.axis_sizes)

    def size(self, axis: AxisName) -> int:
        return self.axis_sizes[self.axis_names.index(axis)]

    def stride(self, axis: AxisName) -> int:
        """Device-id stride between consecutive coordinates on `axis`.

        Row-major: device_id = sum_k coord_k * prod(sizes[k+1:]). So the stride
        of an axis is the product of the sizes to its RIGHT.
        """
        i = self.axis_names.index(axis)
        return math.prod(self.axis_sizes[i + 1:]) if i + 1 < len(self.axis_sizes) else 1

    def coord(self, device: int, axis: AxisName) -> int:
        return (device // self.stride(axis)) % self.size(axis)

    # So this is basically giving, for all x, y, z, etc all the groups (x, y, :, :, z)
    # where the free dims : are ina the axes tuple?
    def replica_groups(self, axes: tuple[AxisName, ...]) -> list[list[int]]:
        """Devices grouped so that each group varies ONLY over `axes`.

        This is the attribute you read off an HLO collective to recover which
        mesh axis it is on: group size tells you the axis extent, and the stride
        between members tells you which axis it is.
        """
        groups: dict[tuple, list[int]] = {}
        others = [a for a in self.axis_names if a not in axes]
        for d in range(self.num_devices):
            key = tuple(self.coord(d, a) for a in others)
            groups.setdefault(key, []).append(d)
        return [groups[k] for k in sorted(groups)]

    def __repr__(self):
        inner = ", ".join(f"{n}={s}" for n, s in zip(self.axis_names, self.axis_sizes))
        return f"Mesh({inner})"


def P(*entries) -> tuple:
    """PartitionSpec. P('fsdp', None) -> dim 0 split on 'fsdp', dim 1 whole."""
    return tuple(entries)


@dataclasses.dataclass(frozen=True)
class Dist:
    """How one value is laid out across the mesh."""

    spec: tuple          # one entry per dim: an axis name, or None
    partial: frozenset = frozenset()    # axes this value is an unreduced sum over

    # Fully replicated tensor
    @staticmethod
    def replicated(rank: int) -> Dist:
        return Dist((None,) * rank)

    @property
    def is_partial(self) -> bool:
        return bool(self.partial)

    # How many dimensions this tensor sits on?
    @property
    def rank(self) -> int:
        return len(self.spec)

    # How many dims this tensor is sharded over
    def sharded_axes(self) -> set:
        return {a for a in self.spec if a is not None}

    def dim_for_axis(self, axis: AxisName) -> int | None:
        for i, a in enumerate(self.spec):
            if a == axis:
                return i
        return None

    def without_partial(self) -> Dist:
        return Dist(self.spec, frozenset())

    def __str__(self):
        body = ",".join("*" if a is None else a for a in self.spec)
        p = f" partial={{{','.join(sorted(self.partial))}}}" if self.partial else ""
        return f"[{body}]{p}"


def local_shape(shape: Shape, dist: Dist, mesh: Mesh) -> Shape:
    """Global shape -> this device's shape. Partial does not change the shape."""
    dims = list(shape.dims)
    for i, axis in enumerate(dist.spec):
        if axis is None:
            continue
        n = mesh.size(axis)
        if dims[i] % n != 0:
            raise ValueError(
                f"cannot shard dim {i} of {shape} (size {dims[i]}) over "
                f"'{axis}' (size {n}): not divisible")
        dims[i] //= n
    return Shape(tuple(dims), shape.dtype)


def spec_axes_in_order(dist: Dist) -> list[tuple[int, AxisName]]:
    return [(i, a) for i, a in enumerate(dist.spec) if a is not None]
