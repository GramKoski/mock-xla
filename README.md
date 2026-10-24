# mock-xla

A teaching reimplementation of XLA's shape/IR/pass machinery, with a jax-like
frontend. Built against the real `openxla/xla` tree as a guide.

```bash
python models/mlp.py        # trace, print HLO, execute, compare to numpy
python tests/test_stage1.py # 14 tests
```

## Stage 1: done

`@jit` traces Python with abstract values into an SSA dataflow IR, prints it as
HLO text, and executes it on numpy.

```
RMSNorm + SwiGLU MLP     27 instructions
max |mock-xla − numpy|   6.05e-09
```

## Read in this order

| file | lines | what to look at |
|---|---|---|
| `dtypes.py` | 85 | `Shape` = dims + dtype, and **nothing else**. `size_bytes` is what stage 3's cost model runs on. |
| `ir.py` | 110 | `Module → Computation → Instruction`. Why this is a graph, not a tape. `validate()` is the invariant passes break first. |
| `ops.py` | 243 | **The registry.** An op is `(shape_rule, impl, cost_rule)` — data, not code. |
| `trace.py` | 212 | `Tracer`, `Builder`, `jit`. Shape inference runs *here*, at trace time. |
| `numpy.py` | 270 | The frontend. Exists only to adapt numpy's loose semantics onto the strict IR. |
| `printer.py` | 98 | The debugging surface. Stage 2 dumps this after every pass. |
| `interp.py` | 72 | Walk the IR, call numpy. Deliberately dumb. |

## Four decisions worth knowing about

**The IR has no implicit broadcasting.** Elementwise ops require identical
shapes *and* dtypes, exactly like real HLO. `numpy.py` inserts the `constant`,
`broadcast`, `reshape` and `convert` instructions — which is why an HLO dump is
full of broadcasts you never wrote. `_binop` is that adapter.

**Shape inference happens at trace time, not in a pass.** So a shape error
raises with a Python traceback pointing at your line, not at IR with no
provenance. This is how JAX's `abstract_eval` works.

**`execute()` asserts the shape rule against the impl, per instruction.** If a
shape rule and its numpy implementation ever disagree, you find out immediately
and precisely. Cheap, and it caught real bugs while this was being written.

**Correctness and performance are measured by different mechanisms.** numpy
gives correctness; stage 3 will give *simulated* roofline cost. Fusion's whole
point is removing HBM round-trips, which you cannot observe on a laptop CPU —
so the cost model computes it instead, and the compiler can print the exact
number it used to decide.

## Deliberate simplifications (and what they cost)

| here | real XLA | why |
|---|---|---|
| `reduce` takes a string `computation="add"` | a full `HloComputation` | sub-computations are scaffolding at this scale; stage 3's reduction emitter only needs the kind |
| one output per instruction | multi-output via `kTuple` | needed only when multi-output fusion arrives |
| `bf16` executes as `f32` | native bf16 | numpy has no bf16; `bits=16` keeps the cost model honest |
| insertion order *is* execution order | an explicit schedule | stage 5 adds real scheduling; `validate()` guards the invariant until then |
| no autodiff | — | **deliberate.** `grad` is a *JAX* transform that runs above the compiler. By the time HLO exists the backward ops are already in the graph. XLA never sees a derivative. |

## Op set

18 ops. Growing toward "enough to write DeepSeek V3 unoptimized" means adding
rows to a table, not writing compiler code:

```
structural    parameter constant iota broadcast reshape transpose convert
elementwise   add subtract multiply divide maximum minimum power
              exp log sqrt rsqrt tanh negate abs floor sign logistic
compare       compare select
reduce        reduce (add/max/min/mul)
contraction   dot_general          <- batch dims + contracting dims, lowered to einsum
```

Still needed for V3: `gather` (simplified `take_along_axis` semantics, **not**
XLA's `offset_dims`/`collapsed_slice_dims` monster), `top_k`, `concatenate`,
`slice`, `pad`. MoE dispatch needs none of them if done densely via one-hot
`dot` — which is what "unoptimized" buys you.

## Next

Ordered to match XLA's real pipeline (`xla/service/gpu/gpu_compiler.cc:1861`,
`OptimizeHloModule`), because pass order is a fidelity property, not a
preference. Line numbers are the real stage boundaries:

```
1911  SPMD                stage 3   Mesh/PartitionSpec, ShardingPropagation,
                                    SpmdPartitioner -> inserts the collectives
1928  optimization        stage 2   algebraic / fold / CSE / DCE     << DONE
1935  collective opt      stage 3b  ReduceScatterCreator: rewrites
                                    dynamic-slice(all-reduce) -> reduce-scatter.
                                    ~30 lines, and it is where FSDP's
                                    reduce-scatter actually comes from.
1947  layout              stage 6   minor-to-major, insert copies on mismatch
1970  fusion              stage 4   roofline cost model + PriorityFusion
                                    + kLoop/kInput emitters
1977  async conversion    stage 5   all-reduce -> all-reduce-start/-done
      scheduling          stage 5   LatencyHidingScheduler + thunks
      buffer assignment   stage 6   liveness, aliasing, donation, peak memory
```

Note stage 2 sits at 1928, so it is already in the right place. SPMD goes
*before* it, which is why the build order is 2 -> 3 -> 4 rather than by
stage number.

### Why SPMD before fusion

Not just fidelity. Fusion must see **per-device** shapes, must be able to fuse
*around* collectives, and its cost model should price collectives
(`V(N-1)/N / beta`) alongside HBM traffic. Building the partitioner first means
the cost model is built on the right numbers from the start.

### Stage 3's acceptance test

Annotate parameters `P("fsdp", None)`; the partitioned 8-device program produces
**bit-identical** results to the single-device one, and the HLO contains an
`all-gather` and a `reduce-scatter` that appear in no source file — with
`replica_groups` decodable back to a mesh axis.

Showing a gradient reduce-scatter needs a backward pass, and there is no
autodiff on purpose. So the test model hand-writes a 2-layer MLP's backward as
ordinary forward ops — six matmuls — which is exactly the graph JAX would have
handed the compiler anyway.
