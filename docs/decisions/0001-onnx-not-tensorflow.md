# 0001 — Ship the ONNX export; no TensorFlow port, no TPU

*Accepted, 2026-10-06.*

## Context

The goal is that `pip install crimm[ml]` gives a working loop-building model in
an ordinary scientific Python environment, with no hardware questions. Two
routes were on the table besides the ONNX export: reimplementing the reduced
RFdiffusion model in TensorFlow (or converting the ONNX graph to it), and
running inference on TPUs.

## Decision

Ship the ONNX export run by ONNX Runtime. Do not port to TensorFlow. Do not
target TPUs.

## Reasons

- **ONNX already meets the goal.** Runtime dependencies are numpy, scipy,
  omegaconf and onnxruntime (~15 MB CPU wheel, loose pins). CPU speed is at
  parity with PyTorch eager: 0.99 s against 0.90 s per forward at L=76, 8
  threads. On GPU it is about 2x faster than PyTorch (0.16 against 0.35 s per
  step on an RTX 2080 Ti).
- **A TensorFlow port is a second implementation forever.** ~5,400 lines of
  SE(3)-Transformer, scatter aggregation and top-k graph construction, plus a
  checkpoint remap, all to be kept in step with the PyTorch reference.
- **Converters would fail where it hurts.** `onnx2tf` (`onnx-tf` is archived)
  would have to handle `NonZero`, `TopK`, `ScatterND`, `ScatterElements` with
  reduction, `Einsum` and a symbolic sequence length. The CUDA `ReduceL2` bug
  found in this project showed how such failures look: silently wrong numbers,
  no error.
- **TensorFlow makes installation harder, not easier.** It pins numpy,
  protobuf and ml-dtypes and its GPU wheel runs to hundreds of MB — the opposite
  of what an environment full of rdkit, biopython and pandas needs.
- **TPUs fit the opposite workload.** They want XLA with static shapes and
  large batches. This is a batch-1, 50-step sequential loop whose graph edge
  count comes from `NonZero`.

## Consequences

- The minimum ONNX Runtime is 1.18 (the model is IR version 10; 1.17 cannot
  load it). Python 3.9 tops out at onnxruntime 1.19, so `requires-python` is
  3.10.
- Speed beyond CPU parity comes from GPUs (`onnxruntime-gpu`), not a framework
  switch.

## Revisit when

AOTInductor can produce a genuinely standalone artifact (re-check each PyTorch
release; `COMPILING.md` §3). Nothing on the TensorFlow side is expected to
change this.
