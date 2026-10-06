# 0002 — The service runs each job in its own worker process

*Accepted, 2026-10-06.*

## Context

A web service has to run several sampling jobs at once. The usual options are
threads inside one process, or a pool of worker processes.

## Decision

Run each job in a worker process from a `ProcessPoolExecutor` created with the
`forkserver` start method. Each worker loads the ONNX session once, in the
pool initializer, and is recycled after a fixed number of jobs
(`max_tasks_per_child`).

## Reasons

Three properties of the sampler, each harmless in its own process and silently
wrong when shared:

1. **Seeding is process-global.** `sample(seed=...)` sets `np.random` and
   `random` for the whole process, as upstream RFdiffusion does. Two threads
   seeding and drawing at once corrupt each other's streams, and nothing would
   fail — results would just stop being reproducible.
2. **The IGSO(3) sampler draws from the global `np.random` on every step**, so
   even unseeded jobs interleave their draws.
3. **Memory grows as L³** (432 MB of attention scratch at L=300). A process can
   be capped and recycled; a thread cannot.

`forkserver`, not `fork`: forking a process that already runs ONNX Runtime's
thread pool copies its locks in an unknown state, and the child can hang on
its first inference.

## Consequences

- Sizing is explicit: on 4 vCPUs, 2 workers × 2 ORT threads. A worker must
  never leave `threads=None`, or every worker claims every core.
- Progress travels from worker to API through shared memory
  (`multiprocessing.Array`), not a queue.

## Revisit when

The random number generator is threaded through IGSO(3) and the denoiser as an
explicit `numpy.random.Generator`. Then jobs could share a process, and
designs could be batched into one forward pass.
