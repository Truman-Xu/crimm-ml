# Instructions for AI agents

crimm-ml runs RFdiffusion loop building for crimm on ONNX Runtime, without PyTorch. It
depends on crimm (`/home/ziqiaoxu/crimm`); crimm calls it for `pip install crimm[ml]`.

**Before doing anything, read [`docs/dev/AGENT_HANDOFF.md`](docs/dev/AGENT_HANDOFF.md)**
sections 0 to 5. It holds the protocol, the owner's standing instructions, environment
facts, known traps, current state and claims. Planned work is in
[`docs/dev/ROADMAP.md`](docs/dev/ROADMAP.md); the developer rules, which bind you too, are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). When you finish, log what you did there.

The rules that matter most, in case you read nothing else:

- Do not commit or push unless the owner asks.
- **Sign commits, never files.** When you do commit, add a `Co-Authored-By:` trailer naming
  your exact model. Put no author names, banners or session notes in any file. Add your
  model family to `AUTHORS.md` with your first commit. Full rules:
  [CONTRIBUTING.md](CONTRIBUTING.md#credit-and-authorship).
- Results must not change by accident: `tests/test_golden.py` pins seeded designs. Run
  `pytest -m "not slow"` always, and the full suite (needs `CRIMM_ML_TEST_MODEL`) for any
  change near the sampler or model.
- Never take a timing on the shared login node; use `srun --partition=cpu --cpus-per-task=32`.
- crimm is a separate repo with other agents working in it. Before touching it, follow its own
  `docs/dev/AGENT_HANDOFF.md`.
- The owner's files in the RFdiffusion fork (`Tests.ipynb`, `.vscode/`) and the fork's
  `rfdiffusion/` package are off limits.
