# crimm-ml roadmap

Written 2026-10-06 against commit `295c46d`. Decisions behind it are in
[`docs/decisions/`](../decisions/README.md); shared agent state is in
[AGENT_HANDOFF.md](AGENT_HANDOFF.md).

Goal: RFdiffusion loop building that installs with `pip install crimm[ml]` and runs as a
containerized web service that takes a structure and returns it with its gaps filled. A
browser GUI follows once the API schema settles.

Sizes: S is under a day, M is a few days, L is a week or more. Stage IDs (S0 to S9) are
stable; cite them with their title.

## Done

| Stage | What | Evidence |
| --- | --- | --- |
| S0 Records | Decision records 0001 to 0003; crimm's `import-fix.md` work order; the fork's DISTRIBUTING.md now recommends ONNX | `docs/decisions/` |
| S1 New repo | Moved from the RFdiffusion fork, renamed to `crimm_ml`, hardened (packaged config, cache paths, `ContigError`, logging, `seed=`) | Fork and new sampler bit-identical at T=15 and T=50; `pytest` passes; all six phases of `tools/export/verify_numpy_sampler.py` pass |
| S2 Model asset | `assets.py` and `crimm-ml assets fetch/verify/path/warm`; checksummed fetch from a URL or a directory | Fetch from a directory and over HTTP, corrupted-file rejection and `verify` all exercised in a clean venv |

## S3 Glue in crimm — tracked in crimm, not here

`crimm/Modeller/RFLoopBuilder.py` (the `build_residues` placeholder from the RFdiffusion
fork's `Tests.ipynb`, cell 7, and the gap loop in cells 9 to 13) and the `ml` extra. Its
rows live in crimm's `docs/dev/ROADMAP.md`, section "Parallel track: crimm-ml
integration". It waits for crimm's "1.5 Make rdkit and nglview optional at import time".

Done when, on this side: `crimm_ml.crimm_utils` and `crimm_ml.minimal_pipeline_loop_builder`
are deleted in the next release, and README points to the crimm function.

## S4 Release

| Item | Done when | Size |
| --- | --- | --- |
| Create the GitHub repo `Truman-Xu/crimm-ml` (owner does this; no `gh` on the machine) and push `main` | `git push -u origin main` succeeds | S |
| Publish the model pair as a GitHub Release asset (repo must be public, or downloads need auth); put the URL template in `src/crimm_ml/data/assets.json` | `crimm-ml assets fetch` with no `--from` works in a clean venv | S |
| Zenodo deposit of the same files as a second mirror with a DOI | Second URL in `assets.json`; fetch falls back to it | S |
| CI: `pytest -m "not slow"` on Python 3.10 to 3.13, plus a job that fetches the model and runs the golden tests | Required check on `main` | M |
| TestPyPI dry run: `python -m build`, `twine check`, upload, install in a fresh venv with `--extra-index-url https://pypi.org/simple`, run the golden fixtures, and confirm `pip install crimm[ml]` resolves the crimm ↔ crimm-ml dependency cycle | All pass from TestPyPI | S |
| Trusted publishing from a GitHub Actions workflow (`id-token: write`, no token secret), exercised against TestPyPI first; first real version `0.1.0a1` | Package on PyPI | S |

A version number can never be re-uploaded to PyPI, even after deletion. Rehearse on TestPyPI.

## S5 Service

FastAPI app under `src/crimm_ml/service/`, behind a `[service]` extra. Design reasons are in
decision 0002.

| Item | Done when | Size |
| --- | --- | --- |
| `settings.py` (pydantic-settings, `CRIMM_ML_` prefix): every limit in one place (max `L`, allowed `T`, max designs, body size, job timeout, queue depth) | Each limit is an env var with a default | S |
| `schemas.py`: the request and response contract, with `api_version` and provenance (crimm_ml and crimm versions, model sha256, seed) in every response | OpenAPI JSON committed; a CI test diffs it | S |
| `jobs.py`: `JobStore` ABC + in-memory implementation. `storage.py`: `ObjectStore` ABC + local directory. Opaque `uuid4` job ids; key layout `jobs/{id}/...` | Redis/Postgres/S3 later means a new subclass, not new handlers | M |
| `pool.py` + `worker.py`: `ProcessPoolExecutor`, `forkserver` start method, initializer loads the ONNX session once, `max_tasks_per_child`; 2 workers × 2 ORT threads on 4 vCPU; progress through a `multiprocessing.Array` fed by `sample(progress=callback)` | Two concurrent jobs give the same coordinates as sequential runs with the same seeds | M |
| Endpoints: `POST /v1/jobs` → 202 + `Location` + `Retry-After`; `GET /v1/jobs/{id}`; `GET /v1/jobs/{id}/result`; `DELETE /v1/jobs/{id}`; `GET /v1/meta`; `/healthz` (process alive) and `/readyz` (model verified, a worker holds a live session) | Malformed contig → 400, over-limit `L` → 422, never 500 | M |
| The job: parse PDB/mmCIF with crimm, find gaps, sample, rebuild residues with crimm's `RFLoopBuilder`, return the filled structure as PDB text plus a per-gap summary | 5IEV submitted and polled to completion matches the library result for the same seed | M (needs S3) |
| `crimm-ml-serve` console script | Runs locally with one command | S |

## S6 Container

| Item | Done when | Size |
| --- | --- | --- |
| `deploy/Dockerfile`, `python:3.11-slim`, three stages: dependencies (metadata only), assets (keyed on an `ASSET_TAG` build arg; `crimm-ml assets fetch` + `warm --T 15 50`), source last | `podman build` works; changing source does not re-download the model | M |
| Non-root user, read-only root filesystem + tmpfs, `MALLOC_ARENA_MAX=2`, `CRIMM_ML_CACHE` pointed at the baked assets | `podman run --read-only` passes the S5 API test | S |
| Memory limit measured with `podman stats` at the configured `L` cap | Limit written into the run instructions | S |

Use podman/buildah; there is no Docker on the machine and none is needed (decision 0003).

## S7 Cloud

One 4 vCPU / 16 GB VM; rootless podman + Quadlet unit (`loginctl enable-linger`); Caddy for
TLS, body-size cap and rate limit; a shared bearer token; a systemd timer deleting results
after 7 days; uptime check on `/readyz`; billing alert. Done when the service survives a
reboot and an SSH logout, and `curl https://<host>/v1/meta` works. Needs the owner's cloud
account. Size M. Move to serverless only under the exit condition in decision 0003.

## S8 Prepare for scale and the GUI

Cheap now, expensive later; most are built into S5: `/v1` prefix, opaque ids, the two ABCs,
no server-side session state, CORS origins from env (default empty), PDB text in results,
`/v1/meta` so a GUI renders its form from server limits, `Retry-After` / `poll_after_ms`,
`X-Request-Id` in every log line.

Library item, its own change: thread a `numpy.random.Generator` through `IGSO3.sample`,
`sample_vec`, `reverse_sample_vectorized` and `Denoise` instead of the global `np.random`.
This changes the random stream, so regenerate the golden fixtures in the same commit and bump
the version. It is what later allows jobs to share a process and designs to be batched.
Size M.

## S9 Teaching document

`docs/serving-explained.md`, written for the owner, who is new to containers and cloud.
Written **as each of S5 to S8 is built**, not ahead of it. Chapters: images, containers and
registries; why a job API; processes versus threads here; serving a model; putting it on the
internet; keeping it running; cloud pricing models; what scaling would look like next (and why
Kubernetes is not on that list yet).

## Smaller open items

| Item | Detail | Size |
| --- | --- | --- |
| Honest CPU benchmark | `tools/export/bench_onnx.py` times PyTorch with one un-warmed call, which flatters ONNX: the quoted 0.95 to 1.11x is closer to 1.1 to 1.3x. Warm both and repeat, on a `--cpus-per-task=32` reservation. | S |
| Next ONNX hot spots | At L=76, 8 threads: `Pad` 114 ms (from the pad+cat basis), `Einsum` 130 ms (~540 small ones), `ScatterElements` 88 ms (torch's `scatter_add_` takes 12 ms) | M each |
| Memory cap | Attention scratch grows as L³ (432 MB at L=300). The library has no cap; the service enforces one. Consider a clear error in the sampler above a configurable `L`. | S |
| Owner's notebook | The fork's `Tests.ipynb` cells 6 and 9 use the pre-refactor API. The owner's file; tell them, do not edit it. | — |
| Fork cleanup | ~1 GB of profile JSONs, four superseded ONNX models and `models/ln_probe/` in the fork's untracked `models/`. Ask before deleting. | S |
