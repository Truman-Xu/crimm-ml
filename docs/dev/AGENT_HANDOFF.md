# AGENT_HANDOFF

Shared memory for AI agents working on crimm-ml. Any model, any tool. Written for agents
first; humans read it second.

```yaml
schema: agent-handoff/1          # same protocol as crimm's docs/dev/AGENT_HANDOFF.md
repo: github.com/Truman-Xu/crimm-ml
local_path: /home/ziqiaoxu/crimm-ml
default_branch: main
sibling_repos:
  crimm: /home/ziqiaoxu/crimm                 # depends on; has its own AGENT_HANDOFF.md
  rfdiffusion_fork: /home/ziqiaoxu/RFdiffusion # provenance, checkpoint, research probes
last_updated: 2026-10-06
```

## 0. Protocol

Identical to crimm's, so an agent moving between the two repos follows one set of rules.
Read this section fully. Skim the rest for what your task touches.

### On arrival

1. Read sections 1 to 4. They are short and change rarely.
2. Read section 5 (STATE) and the newest three entries of section 8 (LOG).
3. Run `git status --short` and `git log --oneline -5`. If they disagree with STATE, trust
   git and correct STATE.
4. Check section 6 (CLAIMS) before editing a file. If another agent holds it, do not edit it.
5. **If your task touches crimm** (anything under `/home/ziqiaoxu/crimm`), also follow the
   arrival steps of crimm's `docs/dev/AGENT_HANDOFF.md`: other agents work there
   concurrently, and claims live in that file, not this one.

### On departure

1. Append one entry to LOG (newest first, template in section 8).
2. Update STATE so it is true now. Rewrite it in place; do not append.
3. Release your CLAIMS.
4. Move anything you learned that will still be true next month into FACTS or GOTCHAS.
5. If you changed anything crimm relies on, or anything about how crimm calls this package,
   log a one-line pointer in crimm's AGENT_HANDOFF too.

### Writing rules

- **Tag every claim with how you know it.** Use exactly one of:
  - `[ran]` you executed it and saw the result this session
  - `[read]` you read the code or file; did not execute
  - `[told]` the owner said so
  - `[inferred]` your conclusion; say from what
- **Date everything** as `YYYY-MM-DD`. Never write "today", "recently" or "currently".
- **Cite locations** as `path:line` or `path::symbol`. Prefer the symbol; line numbers rot.
- **State what you did not check.**
- **One fact, one place.** Link to the files below instead of copying from them.
- **Do not delete another agent's entry.** If it is wrong, add `SUPERSEDED YYYY-MM-DD by
  <agent>: <why>` under it.
- **Identify yourself** as `<model>/<tool>/<short task name>`, for example
  `claude-opus/claude-code/crimm-ml`.
- Keep entries terse.

### What goes where

| Kind of information | File |
| --- | --- |
| What the package is, install, quick start | `README.md` |
| How the ONNX model is built, made fast, verified; all measurements | `docs/COMPILING.md` |
| Why ONNX is the shipped form; what was rejected | `docs/DISTRIBUTING.md`, `docs/decisions/` |
| History of reducing RFdiffusion to the minimal model | `docs/MINIMAL_MODEL.md` (research log, fork paths) |
| Planned work, order, status | `docs/dev/ROADMAP.md` |
| Developer rules, including how people and AI models are credited | `CONTRIBUTING.md` |
| Who the authors are | `AUTHORS.md` (third-party origins: `NOTICE`) |
| Who is doing what, what was tried, traps, state | this file |

## 1. Orientation

- crimm-ml runs RFdiffusion loop building on ONNX Runtime with no PyTorch: a NumPy sampler
  (`src/crimm_ml/minimal_loop_sampler.py::MinimalLoopSampler`, `numpy_rf/`) around a model
  object from `crimm_ml.models.load_model` (ONNX, or a PyTorch checkpoint with the `[torch]`
  extra). Input is PDB lines plus a contig string; output is NumPy coordinates. `[read]`
- Owner: Truman Xu (`Truman-Xu`), who also owns crimm. `[told 2026-10-06]`
- **Relationship to crimm.** crimm-ml depends on crimm. The chain-facing glue (gaps, contig,
  rebuilding full residues) lives in **crimm** (`crimm/Modeller/RFLoopBuilder.py`, planned;
  crimm ROADMAP "Parallel track: crimm-ml integration"). crimm imports `crimm_ml` only inside
  functions; crimm-ml imports crimm normally. Users install `pip install crimm[ml]`.
  `[told 2026-10-06]`
- **Relationship to the RFdiffusion fork** (`/home/ziqiaoxu/RFdiffusion`, branch
  `crimm-integration-onnx`, commits `823fe61` + `763bbe5`): this repo was moved out of its
  `crimm_integration/`. The fork keeps the research probes that need upstream `rfdiffusion`
  and the PyTorch checkpoint `models/Base_ckpt.pt`. `[ran 2026-10-06]`
- Goals: `pip install crimm[ml]` with no hardware configuration; then a containerized REST
  service, structure in, filled structure out; then a web GUI. See ROADMAP. `[told 2026-10-06]`
- Licence: GPL-3.0-or-later; RFdiffusion-derived parts keep their BSD notice (`NOTICE`).
  `[told 2026-10-06]`

## 2. Standing instructions from the owner

| Date | Instruction | Tag |
| --- | --- | --- |
| 2026-10-06 | Do not commit or push unless asked. | `[told]` |
| 2026-10-06 | Never modify `rfdiffusion/` in the fork. | `[told]` |
| 2026-10-06 | The owner's `Tests.ipynb` and `.vscode/` in the fork are theirs; do not edit. `crimm/Modeller/CoordManipulator.py` and `Solvator.py` in crimm carry uncommitted owner work. | `[told]` |
| 2026-10-06 | Keep crimm as a hard dependency. The postprocessing that builds residues is crimm functions, in crimm's repo. | `[told]` |
| 2026-10-06 | No TensorFlow, no TPU. ONNX Runtime is the shipped runtime (decision 0001). | `[told]` |
| 2026-10-06 | First deployment: public cloud, CPU REST API in a container, no GUI yet. | `[told]` |
| 2026-10-06 | Write `docs/serving-explained.md` for the owner as S5 to S8 are built; they are new to containers and cloud and want to learn. | `[told]` |
| 2026-10-06 | AI models that write code here are credited as authors. The owner's only condition: keep the code base clean and decluttered. Rules: `CONTRIBUTING.md`, "Credit and authorship" (sign commits, never files). | `[told]` |
| 2026-09-17 | Keep `compilation.py` / `compiled_model.py` (torch.compile) as an option; they are not part of the sampler. | `[told]` |
| 2026-09-17 | tqdm stays optional; the owner installs packages in their own envs. | `[told]` |

Working agreements (change them by logging why):

- Do not install packages into the owner's conda envs; use a throwaway venv in your scratch
  space. `[inferred: owner installs into their envs themselves]`
- Any change meant to keep results the same must keep `tests/test_golden.py` passing. A change
  meant to alter results regenerates fixtures with `tools/make_golden.py` in the same commit.
- Benchmarks only on a dedicated node (see G1). Never quote a login-node timing.

## 3. Environment facts

Machine: the owner's cluster login node (Rocky Linux 8, SLURM, 40 cores, shared).

| Fact | Tag |
| --- | --- |
| No `python` on `PATH`; `/usr/bin/python3` is 3.6. `/usr/bin/python3.11` and `python3.12` exist (use for venvs). | `[ran 2026-10-06]` |
| Conda env `crimm-export`: Python 3.11, torch 2.14+cu130, onnxruntime 1.29, onnx 1.22, crimm (editable from `/home/ziqiaoxu/crimm`), rdkit, tqdm. No pytest. Use for export and torch verification. | `[ran 2026-10-06]` |
| Conda env `RFDiffusion`: Python 3.10, torch 2.4+cu121, onnxruntime **and** onnxruntime-gpu 1.23.2, crimm editable. The owner's notebook env. | `[ran 2026-10-06]` |
| Model files: `/home/ziqiaoxu/RFdiffusion/models/rf_loop_dynamic_tp_ort.onnx` + `.onnx.data` (current; sha256 in `src/crimm_ml/data/assets.json`) and `Base_ckpt.pt`. Upstream schedule caches in `/home/ziqiaoxu/RFdiffusion/schedules/`. | `[ran 2026-10-06]` |
| podman 4.9 and buildah available; no docker; no `gh`. SSH to GitHub authenticates as Truman-Xu. | `[ran 2026-10-06]` |
| PyPI is reachable from the login node. SLURM compute nodes do not resolve DNS. | `[ran 2026-10-06]` login; `[read crimm AGENT_HANDOFF]` compute |
| `/home` is NFS, visible everywhere; `/tmp` is node-local. Point caches at `$HOME`, not `/tmp`. | `[ran 2026-09]` |
| GPU partitions: `gpu2080` (2080 Ti), `ada5000`, `ada6000`, `gpuA5500`, `cryo-em` (A6000), `zimA10`. Driver 580 (CUDA 13). | `[ran 2026-09]` |

Commands that work `[ran 2026-10-06]`:

```bash
# fast tests in a throwaway venv (torch-free)
python3.11 -m venv $SCRATCH/v && $SCRATCH/v/bin/pip install -e "/home/ziqiaoxu/crimm-ml[dev]"
$SCRATCH/v/bin/python -m pytest -q -m "not slow"
# golden tests (need the model; ~4 min on the login node)
CRIMM_ML_TEST_MODEL=/home/ziqiaoxu/RFdiffusion/models/rf_loop_dynamic_tp_ort.onnx $SCRATCH/v/bin/python -m pytest -q
# cache management
CRIMM_ML_CACHE=$SCRATCH/cache crimm-ml assets fetch --from /home/ziqiaoxu/RFdiffusion/models
crimm-ml assets verify ; crimm-ml assets warm --T 15 50
# torch reference verification (crimm-export env; RFDIFFUSION_DIR defaults to ../RFdiffusion)
/home/ziqiaoxu/.conda/envs/crimm-export/bin/python tools/export/verify_numpy_sampler.py --T 15 --threads 4
# dedicated CPU node for any timing
srun --partition=cpu --cpus-per-task=32 --time=01:00:00 <command>
```

## 4. Gotchas

| ID | Trap | Tag |
| --- | --- | --- |
| G1 | **Login-node and shared-node timings lie.** On busy `gollum` both runtimes ran ~1.8x slower, ORT more so; a 16-CPU `cpu`-partition allocation shares the node and once produced a 5.8x fake cliff. Reserve `--cpus-per-task=32`. | `[ran 2026-09]` |
| G2 | **ORT CUDA `ReduceL2` returns its input unchanged when every reduced axis has length 1** (ORT 1.23.2). Coordinates come out near -2000. Fixed in `se3/layers.py::_feature_norm` and by rewrite 3 in `tools/export/onnx_ort_rewrites.py`; an export without that rewrite is wrong on GPU. | `[ran 2026-09-17]` |
| G3 | ORT silently falls back to CPU when CUDA libraries fail to load. `OnnxRFModel` calls `ort.preload_dlls()` and warns; check `model.providers`. | `[ran 2026-09-17]` |
| G4 | The `.onnx.data` file name is recorded inside the graph. Both files must sit in one directory under their original names. | `[ran 2026-10-06]` |
| G5 | onnxruntime ≥ 1.18 is required (IR version 10); 1.17 cannot load the model. Python 3.9's newest onnxruntime is 1.19. | `[ran 2026-10-06]` |
| G6 | `sample(seed=)` seeds the **process-global** `np.random` and `random`; IGSO(3) draws from the global RNG every step. Never run two samplers in one process concurrently (decision 0002). | `[read]` |
| G7 | A schedule `.npz` takes priority over upstream's `.pkl`. A wrong `.npz` silently wins; delete it rather than "fixing" the pkl. | `[ran 2026-09-17]` |
| G8 | Upstream IGSO(3) tables are float64 (`max_sigma` arrives as a torch tensor), and their `score_norm` is inf/NaN where the density underflows. Computed tables differ only there; T=50 coordinates are identical either way. | `[ran 2026-09-17]` |
| G9 | `import crimm` from PyPI (2026.2.2) fails without rdkit, and pulls nglview + ipywidgets. Until crimm's import fix ships, `crimm_ml.crimm_utils` cannot import in a torch-free, rdkit-free venv; the core sampler is unaffected. | `[ran 2026-10-06]` |
| G10 | `tools/export/bench_onnx.py` times PyTorch with one un-warmed call, flattering ONNX. At 16 threads both runtimes' pools fight in one process; trust 8-thread numbers. | `[read]`, `[ran 2026-09]` |
| G11 | Memory grows as L³ (attention scratch, 432 MB at L=300). `--lmax` is not enforced by ORT. | `[ran 2026-09]` |
| G12 | `onnx.save(save_as_external_data=True)` takes minutes on NFS (reopens the file per tensor). Load graph-only, copy `.data`, re-point initializer locations (`onnx_ort_rewrites.py` does this). | `[ran 2026-09-16]` |
| G13 | `pkill -f <script>` from a shell whose command line contains the script name kills the shell too. | `[ran 2026-09-16]` |
| G14 | Test ONNX export idioms on a 4-line toy module; a full export takes 35 to 90 minutes. | `[ran 2026-09-12]` |
| G15 | On GPU in PyTorch, `index_add_` is not bit-reproducible; set `torch.use_deterministic_algorithms(True, warn_only=True)` before comparing. | `[ran 2026-09-09]` |
| G16 | crimm's `docs/dev/*.md` are edited by other sessions concurrently. Re-read before a targeted edit; never rewrite whole files. | `[ran 2026-10-06]` |

## 5. State

Rewrite in place. True as of `2026-10-06`.

```yaml
branch: main
head: "second commit on main (agent docs, authorship rules); run git log"
uncommitted_agent_work: none
remote: git@github.com:Truman-Xu/crimm-ml.git   # repo not yet created on GitHub; nothing pushed
published: nothing                  # not on PyPI; assets.json has no download URL
tests: {fast: "6 passed + 2 skipped without model", full: "8 passed, 1 skipped", date: 2026-10-06}
roadmap_done: [S0 Records, S1 New repo, S2 Model asset]
roadmap_next:
  - "S4: owner creates the GitHub repo; push"
  - "S3: waits on crimm ROADMAP 1.5 (optional imports), being done by another session"
  - "S5 Service: can start; the job body needs S3"
fork_uncommitted_agent_work:        # /home/ziqiaoxu/RFdiffusion
  - crimm_integration/DISTRIBUTING.md, MINIMAL_MODEL.md   # same text is committed here
```

## 6. Claims

| Path or area | Agent | Since | Purpose |
| --- | --- | --- | --- |
| (none) | | | |

## 7. Open questions

| ID | Question | Raised |
| --- | --- | --- |
| Q1 | Public or private GitHub repo? Release downloads from a private repo need authentication, so it must be public before S4's model release. Owner decision. | 2026-10-06 |
| Q2 | Should crimm get a root `AGENTS.md` pointing to its `docs/dev/AGENT_HANDOFF.md`? Tools that auto-read root files will not find the handoff otherwise. Owner decision, raised in crimm's handoff entry 3. | 2026-10-06 |
| Q3 | Cloud provider and account for S7. Owner decision. | 2026-10-06 |

## 8. Log

Newest first. Append only. Template:

```markdown
### YYYY-MM-DD <model>/<tool>/<task>

- asked: <what the owner asked for, one line>
- did: <what changed, with paths>
- verified: <what you ran and the result>
- not verified: <what you assumed or skipped>
- found: <bugs, surprises; point to ROADMAP rows or GOTCHAS>
- next: <the obvious next step, and anything blocking it>
```

### 2026-10-06 claude-opus/claude-code/crimm-ml (entry 2)

- asked: credit the AI author in the package; set credit rules for all future agents, with
  the owner's one condition (clean, decluttered code base); declare it for humans.
- did: `AUTHORS.md`, `CONTRIBUTING.md` (developer rules; "Credit and authorship" is the
  decision), a rule in `AGENTS.md`, Claude added to `authors` in `pyproject.toml` at the
  owner's instruction, README section. `NOTICE` gained the MIT notices for NVIDIA's
  SE(3)-Transformer and e3nn, which `minimal_rf/se3/` is derived from.
- verified: built metadata lists both authors; fast tests pass; grep finds no signatures in
  `src`, `tools`, `tests`. `[ran]`
- not verified: whether any other file is derived from a project missing from `NOTICE`;
  only `minimal_rf/se3/` was checked, from its own docstrings.
- found: the missing MIT notices above.
- next: unchanged from entry 1.

### 2026-10-06 claude-opus/claude-code/crimm-ml (entry 1)

- asked: split the ML part of crimm into its own repo for `pip install crimm[ml]`; prepare a
  containerized loop-building service; write handoff docs for agents.
- did: created this repo from the fork's `crimm_integration/` (S0 to S2 in ROADMAP); fork
  commit `763bbe5` added `crimm_integration/models/`, which `823fe61` had missed because the
  root `.gitignore` rule `models/` matched at any depth (now `/models/`); wrote
  `/home/ziqiaoxu/crimm/import-fix.md`; added crimm ROADMAP "Parallel track: crimm-ml
  integration"; wrote `AGENTS.md`, `CLAUDE.md`, `docs/dev/ROADMAP.md` and this file.
- verified: fork and new sampler bit-identical at T=15 and T=50 (seed 0); clean-venv wheel
  install samples with neither torch nor crimm imported; pytest 8 passed, 1 skipped (needs
  rdkit); `verify_numpy_sampler.py` all six phases pass; asset fetch from directory and HTTP,
  corrupted-file rejection; onnxruntime 1.17.3 fails to load, 1.18.0 and 1.19.2 match 1.29
  to ~1e-7 relative. `[ran]`
- not verified: GPU path from the new repo (verified in the fork before the move, and the
  code is bit-identical); Python 3.12 and 3.13; NumPy 1.x; Windows and macOS.
- found: G5, G9.
- next: owner creates the GitHub repo, then push; S5 service skeleton.
