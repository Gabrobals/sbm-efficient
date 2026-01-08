# Pre-Flight Checklist – SBM-Efficient

Complete validation checklist for experiment runs.

---

## A. Pre-run (before training)

### A1 — Repo & Environment

- [ ] On correct git branch (not dirty on main)
- [ ] `requirements.txt` installed and PyTorch imports OK
- [ ] Device selection coherent with config (cpu/cuda)
- [ ] `results/runs/` exists and is writable

**Validation**: `python -m src.experiments.preflight <config.yaml>`

### A2 — Config YAML (static validation)

- [ ] `run.task` ∈ {xor, mnist, fashion_mnist, cifar10}
- [ ] `run.model` ∈ {baseline, sbm, random_routing, static_topk}
- [ ] `run.seed` is int
- [ ] `data.batch_size` > 0
- [ ] `train.epochs` > 0
- [ ] If `run.model` is SBM-family: `sbm.*` keys exist

**SBM-specific constraints**:
- [ ] `experts_num` >= 2
- [ ] 1 <= `experts_top_k` <= `experts_num`
- [ ] `decoherence_tau.start` >= `decoherence_tau.end` > 0
- [ ] `entropy_lambda.value` >= 0

**Validation**: `python -m src.common.validate_config <config.yaml>`

### A3 — Output Folder (run_id)

- [ ] Run ID generated: `{date}_{task}_{model}_seed{seed}_{gitShortSha}`
- [ ] Run folder created: `results/runs/<run_id>/`
- [ ] Config copied to: `results/runs/<run_id>/config.yaml`
- [ ] `stdout.log` created (even if empty initially)

### A4 — Reproducibility

Seeds set for:
- [ ] Python `random`
- [ ] NumPy `np.random`
- [ ] PyTorch CPU/GPU `torch.manual_seed()`
- [ ] DataLoader workers `worker_init_fn`

Git metadata captured:
- [ ] `git.sha` (short)
- [ ] `git.dirty` (boolean)

---

## B. In-run (during training)

### B1 — Sanity Check Data

- [ ] First batch loaded without errors
- [ ] Input shape coherent with task (e.g., MNIST: N×1×28×28)
- [ ] Loss finite (not NaN/Inf) within first 2 steps

### B2 — Routing Correctness (routing models only)

For `sbm`, `random_routing`, `static_topk`:
- [ ] `active_modules_count` == K per batch (or per sample)
- [ ] `active_modules_mean` calculated and logged
- [ ] **Anti-pattern absent**: NOT computing all expert outputs then masking

### B3 — Profiling Readiness

If `profiling.enabled: true`:
- [ ] Warmup executed (`warmup_steps`)
- [ ] Timing executed (`timed_steps`)
- [ ] FLOPs counter reset at batch start
- [ ] FLOPs accumulated only on active modules

---

## C. Post-run (validation)

### C1 — metrics.json Exists and Schema Conform

**Root fields** (required):
- [ ] `run_id` (string)
- [ ] `timestamp` (ISO with timezone)
- [ ] `git.sha` (string)
- [ ] `git.dirty` (bool)
- [ ] `config_path` (string)
- [ ] `task`, `model`, `seed`

**final section** (required):
- [ ] `accuracy` (float)
- [ ] `loss` (float)
- [ ] `flops_executed` (int >= 0)
- [ ] `latency_ms` (float >= 0)
- [ ] `active_modules_mean` (float >= 0)
- [ ] `entropy_mean` (float >= 0)

**profile section** (required):
- [ ] `warmup_steps`, `timed_steps` (int)
- [ ] `latency_p50_ms`, `latency_p90_ms`, `latency_p99_ms` (float >= 0)

**Validation**: `python -m src.common.validate_metrics <run_directory>`

### C2 — Consistency Internal

- [ ] `model == baseline` ⇒ `active_modules_mean` = `experts_num` (or coherent)
- [ ] `random_routing`/`static_topk`/`sbm` ⇒ `active_modules_mean` ≈ `experts_top_k`
- [ ] If SBM: `entropy_mean` decreases over time (or doesn't grow uncontrolled) when τ decreases

### C3 — Run Valid/Invalid

**A run is INVALID if**:
- [ ] Missing even 1 mandatory metric
- [ ] FLOPs or latency not measured when `profiling.enabled=true`
- [ ] Routing computes full masked outputs (anti-pattern)
- [ ] Seed/git metadata absent

**Validation**: `python -m src.experiments.postflight <run_directory>`

---

## Validation Commands

### Pre-flight (before run)
```bash
# Validate config only
python -m src.common.validate_config configs/sbm_mnist.yaml

# Full pre-flight checks (git, environment, output dir)
python -m src.experiments.preflight configs/sbm_mnist.yaml
```

### Post-flight (after run)
```bash
# Validate completed run
python -m src.experiments.postflight results/runs/<run_id>/

# Or validate metrics directly
python -m src.common.validate_metrics results/runs/<run_id>/
```

### Validate-only mode (no training)
```bash
# Run full pre-flight without executing training
python -m src.experiments.run --config configs/sbm_mnist.yaml --validate-only
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | OK / VALID |
| 1 | Pre-flight failed |
| 2 | Config invalid |
| 3 | Metrics schema invalid |
| 4 | Consistency invalid |
| 5 | Missing files |

---

## For Opus: Implementation Order

1. ✅ `src/common/validate_config.py` — YAML schema validator
2. ✅ `src/common/validate_metrics.py` — JSON schema + consistency validator
3. ✅ `src/experiments/preflight.py` — Pre-flight orchestration
4. ✅ `src/experiments/postflight.py` — Post-flight orchestration
5. ⏳ `src/experiments/run.py` — Add `--validate-only` flag
6. ⏳ Baseline implementation — Only after validation is working

**Do not proceed to baseline implementation until all validators pass on test configs.**
