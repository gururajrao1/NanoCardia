# NanoCardia handoff — 2026-11-09 ~22:35 IST

How to continue: read this file, then `git log --oneline -5` and
`Get-Content logs/ecg_gen.log | Select-Object -Last 5`.

## Where we are

- Branch `main`, remote `origin https://github.com/gururajrao1/NanoCardia.git`.
- History: `ea2a5b2` (ECGSYN-F synthesis, CfC liquid head, loader
  wall-clock timeout) → `66c5005` (F-boost loss wiring, seq path only) →
  `8d15c84` (fboost wired into single-beat calls + regression test +
  README smoke note).
- Full suite green: `uv run python -m tests.test_upgrades` → 32/32 pass.

## The error that was fixed (09-11-2026)

- The 11/07 session's final edit failed (`no exact match found`) and the
  session ended, leaving `--fboost` half-wired: the flag existed and
  `run_once` accepted it, but both single-beat call sites in `src/train.py`
  never passed it. The main CNN silently trained with `fboost=1.0`.
- Fix: `fboost=a.fboost` on both `run_once` calls (base + weighted).
- Proof: `test_fboost_reaches_single_beat_training` failed before, passes
  after. Synthetic smoke
  (`--synthetic --epochs 15 --loss focal --ros --wcmix --fboost 2.0`):
  weighted macro **0.989** (N 0.96 / S 1.00 / V 1.00 / F 1.00),
  `F-loss boost x2.0 (F weight 2.9)`, 17,529 params.
  Synthetic F is trivially easy — the honest check is the real chain below.

## In flight — MUST RELAUNCH (does not survive session end)

The background chain dies with its session. At handoff time generation was
at ~7/45 records. To relaunch (multi-hour, unattended-safe):

```powershell
cd "C:\Users\gurur\OneDrive\Documents\heart-beat"
uv run python -m src.ode_synth --real-f --method ecgsyn --n-gen 3000 --out ecg_f_beats.npz --seed 0 2>&1 | Tee-Object -FilePath logs/ecg_gen.log | Select-Object -Last 3
uv run python -m src.train --config configs/stabilized.toml --ganf ecg_f_beats.npz --seed 0 --save ecg.pt 2>&1 | Tee-Object -FilePath logs/grid_ecg.log | Select-Object -Last 5
```

- Progress: `Get-Content logs/ecg_gen.log | Select-Object -Last 3`
  (~4 records/hour; hung records are skipped after retries via the
  wall-clock timeout — a stall is not a hang).
- Expected artifacts: `ecg_f_beats.npz`, `ecg.pt`, `logs/grid_ecg.log`.
- When it lands: record DS2 macro + per-class numbers in README, then
  commit + push. Deliberately NO `--fboost` on this run — it measures the
  ECGSYN-F data, not the loss change.

## Gaps to fill (ordered)

1. Chain DS2 numbers pending (see above).
2. F-class collapse: F (~0.8% of beats) ≈ 0.00 on real DS1→DS2 under every
   recipe. fboost unproven on real data; ODE-gated F gave 0.13 once.
3. CfC liquid head (`src/cfc.py`) unbenchmarked on real data — 24 checks
   green, synthetic smoke 0.528 macro only.
4. README still missing CfC method/numbers, ECGSYN-F method/numbers,
   loader wall-clock note, fboost real-data verdict.
5. Reproducibility: keeper 0.686 never reproduced (0.535/0.543/0.535);
   SeqECG+adv 0.680 → 0.543 identical flags+seed. Root cause unknown.
6. Val/test disconnect: RR-norm val 0.678, test ~0.50. Selection metric
   suspect.
7. Minor: `run_seq` fboost path prints nothing; ensemble of weak members
   only 0.51.

## Untracked — your decision pending

`*.pt`, `*.npz`, `logs/`, `__pycache__/` are untracked and were never
committed historically. No `.gitignore` exists (deliberately not created
without your say). Decide what belongs in the repo.

## Best trustworthy numbers (all patient-disjoint DS1→DS2, ~100k beats)

| Model | Macro | N / S / V / F | Params |
|---|---|---|---|
| Self-ONN Q=3 | 0.520 | 0.97 / 0.15 / 0.97 / 0.00 | 18,265 |
| Matched-filter init | 0.507 | 0.95 / 0.09 / 0.96 / 0.02 | 17,529 |
| Stabilized mean (3 seeds) | 0.501 ± 0.008 | F 0.00 | 17,529 |
