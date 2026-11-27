# Sweep of FCT's consistency weight λ on Polyp-PVT (1288 split), seeds 1-6

Runs from `run_lambda.sh`, pushed in 92a99ba86 (branch `lambda-results`): `S3LMUP_L025` (λ = 0.25), `S3LMUP_L10` (λ = 1), `S3LMUP_L20` (λ = 2). Each was trained like the reference `S3LMUP_s1-6` runs (λ = 0.5), with only `--fct_weight` changed and `--grad_diag 50` kept. All 18 runs finished 99 epochs, and the FCT line of every log shows the intended λ with lag-or-margin, a = 2.

## Status: 11 of the 18 runs must be rerun before the official numbers count

The server's `Train_noaug.py` already had the validation-set selection of 3eb44b9e3. It scores `dataset/ValidationDataset` (162 images) and keeps the best epoch on it whenever that folder exists, unless `--val_path ""` is passed. The `run_lambda.sh` that ran was e1f078f73, which did not pass that flag; 4ebeb3fd4 added it afterwards. In an epoch selected on the validation set, the printed `mean:` is the validation score (0.88-0.93), not the average of the five test sets printed above it. That shows which rule each run used:

| Selection rule | Runs |
|---|---|
| Five-set test mean in every epoch, as the reference runs | L025_s1, L025_s2, L025_s4, L10_s1, L10_s4, L20_s1, L20_s4 |
| Switched to the validation set during training | L10_s2 (from epoch 41), L025_s5 (from epoch 86) |
| Validation set in every epoch | L025_s3, L025_s6, L10_s3, L10_s5, L10_s6, L20_s2, L20_s3, L20_s5, L20_s6 |

Training itself is the same under both rules, because `train()` puts the model back in training mode at the start of every epoch. Only the evaluation and the saved `PolypPVT.pth` differ.

`lambda_summary.txt` scores the saved checkpoints, so it mixes the two rules and is not comparable with the reference runs. It is kept as pushed and not used. For the record, its five-set means put every λ at vanilla's level: λ = 0.25 −0.0001, λ = 1 −0.0002, λ = 2 −0.0025.

## Provisional comparison: the five-set rule rebuilt from the logs

Every log prints all five test-set scores for every epoch. So the epoch each run would have kept under the five-set rule can be read off, together with its scores. These are the training-time evaluator's numbers (soft Dice per image), not `eval_dice.py`'s.

The rebuild reproduces the printed best of all 19 runs that used the five-set rule throughout: the 12 reference logs (`PVT_VANILLA_s1-6`, `S3LMUP_s1-6`) and the 7 λ runs that used it. Values are in Dice points; seeds 1-6 are paired; intervals are paired-t 95%.

| Arm | λ | Five-set mean | vs PVT_VANILLA | vs S3LMUP (λ = 0.5) |
|---|---|---|---|---|
| PVT_VANILLA | – | 86.84 ± 0.17 | | |
| S3LMUP_L025 | 0.25 | 87.22 ± 0.31 | +0.38 [+0.14, +0.62], p 0.009, 6/6 | −0.14 [−0.57, +0.29], p 0.435, 2/6 |
| S3LMUP | 0.5 | 87.36 ± 0.32 | +0.52 [+0.14, +0.90], p 0.017, 6/6 | |
| S3LMUP_L10 | 1 | 87.44 ± 0.26 | +0.61 [+0.26, +0.95], p 0.006, 6/6 | +0.09 [−0.50, +0.67], p 0.724, 2/6 |
| S3LMUP_L20 | 2 | 87.16 ± 0.38 | +0.32 [−0.14, +0.79], p 0.133, 4/6 | −0.20 [−0.74, +0.34], p 0.393, 2/6 |

ETIS against vanilla, at the same epochs:

| λ | ETIS gain |
|---|---|
| 0.25 | +1.52 [+0.40, +2.65], p 0.018, 6/6 |
| 0.5 | +1.65 [+0.87, +2.43], p 0.003, 6/6 |
| 1 | +2.67 [+1.72, +3.61], p 0.001, 6/6 |
| 2 | +1.06 [−0.09, +2.21], p 0.064, 5/6 |

Reading:
- From λ = 0.25 to λ = 1, FCT is not sensitive to λ. Each value beats vanilla on 6/6 seeds with the interval above 0, and none differs from λ = 0.5 (all p > 0.4).
- λ = 2 is weaker: its interval includes 0, on the five-set mean and on ETIS.
- For the reference arm, the rebuilt gain (+0.52 [+0.14, +0.90]) matches the official `eval_dice.py` gain (+0.0052 [+0.0015, +0.0090]).

## What selecting the epoch on the test sets adds

For each of the 11 runs selected on the validation set, the logs give the five-set mean at the saved epoch and at the best five-set epoch. Picking the epoch on the test sets adds **0.89 ± 0.45 Dice points** to the five-set mean (range 0.41 to 1.51). This is larger than FCT's gain over vanilla (0.52).

| Run | Saved epoch (validation) | Five-set at saved | Best five-set (epoch) | Difference |
|---|---|---|---|---|
| L025_s3 | 10 | 85.42 | 86.92 (34) | 1.50 |
| L025_s5 | 86 | 86.86 | 87.31 (87) | 0.45 |
| L025_s6 | 33 | 86.86 | 87.28 (76) | 0.42 |
| L10_s2 | 41 | 86.91 | 87.31 (62) | 0.41 |
| L10_s3 | 83 | 86.05 | 87.28 (75) | 1.23 |
| L10_s5 | 21 | 85.71 | 87.21 (95) | 1.51 |
| L10_s6 | 48 | 86.60 | 87.29 (54) | 0.69 |
| L20_s2 | 51 | 86.07 | 86.60 (62) | 0.52 |
| L20_s3 | 18 | 85.78 | 87.19 (11) | 1.41 |
| L20_s5 | 75 | 86.56 | 87.18 (61) | 0.62 |
| L20_s6 | 97 | 86.69 | 87.67 (53) | 0.98 |

How much the same choice adds for vanilla is not known: the vanilla logs have no validation scores. Under validation selection for every arm, FCT's gain over vanilla has not been measured.

## To finish the sweep

Rerun the 11 runs with `run_lambda.sh` at 4ebeb3fd4 or later, which passes `--val_path ""`. First move their logs, checkpoints and maps aside, so they are not skipped as finished; the 7 clean runs are kept.

```bash
git fetch origin claude/sleepy-planck-2zQ5C && git checkout FETCH_HEAD -- run_lambda.sh
mkdir -p valsel_runs/model_pth valsel_runs/result_map
for r in S3LMUP_L025_s3 S3LMUP_L025_s5 S3LMUP_L025_s6 S3LMUP_L10_s2 S3LMUP_L10_s3 S3LMUP_L10_s5 \
         S3LMUP_L10_s6 S3LMUP_L20_s2 S3LMUP_L20_s3 S3LMUP_L20_s5 S3LMUP_L20_s6; do
  mv "logs_$r.txt" valsel_runs/; mv "model_pth/$r" valsel_runs/model_pth/; mv "result_map/$r" valsel_runs/result_map/
done
DRY_RUN=1 bash run_lambda.sh 2 | grep -c -- '--val_path ""'     # 2: L10_s2 and L20_s2 are rerun
GPU=0 nohup bash run_lambda.sh 1 2 3 > lambda_rerun_gpu0.out 2>&1 &   # 5 runs
GPU=1 nohup bash run_lambda.sh 4 5 6 > lambda_rerun_gpu1.out 2>&1 &   # 6 runs
# afterwards
bash run_lambda.sh summary 2>&1 | tee lambda_summary.txt
```

At about 4.8 h per run, that is roughly 29 h on two GPUs. The reruns append new rows to `flip_gap.tsv`; for these 11 runs the later row is the valid one.
