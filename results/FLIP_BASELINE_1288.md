# Rotation-and-flip baseline on the 1288 split: PVT_VANILLA_3X_FLIP, seeds 1-6

Vanilla Polyp-PVT with the stock orientation augmentation, `--orientation_aug 1`. Every training image gets a random rotation within ±90°, then a horizontal flip and a vertical flip, each with probability 0.5; its mask gets the same transform (`utils/dataloader.py`).

The run name says it was trained three times as long as vanilla, as `PVT_VANILLA_3X`. Its stage and logs are not in the repo yet, so the recipe and the selection rule have not been checked here. A five-set selection run ends its log with a "best five-set mean" around 0.86-0.87.

meanDic is from `eval_dice.py` through `seed_summary.py`, seeds 1-6, paired by seed. The output below was pasted by you on 2026-11-27. Its vanilla and FCT per-run scores are identical to those recorded in DISSECTION_1288.md and COMPUTE_CONTROL_1288.md. Every block matches a recomputation from its per-run table to within rounding.

```
meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
PVT_VANILLA_s4              0.9270   0.9347   0.7990   0.8886   0.7869   0.8672
PVT_VANILLA_s5              0.9185   0.9345   0.7974   0.8991   0.7829   0.8665
PVT_VANILLA_s6              0.9164   0.9271   0.8079   0.8915   0.7908   0.8667
PVT_VANILLA_3X_FLIP_s1      0.9215   0.9444   0.7927   0.8699   0.8064   0.8670
PVT_VANILLA_3X_FLIP_s2      0.9164   0.9221   0.7993   0.8850   0.8105   0.8667
PVT_VANILLA_3X_FLIP_s3      0.9058   0.9320   0.7980   0.8770   0.7960   0.8618
PVT_VANILLA_3X_FLIP_s4      0.9027   0.9337   0.7841   0.8949   0.8027   0.8636
PVT_VANILLA_3X_FLIP_s5      0.9115   0.9336   0.7840   0.8806   0.8122   0.8644
PVT_VANILLA_3X_FLIP_s6      0.9078   0.9398   0.7997   0.8748   0.7941   0.8633

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = PVT_VANILLA_3X_FLIP - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD PVT_VANILLA_3X_FLIP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9194+-0.0059    0.9110+-0.0070  -0.0084 [ -0.0204, +0.0037]    0.134   0.056        2/6  no
ClinicDB     0.9317+-0.0064    0.9343+-0.0076  +0.0026 [ -0.0097, +0.0149]    0.611   0.571        3/6  no
ColonDB      0.8028+-0.0068    0.7930+-0.0074  -0.0098 [ -0.0149, -0.0046]    0.005   0.048        0/6  no
CVC-300      0.8879+-0.0086    0.8804+-0.0088  -0.0075 [ -0.0188, +0.0039]    0.151   0.167        2/6  no
ETIS         0.7861+-0.0029    0.8037+-0.0074  +0.0175 [ +0.0068, +0.0282]    0.008   0.002        6/6  yes
mean         0.8656+-0.0017    0.8645+-0.0020  -0.0011 [ -0.0039, +0.0017]    0.358   0.355        2/6  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 2/5
  seed 3: won 3/5
  seed 4: won 2/5
  seed 5: won 1/5
  seed 6: won 2/5
  seed means: won 2/5

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_3X_FLIP_s1      0.9215   0.9444   0.7927   0.8699   0.8064   0.8670
PVT_VANILLA_3X_FLIP_s2      0.9164   0.9221   0.7993   0.8850   0.8105   0.8667
PVT_VANILLA_3X_FLIP_s3      0.9058   0.9320   0.7980   0.8770   0.7960   0.8618
PVT_VANILLA_3X_FLIP_s4      0.9027   0.9337   0.7841   0.8949   0.8027   0.8636
PVT_VANILLA_3X_FLIP_s5      0.9115   0.9336   0.7840   0.8806   0.8122   0.8644
PVT_VANILLA_3X_FLIP_s6      0.9078   0.9398   0.7997   0.8748   0.7941   0.8633
S3LMUP_s1                   0.9202   0.9350   0.8043   0.8697   0.7976   0.8653
S3LMUP_s2                   0.9270   0.9325   0.7988   0.8827   0.8128   0.8708
S3LMUP_s3                   0.9279   0.9369   0.8101   0.8991   0.8011   0.8750
S3LMUP_s4                   0.9208   0.9309   0.8127   0.8884   0.7957   0.8697
S3LMUP_s5                   0.9302   0.9347   0.7985   0.8969   0.8008   0.8722
S3LMUP_s6                   0.9237   0.9314   0.8012   0.8940   0.8077   0.8716

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = S3LMUP - PVT_VANILLA_3X_FLIP with the same seed
set       PVT_VANILLA_3X_FLIP mean+-SD   S3LMUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9110+-0.0070    0.9250+-0.0041  +0.0140 [ +0.0052, +0.0228]    0.010   0.006        5/6  yes
ClinicDB     0.9343+-0.0076    0.9336+-0.0023  -0.0007 [ -0.0088, +0.0074]    0.829   0.842        3/6  no
ColonDB      0.7930+-0.0074    0.8043+-0.0060  +0.0113 [ +0.0003, +0.0223]    0.046   0.015        5/6  yes
CVC-300      0.8804+-0.0088    0.8885+-0.0110  +0.0081 [ -0.0050, +0.0211]    0.173   0.184        3/6  no
ETIS         0.8037+-0.0074    0.8026+-0.0065  -0.0010 [ -0.0112, +0.0091]    0.804   0.807        3/6  no
mean         0.8645+-0.0020    0.8708+-0.0032  +0.0063 [ +0.0011, +0.0115]    0.026   0.006        5/6  yes

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 1/5
  seed 2: won 3/5
  seed 3: won 5/5   GOAL MET
  seed 4: won 2/5
  seed 5: won 4/5
  seed 6: won 4/5
  seed means: won 3/5
```

## Reading

- **Against vanilla, the augmentation does not raise the five-set mean:** −0.0011 [−0.0039, +0.0017], 2/6 seeds.
  - It gains on ETIS: +0.0175 [+0.0068, +0.0282], 6/6.
  - It loses on ColonDB: −0.0098 [−0.0149, −0.0046], 0/6.
  - Kvasir (−0.0084) and CVC-300 (−0.0075) are lower too, though not significantly.
- **FCT, trained 99 epochs, beats this baseline on the five-set mean:** +0.0063 [+0.0011, +0.0115], p 0.026, permutation p 0.006, 5/6 seeds.
  - The gain comes from Kvasir (+0.0140 [+0.0052, +0.0228], 5/6) and ColonDB (+0.0113 [+0.0003, +0.0223], 5/6).
  - On ETIS the two are level: −0.0010 [−0.0112, +0.0091].
- **What this means:** the augmentation buys its ETIS gain at the cost of the other sets. FCT gets a comparable ETIS gain over vanilla (+0.0165) without losing elsewhere.
- **Flip gap** (`flip_gap.tsv`, five-set mean, six seeds): this baseline is at 7.10% ± 0.39, against FCT at 7.47% ± 0.33. FCT minus baseline is +0.37 points [+0.03, +0.70], p 0.036, with FCT higher on 5/6 seeds. Training on rotated and flipped images is enough for a low flip gap, so a low flip gap is not specific to FCT.
- **What this baseline does not isolate:** it rotates as well as flips, so it is not a flip-only control. The control with exactly FCT's three views and no consistency is S3 (DISSECTION_1288.md).
