# Compute control on the 1288 split: vanilla Polyp-PVT trained three times as long

Stage `vanilla_3x` of run_1288.sh: 297 epochs instead of 99, so vanilla gets as many supervised passes as
FCT's three views. The learning rate is constant, as in every 99-epoch run (`--lr_decay_epoch 0`). The five
test sets are scored every 3rd epoch, so the best epoch is picked from 99 evaluations, as for every other arm.
Seeds 1-6. meanDic from eval_dice.py. Logs: logs_PVT_VANILLA_3X_s1-6.txt. The seed_summary.py output below is
vanilla3x_summary.txt as pushed in commit 759515db7.

```
meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_3X_s1           0.9163   0.9303   0.8100   0.8976   0.7788   0.8666
PVT_VANILLA_3X_s2           0.9131   0.9416   0.7969   0.8896   0.7746   0.8632
PVT_VANILLA_3X_s3           0.9187   0.9247   0.8219   0.8975   0.7763   0.8678
PVT_VANILLA_3X_s4           0.9131   0.9358   0.7981   0.8761   0.7840   0.8614
PVT_VANILLA_3X_s5           0.9202   0.9403   0.8057   0.8942   0.7950   0.8711
PVT_VANILLA_3X_s6           0.9113   0.9385   0.7976   0.8893   0.7795   0.8632
S3LMUP_s1                   0.9202   0.9350   0.8043   0.8697   0.7976   0.8653
S3LMUP_s2                   0.9270   0.9325   0.7988   0.8827   0.8128   0.8708
S3LMUP_s3                   0.9279   0.9369   0.8101   0.8991   0.8011   0.8750
S3LMUP_s4                   0.9208   0.9309   0.8127   0.8884   0.7957   0.8697
S3LMUP_s5                   0.9302   0.9347   0.7985   0.8969   0.8008   0.8722
S3LMUP_s6                   0.9237   0.9314   0.8012   0.8940   0.8077   0.8716

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = S3LMUP - PVT_VANILLA_3X with the same seed
set       PVT_VANILLA_3X mean+-SD   S3LMUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9155+-0.0035    0.9250+-0.0041  +0.0095 [ +0.0058, +0.0133]    0.001   0.004        6/6  yes
ClinicDB     0.9352+-0.0065    0.9336+-0.0023  -0.0016 [ -0.0103, +0.0070]    0.648   0.580        2/6  no
ColonDB      0.8050+-0.0098    0.8043+-0.0060  -0.0008 [ -0.0107, +0.0092]    0.851   0.879        3/6  no
CVC-300      0.8907+-0.0080    0.8885+-0.0110  -0.0022 [ -0.0169, +0.0124]    0.711   0.693        4/6  no
ETIS         0.7814+-0.0074    0.8026+-0.0065  +0.0213 [ +0.0090, +0.0335]    0.007   0.002        6/6  yes
mean         0.8656+-0.0036    0.8708+-0.0032  +0.0052 [ +0.0008, +0.0096]    0.028   0.030        5/6  yes

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 3/5
  seed 3: won 4/5
  seed 4: won 4/5
  seed 5: won 3/5
  seed 6: won 4/5
  seed means: won 2/5

##### PVT_VANILLA vs PVT_VANILLA_3X

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
PVT_VANILLA_s4              0.9270   0.9347   0.7990   0.8886   0.7869   0.8672
PVT_VANILLA_s5              0.9185   0.9345   0.7974   0.8991   0.7829   0.8665
PVT_VANILLA_s6              0.9164   0.9271   0.8079   0.8915   0.7908   0.8667
PVT_VANILLA_3X_s1           0.9163   0.9303   0.8100   0.8976   0.7788   0.8666
PVT_VANILLA_3X_s2           0.9131   0.9416   0.7969   0.8896   0.7746   0.8632
PVT_VANILLA_3X_s3           0.9187   0.9247   0.8219   0.8975   0.7763   0.8678
PVT_VANILLA_3X_s4           0.9131   0.9358   0.7981   0.8761   0.7840   0.8614
PVT_VANILLA_3X_s5           0.9202   0.9403   0.8057   0.8942   0.7950   0.8711
PVT_VANILLA_3X_s6           0.9113   0.9385   0.7976   0.8893   0.7795   0.8632

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = PVT_VANILLA_3X - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD PVT_VANILLA_3X mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9194+-0.0059    0.9155+-0.0035  -0.0039 [ -0.0102, +0.0025]    0.176   0.188        2/6  no
ClinicDB     0.9317+-0.0064    0.9352+-0.0065  +0.0035 [ -0.0014, +0.0085]    0.129   0.370        5/6  no
ColonDB      0.8028+-0.0068    0.8050+-0.0098  +0.0023 [ -0.0097, +0.0142]    0.645   0.647        3/6  no
CVC-300      0.8879+-0.0086    0.8907+-0.0080  +0.0028 [ -0.0111, +0.0168]    0.624   0.552        2/6  no
ETIS         0.7861+-0.0029    0.7814+-0.0074  -0.0048 [ -0.0140, +0.0045]    0.243   0.184        1/6  no
mean         0.8656+-0.0017    0.8656+-0.0036  -0.0000 [ -0.0048, +0.0048]    0.997   0.998        3/6  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 2/5
  seed 2: won 2/5
  seed 3: won 3/5
  seed 4: won 1/5
  seed 5: won 4/5
  seed 6: won 1/5
  seed means: won 3/5
```

## Reading

- Training three times as long gives vanilla nothing: the five-set mean is -0.0000 [-0.0048, +0.0048],
  p 0.997, up on 3 of 6 seeds. Both six-seed means are 0.8656. (The message of commit 759515db7 says
  +0.0016; its own summary file says -0.0000.) The 99-epoch vanilla runs had their best epoch between 14
  and 79, so vanilla had stopped improving well before epoch 99.
- FCT beats the compute-matched vanilla by +0.0052 [+0.0008, +0.0096] on the five-set mean (p 0.028, 5/6
  seeds), +0.0213 [+0.0090, +0.0335] on ETIS (6/6) and +0.0095 [+0.0058, +0.0133] on Kvasir (6/6).
  ClinicDB, ColonDB and CVC-300 are within noise. Seed 1 is the only seed where vanilla_3x is ahead on the
  mean (by 0.0013).
- So the gain of FCT is not the extra training its three views give: three times the training changes
  vanilla by 0.00 points, while FCT changes it by +0.52.
