# ASGNet on the 1288 split: vanilla vs FCT (s3lmup), seeds 1-6

ASGNet (PVTv2-b3, TCSVT 2026) with its own recipe (Adam 1e-4, batch 40 as 8x5 --accum 8, 80 epochs, fp32). Trained on the re-annotated TrainDataset (1288); best epoch selected on the five-set test mean (PolypPVT.pth), as Train_noaug.py does. meanDic from eval_dice.py. Published rows are not comparable.

```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
ASGNET_VANILLA_s1           0.9086   0.9098   0.8110   0.8946   0.7826   0.8613
ASGNET_VANILLA_s2           0.9078   0.8936   0.7993   0.8970   0.7567   0.8509
ASGNET_VANILLA_s3           0.9074   0.9092   0.8129   0.8987   0.7653   0.8587
ASGNET_VANILLA_s4           0.9109   0.9126   0.7959   0.8892   0.7900   0.8597
ASGNET_VANILLA_s5           0.9097   0.9285   0.8067   0.8889   0.7807   0.8629
ASGNET_VANILLA_s6           0.9070   0.9116   0.7989   0.8955   0.7711   0.8568
ASGNET_S3LMUP_s1            0.9065   0.9156   0.8173   0.8988   0.7921   0.8661
ASGNET_S3LMUP_s2            0.9055   0.9008   0.7919   0.8970   0.7919   0.8574
ASGNET_S3LMUP_s3            0.9052   0.9070   0.8090   0.8849   0.7825   0.8577
ASGNET_S3LMUP_s4            0.9076   0.9033   0.8226   0.8983   0.7808   0.8625
ASGNET_S3LMUP_s5            0.9091   0.9200   0.8083   0.8760   0.8030   0.8633
ASGNET_S3LMUP_s6            0.9141   0.9057   0.8096   0.8824   0.7970   0.8617

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = ASGNET_S3LMUP - ASGNET_VANILLA with the same seed
set       ASGNET_VANILLA mean+-SD ASGNET_S3LMUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9086+-0.0015    0.9080+-0.0033  -0.0006 [ -0.0046, +0.0034]    0.722   0.714        1/6  no
ClinicDB     0.9109+-0.0111    0.9087+-0.0075  -0.0021 [ -0.0096, +0.0053]    0.495   0.710        2/6  no
ColonDB      0.8041+-0.0071    0.8098+-0.0104  +0.0057 [ -0.0072, +0.0185]    0.309   0.292        4/6  no
CVC-300      0.8940+-0.0041    0.8896+-0.0097  -0.0044 [ -0.0150, +0.0062]    0.336   0.318        3/6  no
ETIS         0.7744+-0.0123    0.7912+-0.0085  +0.0169 [ +0.0008, +0.0330]    0.043   0.015        5/6  yes
mean         0.8584+-0.0042    0.8615+-0.0033  +0.0031 [ +0.0000, +0.0061]    0.049   0.203        5/6  yes

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 4/5
  seed 2: won 3/5
  seed 3: won 1/5
  seed 4: won 2/5
  seed 5: won 2/5
  seed 6: won 3/5
  seed means: won 2/5
```
