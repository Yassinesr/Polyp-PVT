# PolypGen, single frames (C1-C6, 1,411 images): FCT and its parts on Polyp-PVT

Trained on the 1288 split; best epoch from the five-set test mean, as for every arm; meanDic
from eval_dice.py (`run_polypgen.sh`). Pooled = all 1,411 images, the image-weighted mean of
the six centres.

Order and status. The plan named FCT (S3LMUP, a = 2) against vanilla, pooled primary, centres
secondary; its maps are in this repo (commit 76a5357cc). FCT lost on C2 (n.s.). After that, the
other arms below were scored on PolypGen. They are checkpoints that already existed, so this
was inference only, done to look for a variant that also wins C2. Those comparisons are
exploratory and are reported as such. None of them wins C2. S3LMUP's pooled score is the
image-weighted mean of its 4-decimal centre scores (error < 1e-4).

## meanDic per run

```
run                    PG-C1   PG-C2   PG-C3   PG-C4   PG-C5   PG-C6  pooled
PVT_VANILLA_s1        0.8675  0.8470  0.9014  0.6753  0.6678  0.8264  0.8230
PVT_VANILLA_s2        0.8687  0.8491  0.9014  0.6766  0.6316  0.8293  0.8187
PVT_VANILLA_s3        0.8619  0.8503  0.8979  0.6727  0.6691  0.8265  0.8215
PVT_VANILLA_s4        0.8696  0.8421  0.9004  0.6873  0.6658  0.8262  0.8231
PVT_VANILLA_s5        0.8607  0.8491  0.8952  0.6908  0.6672  0.8327  0.8221
PVT_VANILLA_s6        0.8706  0.8384  0.8996  0.6971  0.6407  0.8319  0.8200
S3_s1                 0.8750  0.8375  0.9039  0.6970  0.6971  0.8371  0.8305
S3_s2                 0.8822  0.8464  0.9040  0.7020  0.6738  0.8278  0.8301
S3_s3                 0.8763  0.8473  0.9055  0.6603  0.6633  0.8399  0.8246
S3_s4                 0.8771  0.8401  0.9027  0.7002  0.6645  0.8401  0.8267
S3_s5                 0.8547  0.8370  0.8944  0.6935  0.6422  0.8441  0.8158
S3_s6                 0.8744  0.8426  0.9039  0.7007  0.6863  0.8483  0.8308
S3LM_s1               0.8782  0.8369  0.9044  0.7235  0.6813  0.8422  0.8319
S3LM_s2               0.8739  0.8482  0.9000  0.6915  0.6837  0.8365  0.8285
S3LM_s3               0.8826  0.8465  0.9014  0.6910  0.6855  0.8377  0.8305
S3LM_s4               0.8810  0.8396  0.9023  0.7171  0.6734  0.8418  0.8303
S3LM_s5               0.8805  0.8497  0.9020  0.7122  0.6892  0.8396  0.8338
S3LM_s6               0.8784  0.8394  0.9046  0.7088  0.6908  0.8552  0.8331
S3LMUP_A05_s1         0.8784  0.8475  0.9037  0.6901  0.6736  0.8424  0.8292
S3LMUP_A05_s2         0.8675  0.8456  0.9023  0.7062  0.6642  0.8504  0.8272
S3LMUP_A05_s3         0.8755  0.8462  0.9041  0.6826  0.7099  0.8374  0.8327
S3LMUP_A05_s4         0.8709  0.8539  0.9069  0.6931  0.6549  0.8528  0.8283
S3LMUP_A05_s5         0.8835  0.8445  0.9049  0.7094  0.7128  0.8394  0.8374
S3LMUP_A05_s6         0.8666  0.8333  0.9032  0.6936  0.6861  0.8334  0.8258
S3LMUP_A10_s1         0.8747  0.8519  0.9075  0.7035  0.6895  0.8376  0.8340
S3LMUP_A10_s2         0.8815  0.8477  0.9009  0.6991  0.6805  0.8302  0.8301
S3LMUP_A10_s3         0.8815  0.8338  0.9047  0.7020  0.7033  0.8464  0.8332
S3LMUP_A10_s4         0.8767  0.8405  0.9069  0.7329  0.6337  0.8465  0.8274
S3LMUP_A10_s5         0.8718  0.8530  0.9075  0.6741  0.6823  0.8401  0.8297
S3LMUP_A10_s6         0.8891  0.8429  0.9040  0.7133  0.6922  0.8227  0.8342
S3LMUP_A15_s1         0.8717  0.8482  0.9021  0.6834  0.6787  0.8367  0.8273
S3LMUP_A15_s2         0.8797  0.8442  0.9059  0.7080  0.6717  0.8292  0.8303
S3LMUP_A15_s3         0.8686  0.8490  0.9054  0.6837  0.6835  0.8494  0.8294
S3LMUP_A15_s4         0.8763  0.8340  0.9038  0.6907  0.6815  0.8472  0.8277
S3LMUP_A15_s5         0.8889  0.8434  0.9047  0.7032  0.6787  0.8410  0.8326
S3LMUP_A15_s6         0.8728  0.8470  0.9055  0.6853  0.6889  0.8363  0.8300
S3LMUP_s1             0.8696  0.8422  0.9044  0.6768  0.6927  0.8287  0.8274
S3LMUP_s2             0.8775  0.8335  0.9030  0.7116  0.6716  0.8583  0.8289
S3LMUP_s3             0.8808  0.8488  0.9047  0.7214  0.6701  0.8424  0.8329
S3LMUP_s4             0.8802  0.8432  0.9055  0.6657  0.6597  0.8476  0.8250
S3LMUP_s5             0.8763  0.8466  0.9040  0.6842  0.6972  0.8407  0.8314
S3LMUP_s6             0.8827  0.8383  0.9050  0.7137  0.6965  0.8417  0.8343
```

## Each arm against vanilla (seeds 1-6, same seed paired; seed_summary.py)

### S3

```
set       PVT_VANILLA mean+-SD      S3 mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
PG-C1        0.8665+-0.0042    0.8733+-0.0095  +0.0067 [ -0.0010, +0.0145]    0.076   0.152        5/6  no
PG-C2        0.8460+-0.0047    0.8418+-0.0044  -0.0042 [ -0.0103, +0.0020]    0.141   0.130        1/6  no
PG-C3        0.8993+-0.0024    0.9024+-0.0040  +0.0031 [ +0.0002, +0.0060]    0.041   0.147        5/6  yes
PG-C4        0.6833+-0.0099    0.6923+-0.0160  +0.0090 [ -0.0057, +0.0236]    0.176   0.277        5/6  no
PG-C5        0.6571+-0.0165    0.6712+-0.0192  +0.0141 [ -0.0162, +0.0445]    0.284   0.195        3/6  no
PG-C6        0.8288+-0.0029    0.8395+-0.0070  +0.0107 [ +0.0041, +0.0173]    0.009   0.009        5/6  yes
pooled       0.8214+-0.0018    0.8264+-0.0058  +0.0050 [ -0.0019, +0.0119]    0.121   0.067        5/6  no
```

### S3LM

```
set       PVT_VANILLA mean+-SD    S3LM mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
PG-C1        0.8665+-0.0042    0.8791+-0.0030  +0.0126 [ +0.0059, +0.0192]    0.005   0.002        6/6  yes
PG-C2        0.8460+-0.0047    0.8434+-0.0054  -0.0026 [ -0.0069, +0.0017]    0.181   0.351        2/6  no
PG-C3        0.8993+-0.0024    0.9024+-0.0018  +0.0031 [ +0.0001, +0.0061]    0.044   0.024        5/6  yes
PG-C4        0.6833+-0.0099    0.7074+-0.0134  +0.0241 [ +0.0101, +0.0381]    0.007   0.011        6/6  yes
PG-C5        0.6571+-0.0165    0.6840+-0.0063  +0.0269 [ +0.0067, +0.0472]    0.019   0.002        6/6  yes
PG-C6        0.8288+-0.0029    0.8422+-0.0068  +0.0133 [ +0.0068, +0.0199]    0.003   0.002        6/6  yes
pooled       0.8214+-0.0018    0.8314+-0.0019  +0.0099 [ +0.0077, +0.0122]    0.000   0.002        6/6  yes
```

### S3LMUP_A05

```
set       PVT_VANILLA mean+-SD S3LMUP_A05 mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
PG-C1        0.8665+-0.0042    0.8737+-0.0066  +0.0072 [ -0.0036, +0.0180]    0.147   0.054        4/6  no
PG-C2        0.8460+-0.0047    0.8452+-0.0067  -0.0008 [ -0.0076, +0.0060]    0.773   0.818        2/6  no
PG-C3        0.8993+-0.0024    0.9042+-0.0016  +0.0049 [ +0.0015, +0.0083]    0.014   0.002        6/6  yes
PG-C4        0.6833+-0.0099    0.6958+-0.0101  +0.0125 [ +0.0006, +0.0245]    0.042   0.056        5/6  yes
PG-C5        0.6571+-0.0165    0.6836+-0.0239  +0.0265 [ +0.0017, +0.0513]    0.040   0.054        5/6  yes
PG-C6        0.8288+-0.0029    0.8426+-0.0076  +0.0138 [ +0.0041, +0.0235]    0.015   0.002        6/6  yes
pooled       0.8214+-0.0018    0.8301+-0.0043  +0.0087 [ +0.0046, +0.0128]    0.003   0.002        6/6  yes
```

### S3LMUP_A10

```
set       PVT_VANILLA mean+-SD S3LMUP_A10 mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
PG-C1        0.8665+-0.0042    0.8792+-0.0061  +0.0127 [ +0.0070, +0.0183]    0.002   0.002        6/6  yes
PG-C2        0.8460+-0.0047    0.8450+-0.0073  -0.0010 [ -0.0095, +0.0076]    0.777   0.779        3/6  no
PG-C3        0.8993+-0.0024    0.9053+-0.0026  +0.0060 [ +0.0016, +0.0103]    0.017   0.006        5/6  yes
PG-C4        0.6833+-0.0099    0.7041+-0.0192  +0.0208 [ -0.0010, +0.0427]    0.058   0.039        5/6  no
PG-C5        0.6571+-0.0165    0.6803+-0.0242  +0.0232 [ -0.0090, +0.0554]    0.123   0.076        5/6  no
PG-C6        0.8288+-0.0029    0.8373+-0.0094  +0.0084 [ -0.0035, +0.0204]    0.129   0.067        5/6  no
pooled       0.8214+-0.0018    0.8314+-0.0028  +0.0100 [ +0.0063, +0.0137]    0.001   0.002        6/6  yes
```

### S3LMUP_A15

```
set       PVT_VANILLA mean+-SD S3LMUP_A15 mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
PG-C1        0.8665+-0.0042    0.8763+-0.0073  +0.0098 [ -0.0001, +0.0198]    0.052   0.009        6/6  no
PG-C2        0.8460+-0.0047    0.8443+-0.0055  -0.0017 [ -0.0080, +0.0047]    0.525   0.582        2/6  no
PG-C3        0.8993+-0.0024    0.9046+-0.0014  +0.0053 [ +0.0020, +0.0085]    0.009   0.002        6/6  yes
PG-C4        0.6833+-0.0099    0.6924+-0.0107  +0.0091 [ -0.0056, +0.0238]    0.173   0.162        5/6  no
PG-C5        0.6571+-0.0165    0.6805+-0.0057  +0.0234 [ +0.0063, +0.0406]    0.017   0.002        6/6  yes
PG-C6        0.8288+-0.0029    0.8400+-0.0075  +0.0111 [ +0.0016, +0.0207]    0.030   0.009        5/6  yes
pooled       0.8214+-0.0018    0.8295+-0.0019  +0.0081 [ +0.0049, +0.0114]    0.001   0.002        6/6  yes
```

## FCT's parts against each other (same statistics, from the scores above)

```
difference = S3LMUP - PVT_VANILLA with the same seed: FCT (a = 2), the arm the PolypGen test was planned for
set          diff              95% CI p paired  p perm   seeds won
PG-C1     +0.0113 [ +0.0053,  +0.0174]    0.005   0.006       6/6
PG-C2     -0.0039 [ -0.0103,  +0.0025]    0.177   0.212       1/6
PG-C3     +0.0051 [ +0.0024,  +0.0078]    0.005   0.002       6/6
PG-C4     +0.0123 [ -0.0154,  +0.0400]    0.306   0.240       4/6
PG-C5     +0.0243 [ -0.0003,  +0.0488]    0.052   0.043       5/6
PG-C6     +0.0144 [ +0.0042,  +0.0246]    0.015   0.009       6/6
pooled    +0.0086 [ +0.0037,  +0.0134]    0.006   0.002       6/6

difference = S3LM - S3 with the same seed: the flip pulls, on top of the three supervised views
set          diff              95% CI p paired  p perm   seeds won
PG-C1     +0.0058 [ -0.0058,  +0.0174]    0.254   0.145       5/6
PG-C2     +0.0016 [ -0.0044,  +0.0075]    0.529   0.589       2/6
PG-C3     +0.0000 [ -0.0044,  +0.0045]    0.978   0.994       3/6
PG-C4     +0.0151 [ -0.0005,  +0.0306]    0.055   0.093       5/6
PG-C5     +0.0128 [ -0.0091,  +0.0346]    0.193   0.156       5/6
PG-C6     +0.0026 [ -0.0029,  +0.0081]    0.274   0.537       4/6
pooled    +0.0049 [ -0.0023,  +0.0121]    0.139   0.048       5/6

difference = S3LMUP - S3LM with the same seed: the upright pull, on top of the flip pulls
set          diff              95% CI p paired  p perm   seeds won
PG-C1     -0.0012 [ -0.0063,  +0.0038]    0.555   0.602       2/6
PG-C2     -0.0013 [ -0.0089,  +0.0063]    0.683   0.686       3/6
PG-C3     +0.0020 [ +0.0004,  +0.0035]    0.021   0.032       5/6
PG-C4     -0.0118 [ -0.0485,  +0.0249]    0.447   0.281       3/6
PG-C5     -0.0027 [ -0.0156,  +0.0102]    0.616   0.706       3/6
PG-C6     +0.0011 [ -0.0129,  +0.0151]    0.852   0.838       4/6
pooled    -0.0014 [ -0.0047,  +0.0020]    0.342   0.424       3/6

difference = S3LMUP - S3 with the same seed: both pulls, on top of the three supervised views
set          diff              95% CI p paired  p perm   seeds won
PG-C1     +0.0046 [ -0.0058,  +0.0150]    0.310   0.368       4/6
PG-C2     +0.0003 [ -0.0080,  +0.0086]    0.933   0.924       4/6
PG-C3     +0.0020 [ -0.0021,  +0.0062]    0.264   0.236       4/6
PG-C4     +0.0033 [ -0.0319,  +0.0384]    0.820   0.768       3/6
PG-C5     +0.0101 [ -0.0139,  +0.0341]    0.328   0.335       3/6
PG-C6     +0.0037 [ -0.0114,  +0.0188]    0.558   0.470       3/6
pooled    +0.0036 [ -0.0040,  +0.0112]    0.281   0.232       3/6

```

## Reading

- Every arm with a consistency term improves pooled PolypGen on all six seeds, from +0.0081 to
  +0.0100, every 95% interval above 0. The three supervised views alone (S3) give +0.0050, with
  an interval that includes 0.
- The flip pulls on top of the views (S3LM - S3): +0.0049 pooled, positive on all six centres
  and on 5/6 seeds. Not significant: paired p 0.139, permutation p 0.048.
- The upright pull adds nothing here (S3LMUP - S3LM: pooled -0.0014, n.s.). The margin factor
  a does not matter either: pooled +0.0087 / +0.0100 / +0.0081 / +0.0086 for a = 0.5 / 1 / 1.5 / 2.
- C2 is below vanilla for every arm: -0.0042 with the views alone, -0.0026 with the flip pulls,
  -0.0008 to -0.0039 across a. No interval excludes 0. With six seeds, C2's 95% interval is
  about +-0.006 wide, so differences of this size cannot be told from noise.
