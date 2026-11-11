# FCT component dissection on the 1288 split (Polyp-PVT, run_dissection.sh, GPU 0)

Trained on the re-annotated TrainDataset (1288); best epoch selected on the five-set test mean (PolypPVT.pth), as Train_noaug.py does. meanDic from eval_dice.py. Published rows are not comparable.

- s3: seeds 1-6 done. s3up: seeds 1-6 done (re-run after the fix to the crash below). s3fup: seeds 1-5 done, seed 6 not finished yet. s3relup: seeds 1-3.
- s3up's first attempt crashed on all six seeds at the first log line (Train_noaug.py:665, `cons_record.show()` on an empty meter: with --fct_flip_pull 0 no flip-pull loss is recorded); fixed and re-run.

## PVT_VANILLA -> S3, seeds 1 2 3 4 5 6
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
PVT_VANILLA_s4              0.9270   0.9347   0.7990   0.8886   0.7869   0.8672
PVT_VANILLA_s5              0.9185   0.9345   0.7974   0.8991   0.7829   0.8665
PVT_VANILLA_s6              0.9164   0.9271   0.8079   0.8915   0.7908   0.8667
S3_s1                       0.9275   0.9271   0.8033   0.8843   0.8033   0.8691
S3_s2                       0.9228   0.9298   0.8077   0.8956   0.8008   0.8713
S3_s3                       0.9155   0.9329   0.8093   0.8849   0.8095   0.8704
S3_s4                       0.9246   0.9224   0.8045   0.8879   0.8109   0.8701
S3_s5                       0.9193   0.9171   0.8123   0.8929   0.7763   0.8636
S3_s6                       0.9274   0.9323   0.8106   0.8787   0.8050   0.8708

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = S3 - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD       S3 mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9194+-0.0059    0.9228+-0.0047  +0.0035 [ -0.0051, +0.0121]    0.344   0.290        4/6  no
ClinicDB     0.9317+-0.0064    0.9269+-0.0062  -0.0048 [ -0.0163, +0.0068]    0.338   0.221        2/6  no
ColonDB      0.8028+-0.0068    0.8079+-0.0035  +0.0052 [ -0.0012, +0.0115]    0.090   0.130        4/6  no
CVC-300      0.8879+-0.0086    0.8874+-0.0062  -0.0005 [ -0.0095, +0.0085]    0.900   0.913        2/6  no
ETIS         0.7861+-0.0029    0.8010+-0.0127  +0.0148 [ +0.0032, +0.0265]    0.022   0.019        5/6  yes
mean         0.8656+-0.0017    0.8692+-0.0029  +0.0037 [ -0.0001, +0.0074]    0.052   0.026        5/6  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 3/5
  seed 3: won 3/5
  seed 4: won 2/5
  seed 5: won 2/5
  seed 6: won 4/5
  seed means: won 3/5
```

## S3 -> S3LMUP, seeds 1 2 3 4 5 6
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
S3_s1                       0.9275   0.9271   0.8033   0.8843   0.8033   0.8691
S3_s2                       0.9228   0.9298   0.8077   0.8956   0.8008   0.8713
S3_s3                       0.9155   0.9329   0.8093   0.8849   0.8095   0.8704
S3_s4                       0.9246   0.9224   0.8045   0.8879   0.8109   0.8701
S3_s5                       0.9193   0.9171   0.8123   0.8929   0.7763   0.8636
S3_s6                       0.9274   0.9323   0.8106   0.8787   0.8050   0.8708
S3LMUP_s1                   0.9202   0.9350   0.8043   0.8697   0.7976   0.8653
S3LMUP_s2                   0.9270   0.9325   0.7988   0.8827   0.8128   0.8708
S3LMUP_s3                   0.9279   0.9369   0.8101   0.8991   0.8011   0.8750
S3LMUP_s4                   0.9208   0.9309   0.8127   0.8884   0.7957   0.8697
S3LMUP_s5                   0.9302   0.9347   0.7985   0.8969   0.8008   0.8722
S3LMUP_s6                   0.9237   0.9314   0.8012   0.8940   0.8077   0.8716

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = S3LMUP - S3 with the same seed
set             S3 mean+-SD   S3LMUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9228+-0.0047    0.9250+-0.0041  +0.0021 [ -0.0066, +0.0108]    0.556   0.407        3/6  no
ClinicDB     0.9269+-0.0062    0.9336+-0.0023  +0.0066 [ -0.0001, +0.0134]    0.053   0.024        5/6  no
ColonDB      0.8079+-0.0035    0.8043+-0.0060  -0.0037 [ -0.0124, +0.0050]    0.328   0.225        3/6  no
CVC-300      0.8874+-0.0062    0.8885+-0.0110  +0.0011 [ -0.0124, +0.0145]    0.848   0.840        4/6  no
ETIS         0.8010+-0.0127    0.8026+-0.0065  +0.0017 [ -0.0137, +0.0170]    0.793   0.838        3/6  no
mean         0.8692+-0.0029    0.8708+-0.0032  +0.0016 [ -0.0030, +0.0062]    0.424   0.374        3/6  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 2/5
  seed 2: won 3/5
  seed 3: won 4/5
  seed 4: won 3/5
  seed 5: won 4/5
  seed 6: won 2/5
  seed means: won 4/5
```

## PVT_VANILLA -> S3LMUP, seeds 1 2 3 4 5 6
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
PVT_VANILLA_s4              0.9270   0.9347   0.7990   0.8886   0.7869   0.8672
PVT_VANILLA_s5              0.9185   0.9345   0.7974   0.8991   0.7829   0.8665
PVT_VANILLA_s6              0.9164   0.9271   0.8079   0.8915   0.7908   0.8667
S3LMUP_s1                   0.9202   0.9350   0.8043   0.8697   0.7976   0.8653
S3LMUP_s2                   0.9270   0.9325   0.7988   0.8827   0.8128   0.8708
S3LMUP_s3                   0.9279   0.9369   0.8101   0.8991   0.8011   0.8750
S3LMUP_s4                   0.9208   0.9309   0.8127   0.8884   0.7957   0.8697
S3LMUP_s5                   0.9302   0.9347   0.7985   0.8969   0.8008   0.8722
S3LMUP_s6                   0.9237   0.9314   0.8012   0.8940   0.8077   0.8716

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = S3LMUP - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD   S3LMUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9194+-0.0059    0.9250+-0.0041  +0.0056 [ -0.0031, +0.0144]    0.159   0.084        4/6  no
ClinicDB     0.9317+-0.0064    0.9336+-0.0023  +0.0019 [ -0.0064, +0.0101]    0.585   0.537        4/6  no
ColonDB      0.8028+-0.0068    0.8043+-0.0060  +0.0015 [ -0.0080, +0.0110]    0.698   0.684        3/6  no
CVC-300      0.8879+-0.0086    0.8885+-0.0110  +0.0006 [ -0.0139, +0.0151]    0.920   0.916        2/6  no
ETIS         0.7861+-0.0029    0.8026+-0.0065  +0.0165 [ +0.0088, +0.0242]    0.003   0.002        6/6  yes
mean         0.8656+-0.0017    0.8708+-0.0032  +0.0052 [ +0.0015, +0.0090]    0.016   0.011        6/6  yes

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 2/5
  seed 3: won 4/5
  seed 4: won 2/5
  seed 5: won 4/5
  seed 6: won 4/5
  seed means: won 5/5   GOAL MET
```

## PVT_VANILLA -> S3FUP, seeds 1 2 3
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
S3FUP_s1                    0.9259   0.9307   0.8108   0.8798   0.7919   0.8678
S3FUP_s2                    0.9213   0.9182   0.8064   0.8885   0.7936   0.8656
S3FUP_s3                    0.9238   0.9175   0.7921   0.8967   0.8149   0.8690

seeds [1, 2, 3] (n = 3); difference = S3FUP - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD    S3FUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9181+-0.0072    0.9237+-0.0023  +0.0056 [ -0.0084, +0.0196]    0.227   0.200        2/3  no
ClinicDB     0.9313+-0.0092    0.9222+-0.0074  -0.0091 [ -0.0358, +0.0175]    0.278   0.200        0/3  no
ColonDB      0.8041+-0.0088    0.8031+-0.0098  -0.0010 [ -0.0444, +0.0425]    0.933   0.800        1/3  no
CVC-300      0.8827+-0.0086    0.8883+-0.0084  +0.0057 [ -0.0327, +0.0441]    0.590   0.500        1/3  no
ETIS         0.7854+-0.0021    0.8001+-0.0128  +0.0147 [ -0.0140, +0.0434]    0.158   0.100        3/3  no
mean         0.8643+-0.0016    0.8675+-0.0017  +0.0032 [ -0.0050, +0.0114]    0.237   0.200        2/3  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 2/5
  seed 3: won 2/5
  seed means: won 3/5
```

## PVT_VANILLA -> S3RELUP, seeds 1 2 3
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
S3RELUP_s1                  0.9212   0.9401   0.8052   0.8886   0.7966   0.8703
S3RELUP_s2                  0.9146   0.9293   0.8039   0.8952   0.7958   0.8678
S3RELUP_s3                  0.9211   0.9114   0.8158   0.8815   0.7920   0.8644

seeds [1, 2, 3] (n = 3); difference = S3RELUP - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD  S3RELUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9181+-0.0072    0.9190+-0.0038  +0.0009 [ -0.0081, +0.0099]    0.706   0.800        2/3  no
ClinicDB     0.9313+-0.0092    0.9269+-0.0145  -0.0044 [ -0.0296, +0.0208]    0.531   0.700        1/3  no
ColonDB      0.8041+-0.0088    0.8083+-0.0065  +0.0043 [ -0.0146, +0.0232]    0.433   0.600        2/3  no
CVC-300      0.8827+-0.0086    0.8884+-0.0068  +0.0058 [ +0.0002, +0.0114]    0.047   0.500        3/3  yes
ETIS         0.7854+-0.0021    0.7948+-0.0025  +0.0094 [ -0.0007, +0.0194]    0.057   0.100        3/3  no
mean         0.8643+-0.0016    0.8675+-0.0030  +0.0032 [ -0.0045, +0.0109]    0.216   0.200        3/3  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 5/5   GOAL MET
  seed 2: won 3/5
  seed 3: won 3/5
  seed means: won 4/5
```

## S3LMUP -> S3FUP, seeds 1 2 3
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
S3LMUP_s1                   0.9202   0.9350   0.8043   0.8697   0.7976   0.8653
S3LMUP_s2                   0.9270   0.9325   0.7988   0.8827   0.8128   0.8708
S3LMUP_s3                   0.9279   0.9369   0.8101   0.8991   0.8011   0.8750
S3FUP_s1                    0.9259   0.9307   0.8108   0.8798   0.7919   0.8678
S3FUP_s2                    0.9213   0.9182   0.8064   0.8885   0.7936   0.8656
S3FUP_s3                    0.9238   0.9175   0.7921   0.8967   0.8149   0.8690

seeds [1, 2, 3] (n = 3); difference = S3FUP - S3LMUP with the same seed
set         S3LMUP mean+-SD    S3FUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9250+-0.0042    0.9237+-0.0023  -0.0014 [ -0.0167, +0.0140]    0.736   0.600        1/3  no
ClinicDB     0.9348+-0.0022    0.9222+-0.0074  -0.0126 [ -0.0317, +0.0065]    0.105   0.100        0/3  no
ColonDB      0.8044+-0.0056    0.8031+-0.0098  -0.0013 [ -0.0372, +0.0346]    0.893   0.900        2/3  no
CVC-300      0.8838+-0.0147    0.8883+-0.0084  +0.0045 [ -0.0112, +0.0203]    0.343   0.800        2/3  no
ETIS         0.8038+-0.0080    0.8001+-0.0128  -0.0037 [ -0.0448, +0.0374]    0.736   0.600        1/3  no
mean         0.8704+-0.0048    0.8675+-0.0017  -0.0029 [ -0.0145, +0.0087]    0.396   0.500        1/3  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 2/5
  seed 3: won 1/5
  seed means: won 1/5
```

## S3LMUP -> S3RELUP, seeds 1 2 3
```

meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
S3LMUP_s1                   0.9202   0.9350   0.8043   0.8697   0.7976   0.8653
S3LMUP_s2                   0.9270   0.9325   0.7988   0.8827   0.8128   0.8708
S3LMUP_s3                   0.9279   0.9369   0.8101   0.8991   0.8011   0.8750
S3RELUP_s1                  0.9212   0.9401   0.8052   0.8886   0.7966   0.8703
S3RELUP_s2                  0.9146   0.9293   0.8039   0.8952   0.7958   0.8678
S3RELUP_s3                  0.9211   0.9114   0.8158   0.8815   0.7920   0.8644

seeds [1, 2, 3] (n = 3); difference = S3RELUP - S3LMUP with the same seed
set         S3LMUP mean+-SD  S3RELUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9250+-0.0042    0.9190+-0.0038  -0.0061 [ -0.0228, +0.0107]    0.260   0.300        1/3  no
ClinicDB     0.9348+-0.0022    0.9269+-0.0145  -0.0079 [ -0.0471, +0.0314]    0.479   0.500        1/3  no
ColonDB      0.8044+-0.0056    0.8083+-0.0065  +0.0040 [ -0.0026, +0.0105]    0.121   0.500        3/3  no
CVC-300      0.8838+-0.0147    0.8884+-0.0068  +0.0046 [ -0.0437, +0.0529]    0.722   0.700        2/3  no
ETIS         0.8038+-0.0080    0.7948+-0.0025  -0.0090 [ -0.0289, +0.0109]    0.190   0.100        0/3  no
mean         0.8704+-0.0048    0.8675+-0.0030  -0.0029 [ -0.0223, +0.0165]    0.588   0.400        1/3  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 4/5
  seed 2: won 2/5
  seed 3: won 1/5
  seed means: won 2/5
```

## S3 -> S3UP, seeds 1 2 3 4 5 6
```
meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
S3_s1                       0.9275   0.9271   0.8033   0.8843   0.8033   0.8691
S3_s2                       0.9228   0.9298   0.8077   0.8956   0.8008   0.8713
S3_s3                       0.9155   0.9329   0.8093   0.8849   0.8095   0.8704
S3_s4                       0.9246   0.9224   0.8045   0.8879   0.8109   0.8701
S3_s5                       0.9193   0.9171   0.8123   0.8929   0.7763   0.8636
S3_s6                       0.9274   0.9323   0.8106   0.8787   0.8050   0.8708
S3UP_s1                     0.9276   0.9341   0.8015   0.8836   0.7914   0.8676
S3UP_s2                     0.9276   0.9270   0.8111   0.8804   0.7910   0.8674
S3UP_s3                     0.9301   0.9289   0.7966   0.8910   0.7882   0.8670
S3UP_s4                     0.9157   0.9359   0.7983   0.8951   0.8034   0.8697
S3UP_s5                     0.9209   0.9181   0.8003   0.8902   0.7930   0.8645
S3UP_s6                     0.9231   0.9343   0.8061   0.8923   0.8126   0.8737

seeds [1, 2, 3, 4, 5, 6] (n = 6); difference = S3UP - S3 with the same seed
set             S3 mean+-SD     S3UP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9228+-0.0047    0.9241+-0.0053  +0.0013 [ -0.0072, +0.0098]    0.708   0.636        4/6  no
ClinicDB     0.9269+-0.0062    0.9297+-0.0067  +0.0028 [ -0.0041, +0.0096]    0.346   0.455        4/6  no
ColonDB      0.8079+-0.0035    0.8023+-0.0054  -0.0056 [ -0.0121, +0.0008]    0.075   0.067        1/6  no
CVC-300      0.8874+-0.0062    0.8888+-0.0056  +0.0014 [ -0.0092, +0.0119]    0.754   0.695        3/6  no
ETIS         0.8010+-0.0127    0.7966+-0.0094  -0.0044 [ -0.0190, +0.0103]    0.477   0.517        2/6  no
mean         0.8692+-0.0029    0.8683+-0.0031  -0.0009 [ -0.0037, +0.0018]    0.434   0.610        2/6  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 2/5
  seed 2: won 2/5
  seed 3: won 2/5
  seed 4: won 2/5
  seed 5: won 3/5
  seed 6: won 3/5
  seed means: won 3/5
```

## The whole ladder, and the comparisons not printed by seed_summary.py

Mean meanDic over the six seeds (points):

```
               Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA     91.94    93.17    80.28    88.79    78.61    86.55
S3              92.28    92.69    80.80    88.74    80.10    86.92
S3UP            92.42    92.97    80.23    88.88    79.66    86.83
S3LM            92.34    93.04    80.32    88.85    80.58    87.02
S3LMUP          92.50    93.36    80.43    88.85    80.26    87.08
```

Computed here with seed_summary.py's statistics from the 4-decimal per-run scores in this file
(S3LM's from seeds_s3lm.out), so a last digit can differ by 1 from its own output:

```
difference = S3UP - PVT_VANILLA with the same seed: the three views and the pull to the best flipped view, against vanilla
set          diff              95% CI p paired  p perm   seeds won
Kvasir    +0.0048 [ -0.0050,  +0.0147]    0.264   0.175       5/6
ClinicDB  -0.0020 [ -0.0127,  +0.0087]    0.653   0.600       4/6
ColonDB   -0.0004 [ -0.0081,  +0.0072]    0.890   0.913       3/6
CVC-300   +0.0009 [ -0.0098,  +0.0116]    0.838   0.844       3/6
ETIS      +0.0105 [ +0.0025,  +0.0184]    0.019   0.004       6/6
mean      +0.0028 [ -0.0004,  +0.0059]    0.074   0.078       5/6

difference = S3LMUP - S3UP with the same seed: the pull to the upright view, on top of the pull to the best flipped view
set          diff              95% CI p paired  p perm   seeds won
Kvasir    +0.0008 [ -0.0053,  +0.0069]    0.750   0.790       3/6
ClinicDB  +0.0038 [ -0.0045,  +0.0122]    0.288   0.221       4/6
ColonDB   +0.0019 [ -0.0091,  +0.0130]    0.669   0.563       3/6
CVC-300   -0.0003 [ -0.0092,  +0.0086]    0.934   0.961       4/6
ETIS      +0.0060 [ -0.0055,  +0.0176]    0.239   0.221       4/6
mean      +0.0024 [ -0.0024,  +0.0073]    0.254   0.212       3/6

difference = S3LM - S3 with the same seed: the pull to the upright view alone, on top of the three views
set          diff              95% CI p paired  p perm   seeds won
Kvasir    +0.0005 [ -0.0077,  +0.0088]    0.878   0.855       3/6
ClinicDB  +0.0035 [ -0.0057,  +0.0126]    0.376   0.364       4/6
ColonDB   -0.0047 [ -0.0141,  +0.0046]    0.248   0.165       2/6
CVC-300   +0.0011 [ -0.0086,  +0.0108]    0.788   0.762       3/6
ETIS      +0.0049 [ -0.0124,  +0.0221]    0.502   0.513       2/6
mean      +0.0010 [ -0.0045,  +0.0066]    0.652   0.619       3/6

difference = S3LMUP - S3LM with the same seed: the pull to the best flipped view, on top of the pull to the upright view
set          diff              95% CI p paired  p perm   seeds won
Kvasir    +0.0016 [ -0.0059,  +0.0091]    0.606   0.545       3/6
ClinicDB  +0.0032 [ -0.0051,  +0.0114]    0.371   0.372       4/6
ColonDB   +0.0011 [ -0.0070,  +0.0091]    0.747   0.807       3/6
CVC-300   +0.0000 [ -0.0086,  +0.0086]    0.996   1.000       3/6
ETIS      -0.0032 [ -0.0167,  +0.0103]    0.569   0.405       2/6
mean      +0.0005 [ -0.0037,  +0.0047]    0.764   0.812       3/6
```

Terms. Pull to the upright view: the flipped views move toward the upright view's prediction, fully
where the upright view is more accurate, partly where it is less accurate but still right, not at all
where it is wrong (lag-or-margin, a = 2). Pull to the best flipped view: the upright view moves toward
the flipped view closest to the label, where that view is more accurate. In the code and run names:
flip pulls (--fct_flip_pull) and the upright pull (--fct_up, the UP in S3UP and S3LMUP).

Reading. The pull to the best flipped view alone adds nothing to the three views (S3UP - S3: -0.0009,
n.s.). Neither pull on its own separates from the views (pull to the upright view +0.0010, n.s.), nor
does either one on top of the other (pull to the upright view +0.0024, pull to the best flipped view
+0.0005, both n.s.). FCT, with both, has the highest
five-set mean of the ladder. It is also the only arm of the ladder whose five-set gain over vanilla
has a 95% interval above 0: +0.0052, p 0.016, 6/6 seeds (section PVT_VANILLA -> S3LMUP above).
S3LM comes closest: +0.0047, interval [-0.0002, +0.0096], p 0.058 (seeds_s3lm.out).

## PVT_VANILLA -> S3FUP, seeds 1 2 3 4 5 (seed 6 not finished)
```
meanDic per run             Kvasir ClinicDB  ColonDB  CVC-300     ETIS     mean
PVT_VANILLA_s1              0.9204   0.9327   0.7941   0.8847   0.7860   0.8636
PVT_VANILLA_s2              0.9100   0.9397   0.8078   0.8900   0.7831   0.8661
PVT_VANILLA_s3              0.9238   0.9215   0.8103   0.8733   0.7871   0.8632
PVT_VANILLA_s4              0.9270   0.9347   0.7990   0.8886   0.7869   0.8672
PVT_VANILLA_s5              0.9185   0.9345   0.7974   0.8991   0.7829   0.8665
PVT_VANILLA_s6              0.9164   0.9271   0.8079   0.8915   0.7908   0.8667
S3FUP_s1                    0.9259   0.9307   0.8108   0.8798   0.7919   0.8678
S3FUP_s2                    0.9213   0.9182   0.8064   0.8885   0.7936   0.8656
S3FUP_s3                    0.9238   0.9175   0.7921   0.8967   0.8149   0.8690
S3FUP_s4                    0.9270   0.9344   0.8143   0.8820   0.8096   0.8735
S3FUP_s5                    0.9245   0.9207   0.8032   0.8912   0.7952   0.8670
S3FUP_s6                    missing or incomplete

!! seeds without both runs (left out): [6]

seeds [1, 2, 3, 4, 5] (n = 5); difference = S3FUP - PVT_VANILLA with the same seed
set       PVT_VANILLA mean+-SD    S3FUP mean+-SD     diff              95% CI p paired  p perm   seeds won  improved?
Kvasir       0.9199+-0.0064    0.9245+-0.0022  +0.0046 [ -0.0013, +0.0104]    0.097   0.151        3/5  no
ClinicDB     0.9326+-0.0068    0.9243+-0.0077  -0.0083 [ -0.0195, +0.0029]    0.109   0.071        0/5  no
ColonDB      0.8017+-0.0070    0.8054+-0.0085  +0.0036 [ -0.0141, +0.0214]    0.599   0.468        3/5  no
CVC-300      0.8871+-0.0094    0.8876+-0.0069  +0.0005 [ -0.0157, +0.0167]    0.937   0.944        1/5  no
ETIS         0.7852+-0.0021    0.8010+-0.0105  +0.0158 [ +0.0046, +0.0271]    0.018   0.008        5/5  yes
mean         0.8653+-0.0018    0.8686+-0.0030  +0.0032 [ -0.0006, +0.0071]    0.079   0.071        4/5  no

your goal (all 5 sets won, one by >= 0.001):
  seed 1: won 3/5
  seed 2: won 2/5
  seed 3: won 2/5
  seed 4: won 2/5
  seed 5: won 3/5
  seed means: won 4/5
```
