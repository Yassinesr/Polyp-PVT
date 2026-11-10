# PolypGen per image: FCT (S3LMUP) against vanilla, Polyp-PVT, seeds 1-6

From the PolypGen maps of PVT_VANILLA_s1-6 and S3LMUP_s1-6 in result_map (commit 76a5357cc), scored per image with eval_dice.py's measures. Gaps are FCT minus vanilla in meanDic; p-values are exact over the 924 ways to split the 12 runs. The C2 section is the analysis of FCT's one losing centre.

```
PG-C1: 251 images
PG-C2: 270 images
PG-C3: 455 images
PG-C4: 146 images
PG-C5: 206 images
PG-C6: 83 images

1. SCORES  meanDic (mean Dice over 256 thresholds, as in the Polyp-PVT paper)
                     PG-C1    PG-C2    PG-C3    PG-C4    PG-C5    PG-C6     mean
  PVT_VANILLA_s1    0.8675   0.8470   0.9014   0.6753   0.6678   0.8264   0.7976
  PVT_VANILLA_s2    0.8687   0.8491   0.9014   0.6766   0.6316   0.8293   0.7928
  PVT_VANILLA_s3    0.8619   0.8503   0.8979   0.6727   0.6691   0.8265   0.7964
  PVT_VANILLA_s4    0.8696   0.8421   0.9004   0.6873   0.6658   0.8262   0.7986
  PVT_VANILLA_s5    0.8607   0.8491   0.8952   0.6908   0.6672   0.8327   0.7993
  PVT_VANILLA_s6    0.8706   0.8384   0.8996   0.6971   0.6407   0.8319   0.7964
  S3LMUP_s1         0.8696   0.8422   0.9044   0.6768   0.6927   0.8287   0.8024
  S3LMUP_s2         0.8775   0.8335   0.9030   0.7116   0.6716   0.8583   0.8092
  S3LMUP_s3         0.8808   0.8488   0.9047   0.7214   0.6701   0.8424   0.8114
  S3LMUP_s4         0.8802   0.8432   0.9055   0.6657   0.6597   0.8476   0.8003
  S3LMUP_s5         0.8763   0.8466   0.9040   0.6842   0.6972   0.8407   0.8081
  S3LMUP_s6         0.8827   0.8383   0.9050   0.7137   0.6965   0.8417   0.8130

2-3. FCT (6 runs) vs vanilla (6 runs) on meandic; p-values are exact over 924 labellings (smallest possible p 0.001)

== PG-C1  (251 images; vanilla 0.8665, FCT 0.8778, gap +0.0113, permutation p 0.006)
  from: 10 worst images -0.0058, 10 best images +0.0104, the other 231 +0.0067
  by polyp size (quarters of the set): smallest +0.0172 (p 0.058), small +0.0119 (p 0.050), large +0.0014 (p 0.864), largest +0.0148 (p 0.006)
  images where every FCT run < every vanilla run: 3 (p 0.010);  every FCT run > every vanilla run: 4 (p 0.003)
  at 0.5: missed polyp area 0.088 vs 0.081, false-positive area 0.451 vs 0.332 (vanilla vs FCT, fraction of polyp area); complete misses per run 5.0 vs 3.8

== PG-C2  (270 images; vanilla 0.8460, FCT 0.8421, gap -0.0039, permutation p 0.212)
  from: 10 worst images -0.0095, 10 best images +0.0071, the other 250 -0.0015
  by polyp size (quarters of the set): smallest -0.0191 (p 0.048), small -0.0009 (p 0.874), large -0.0036 (p 0.450), largest +0.0082 (p 0.195)
  images where every FCT run < every vanilla run: 3 (p 0.006);  every FCT run > every vanilla run: 7 (p 0.001)
  at 0.5: missed polyp area 0.114 vs 0.110, false-positive area 356.422 vs 473.129 (vanilla vs FCT, fraction of polyp area); complete misses per run 6.5 vs 7.8
  10 images where FCT loses most (FCT mean - vanilla mean):
    EndoCV2021_001188        polyp   1.6%  vanilla 0.603 0.899 0.831 0.833 0.384 0.652  FCT 0.223 (0.000-0.885)  below every vanilla run in 5/6 FCT runs  FCT misses 3  FCT error: false pos
    EndoCV2021_001073        polyp   4.6%  vanilla 0.920 0.911 0.922 0.917 0.923 0.923  FCT 0.598 (0.216-0.931)  below every vanilla run in 4/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001321        polyp   4.9%  vanilla 0.718 0.438 0.879 0.433 0.773 0.647  FCT 0.361 (0.109-0.648)  below every vanilla run in 5/6 FCT runs  FCT misses 0  FCT error: missed
    EndoCV2021_001039        polyp   1.8%  vanilla 0.677 0.939 0.911 0.933 0.231 0.349  FCT 0.403 (0.242-0.803)  below every vanilla run in 0/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001091        polyp   7.5%  vanilla 0.804 0.347 0.934 0.741 0.854 0.777  FCT 0.489 (0.348-0.807)  below every vanilla run in 0/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001404        polyp   2.2%  vanilla 0.436 0.931 0.928 0.285 0.890 0.894  FCT 0.500 (0.335-0.855)  below every vanilla run in 0/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001041        polyp   1.1%  vanilla 0.915 0.877 0.865 0.926 0.926 0.909  FCT 0.703 (0.211-0.935)  below every vanilla run in 3/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001117        polyp   1.8%  vanilla 0.739 0.743 0.720 0.746 0.729 0.718  FCT 0.535 (0.140-0.740)  below every vanilla run in 4/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001406        polyp   0.4%  vanilla 0.875 0.458 0.866 0.779 0.772 0.758  FCT 0.586 (0.436-0.811)  below every vanilla run in 1/6 FCT runs  FCT misses 0  FCT error: false pos
    EndoCV2021_001426        polyp   2.1%  vanilla 0.932 0.866 0.948 0.939 0.710 0.907  FCT 0.732 (0.566-0.933)  below every vanilla run in 3/6 FCT runs  FCT misses 0  FCT error: false pos

== PG-C3  (455 images; vanilla 0.8993, FCT 0.9044, gap +0.0051, permutation p 0.002)
  from: 10 worst images -0.0021, 10 best images +0.0025, the other 435 +0.0047
  by polyp size (quarters of the set): smallest +0.0113 (p 0.009), small +0.0051 (p 0.032), large +0.0011 (p 0.630), largest +0.0029 (p 0.143)
  images where every FCT run < every vanilla run: 2 (p 0.097);  every FCT run > every vanilla run: 13 (p 0.001)
  at 0.5: missed polyp area 0.073 vs 0.075, false-positive area 0.259 vs 0.222 (vanilla vs FCT, fraction of polyp area); complete misses per run 2.2 vs 2.2

== PG-C4  (146 images; vanilla 0.6833, FCT 0.6956, gap +0.0123, permutation p 0.240)
  from: 10 worst images -0.0137, 10 best images +0.0194, the other 126 +0.0065
  by polyp size (quarters of the set): smallest +0.0084 (p 0.745), small +0.0174 (p 0.190), large +0.0244 (p 0.024), largest -0.0006 (p 0.963)
  images where every FCT run < every vanilla run: 1 (p 0.133);  every FCT run > every vanilla run: 5 (p 0.001)
  at 0.5: missed polyp area 0.150 vs 0.140, false-positive area 1.835 vs 1.914 (vanilla vs FCT, fraction of polyp area); complete misses per run 11.0 vs 10.2

== PG-C5  (206 images; vanilla 0.6571, FCT 0.6813, gap +0.0242, permutation p 0.043)
  from: 10 worst images -0.0086, 10 best images +0.0148, the other 186 +0.0181
  by polyp size (quarters of the set): smallest +0.0641 (p 0.009), small +0.0093 (p 0.502), large +0.0160 (p 0.076), largest +0.0071 (p 0.537)
  images where every FCT run < every vanilla run: 0 (p 1.000);  every FCT run > every vanilla run: 3 (p 0.002)
  at 0.5: missed polyp area 0.257 vs 0.250, false-positive area 1.280 vs 1.315 (vanilla vs FCT, fraction of polyp area); complete misses per run 22.5 vs 22.0

== PG-C6  (83 images; vanilla 0.8288, FCT 0.8432, gap +0.0144, permutation p 0.009)
  from: 10 worst images -0.0025, 10 best images +0.0124, the other 63 +0.0045
  by polyp size (quarters of the set): smallest +0.0311 (p 0.093), small +0.0126 (p 0.019), large +0.0142 (p 0.004), largest -0.0004 (p 0.924)
  images where every FCT run < every vanilla run: 0 (p 1.000);  every FCT run > every vanilla run: 2 (p 0.003)
  at 0.5: missed polyp area 0.137 vs 0.124, false-positive area 1.402 vs 1.333 (vanilla vs FCT, fraction of polyp area); complete misses per run 5.3 vs 4.8

4. BAR  S3LMUP_s1 - PVT_VANILLA_s1 per set, and the 5 images it loses most on
  PG-C1: +0.0021  = 10 worst images -0.0121, 10 best +0.0098, other 231 +0.0044;  complete misses 6 vs 6
  PG-C2: -0.0047  = 10 worst images -0.0141, 10 best +0.0165, other 250 -0.0071;  complete misses 6 vs 6
      EndoCV2021_001166        polyp   2.5%  PVT_VANILLA_s1 0.846  S3LMUP_s1 0.311  (S3LMUP_s1 at 0.5: missed 0.18, false pos 3.45 of polyp area)
      EndoCV2021_001117        polyp   1.8%  PVT_VANILLA_s1 0.739  S3LMUP_s1 0.220  (S3LMUP_s1 at 0.5: missed 0.41, false pos 3.73 of polyp area)
      EndoCV2021_001041        polyp   1.1%  PVT_VANILLA_s1 0.915  S3LMUP_s1 0.438  (S3LMUP_s1 at 0.5: missed 0.12, false pos 2.14 of polyp area)
      EndoCV2021_001292        polyp  10.1%  PVT_VANILLA_s1 0.657  S3LMUP_s1 0.245  (S3LMUP_s1 at 0.5: missed 0.77, false pos 0.62 of polyp area)
      EndoCV2021_001406        polyp   0.4%  PVT_VANILLA_s1 0.875  S3LMUP_s1 0.492  (S3LMUP_s1 at 0.5: missed 0.06, false pos 1.84 of polyp area)
  PG-C3: +0.0030  = 10 worst images -0.0062, 10 best +0.0053, other 435 +0.0039;  complete misses 2 vs 2
  PG-C4: +0.0015  = 10 worst images -0.0365, 10 best +0.0323, other 126 +0.0057;  complete misses 14 vs 13
  PG-C5: +0.0249  = 10 worst images -0.0207, 10 best +0.0283, other 186 +0.0173;  complete misses 19 vs 20
  PG-C6: +0.0022  = 10 worst images -0.0133, 10 best +0.0139, other 63 +0.0017;  complete misses 6 vs 5

per-image numbers for every run: results/POLYPGEN_PER_IMAGE.csv
```
