# 夜间批次（多管箱泛化）结果汇总

生成时间：2026-11-26 09:31:04

**口径**：val NRMS 取 JSON 的 `best_val_NRMS`（300 ep 内最优）；test NRMS 为留出测试段。数值越小越好。

**来源**：z100 凌晨批次（34 个） + bw 补跑（5 个，2026-11-26 09:26）。`训练秒` 一列跨机不可比：z100 为 18 作业挤 7 节点的实测（28~46 s/ep），bw 为单卡 5 路并发（2.6~3.2 s/ep），同素材 bw 约快 10 倍。

## 一、全量结果

| 模型 | 管箱 | 参数 | ep | val NRMS | test NRMS | test ESR | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| standard_ts9_s1_s7 | ts9_s1 | 13801 | 300 | 0.0148 | 0.0139 | 0.0002 | 824.9 |
| lite_ts9_s1_s7 | ts9_s1 | 6553 | 300 | 0.0159 | 0.0149 | 0.0002 | 10150.9 |
| lite_ts9_s2_s7 | ts9_s2 | 6553 | 300 | 0.0177 | 0.0167 | 0.0003 | 10147.3 |
| standard_markv_clean_s7 | markv_clean | 13801 | 300 | 0.0206 | 0.0368 | 0.0014 | 751.9 |
| lite_markv_clean_s7 | markv_clean | 6553 | 300 | 0.0245 | 0.0389 | 0.0015 | 9182.5 |
| standard_bigmuff_diy_s7 | bigmuff_diy | 13801 | 300 | 0.0399 | 0.0319 | 0.0010 | 936.8 |
| lite_bigmuff_diy_s7 | bigmuff_diy | 6553 | 300 | 0.0500 | 0.0427 | 0.0018 | 10148.2 |
| standard_s11 | tsa15 | 13801 | 300 | 0.0650 | 0.0658 | 0.0043 | 10047.9 |
| standard_marshall_jvm410h_chod1_s7 | marshall_jvm410h_chod1 | 13801 | 300 | 0.0651 | 0.0801 | 0.0064 | 13750.0 |
| standard_ua1176_s7 | ua1176 | 13801 | 300 | 0.0659 | 0.0928 | 0.0086 | 945.4 |
| standard_rf50_s7 | tsa15 | 13801 | 300 | 0.0715 | 0.0727 | 0.0053 | 9207.6 |
| standard_marshall_jvm410h_chod1_spk_s7 | marshall_jvm410h_chod1_spk | 13801 | 300 | 0.0739 | 0.0870 | 0.0076 | 13749.5 |
| lite_marshall_jvm410h_chod1_s7 | marshall_jvm410h_chod1 | 6553 | 300 | 0.0743 | 0.0926 | 0.0086 | 12568.5 |
| lite_bsht5m_s7 | bsht5m | 6553 | 300 | 0.0790 | 0.0819 | 0.0067 | 9182.8 |
| lite_s11 | tsa15 | 6553 | 300 | 0.0791 | 0.0760 | 0.0058 | 9187.3 |
| feather_marshall_jvm410h_chod1_s7 | marshall_jvm410h_chod1 | 3025 | 300 | 0.0808 | 0.0996 | 0.0099 | 12044.1 |
| lite_s7 | tsa15 | 6553 | 300 | 0.0812 | 0.0785 | 0.0062 | 9192.1 |
| lite_rf75_s7 | tsa15 | 6553 | 300 | 0.0818 | 0.0796 | 0.0063 | 8915.4 |
| lite_marshall_jvm410h_chod1_spk_s7 | marshall_jvm410h_chod1_spk | 6553 | 300 | 0.0829 | 0.0962 | 0.0092 | 12571.3 |
| lite_markvx_s7 | markvx | 6553 | 300 | 0.0830 | 0.0877 | 0.0077 | 9183.8 |
| lite_ua1176_s7 | ua1176 | 6553 | 300 | 0.0831 | 0.0956 | 0.0091 | 9189.4 |
| lite_rf50_s11 | tsa15 | 6553 | 300 | 0.0843 | 0.0840 | 0.0071 | 8705.3 |
| lite_rf50_s7 | tsa15 | 6553 | 300 | 0.0883 | 0.0885 | 0.0078 | 8719.0 |
| lite_mesa550_crunch_s7 | mesa550_crunch | 6553 | 300 | 0.0941 | 0.0978 | 0.0096 | 10341.3 |
| lite_englrt50_s7 | englrt50 | 6553 | 300 | 0.0942 | 0.1000 | 0.0100 | 9181.6 |
| feather_marshall_jvm410h_chod1_spk_s7 | marshall_jvm410h_chod1_spk | 3025 | 300 | 0.0981 | 0.1106 | 0.0122 | 12040.3 |
| feather_s11 | tsa15 | 3025 | 300 | 0.0983 | 0.0961 | 0.0092 | 8804.0 |
| lite_mesa550_clean_s7 | mesa550_clean | 6553 | 300 | 0.1008 | 0.0969 | 0.0094 | 928.1 |
| lite_rf25_s7 | tsa15 | 6553 | 300 | 0.1012 | 0.1029 | 0.0106 | 8543.4 |
| feather_rf50_s7 | tsa15 | 3025 | 300 | 0.1013 | 0.1005 | 0.0101 | 8466.1 |
| feather_s7 | tsa15 | 3025 | 300 | 0.1031 | 0.1014 | 0.0103 | 8806.3 |
| feather_rf50_s11 | tsa15 | 3025 | 300 | 0.1075 | 0.1065 | 0.0113 | 8460.7 |
| lite_mesa550_burn_s7 | mesa550_burn | 6553 | 300 | 0.1218 | 0.1194 | 0.0143 | 9185.9 |
| nano_marshall_jvm410h_chod1_spk_s7 | marshall_jvm410h_chod1_spk | 841 | 300 | 0.1256 | 0.1394 | 0.0194 | 11548.3 |
| nano_marshall_jvm410h_chod1_s7 | marshall_jvm410h_chod1 | 841 | 300 | 0.1388 | 0.1623 | 0.0263 | 11560.7 |
| nano_rf50_s7 | tsa15 | 841 | 300 | 0.1443 | 0.1412 | 0.0199 | 8283.2 |
| nano_s7 | tsa15 | 841 | 300 | 0.1508 | 0.1485 | 0.0221 | 8450.4 |
| nano_rf50_s11 | tsa15 | 841 | 300 | 0.1856 | 0.1793 | 0.0321 | 8270.7 |
| nano_s11 | tsa15 | 841 | 300 | 0.1982 | 0.1912 | 0.0365 | 8445.7 |

## 二、管箱 × 规模矩阵（val NRMS，rf=100 主配置）

| 管箱 | nano | feather | lite | standard |
|---|---|---|---|---|
| ts9_s1 | — | — | 0.0159 | 0.0148 |
| ts9_s2 | — | — | 0.0177 | — |
| markv_clean | — | — | 0.0245 | 0.0206 |
| bigmuff_diy | — | — | 0.0500 | 0.0399 |
| tsa15 | 0.1508 | 0.1031 | 0.0812 | 0.0650 |
| marshall_jvm410h_chod1 | 0.1388 | 0.0808 | 0.0743 | 0.0651 |
| ua1176 | — | — | 0.0831 | 0.0659 |
| marshall_jvm410h_chod1_spk | 0.1256 | 0.0981 | 0.0829 | 0.0739 |
| bsht5m | — | — | 0.0790 | — |
| markvx | — | — | 0.0830 | — |
| mesa550_crunch | — | — | 0.0941 | — |
| englrt50 | — | — | 0.0942 | — |
| mesa550_clean | — | — | 0.1008 | — |
| mesa550_burn | — | — | 0.1218 | — |

## 三、RF 缩放变体（非主配置，仅供参考）

- tsa15                    standard_rf50  0.0715
- tsa15                    lite_rf75      0.0818
- tsa15                    lite_rf50      0.0843
- tsa15                    lite_rf50      0.0883
- tsa15                    lite_rf25      0.1012
- tsa15                    feather_rf50   0.1013
- tsa15                    feather_rf50   0.1075
- tsa15                    nano_rf50      0.1443
- tsa15                    nano_rf50      0.1856

## 四、按 val NRMS 排序（越小越好）

- ts9_s1                         standard_ts9_s1_s7                           0.0148
- ts9_s1                         lite_ts9_s1_s7                               0.0159
- ts9_s2                         lite_ts9_s2_s7                               0.0177
- markv_clean                    standard_markv_clean_s7                      0.0206
- markv_clean                    lite_markv_clean_s7                          0.0245
- bigmuff_diy                    standard_bigmuff_diy_s7                      0.0399
- bigmuff_diy                    lite_bigmuff_diy_s7                          0.0500
- tsa15                          standard_s11                                 0.0650
- marshall_jvm410h_chod1         standard_marshall_jvm410h_chod1_s7           0.0651
- ua1176                         standard_ua1176_s7                           0.0659
- tsa15                          standard_rf50_s7                             0.0715
- marshall_jvm410h_chod1_spk     standard_marshall_jvm410h_chod1_spk_s7       0.0739
- marshall_jvm410h_chod1         lite_marshall_jvm410h_chod1_s7               0.0743
- bsht5m                         lite_bsht5m_s7                               0.0790
- tsa15                          lite_s11                                     0.0791
- marshall_jvm410h_chod1         feather_marshall_jvm410h_chod1_s7            0.0808
- tsa15                          lite_s7                                      0.0812
- tsa15                          lite_rf75_s7                                 0.0818
- marshall_jvm410h_chod1_spk     lite_marshall_jvm410h_chod1_spk_s7           0.0829
- markvx                         lite_markvx_s7                               0.0830
- ua1176                         lite_ua1176_s7                               0.0831
- tsa15                          lite_rf50_s11                                0.0843
- tsa15                          lite_rf50_s7                                 0.0883
- mesa550_crunch                 lite_mesa550_crunch_s7                       0.0941
- englrt50                       lite_englrt50_s7                             0.0942
- marshall_jvm410h_chod1_spk     feather_marshall_jvm410h_chod1_spk_s7        0.0981
- tsa15                          feather_s11                                  0.0983
- mesa550_clean                  lite_mesa550_clean_s7                        0.1008
- tsa15                          lite_rf25_s7                                 0.1012
- tsa15                          feather_rf50_s7                              0.1013
- tsa15                          feather_s7                                   0.1031
- tsa15                          feather_rf50_s11                             0.1075
- mesa550_burn                   lite_mesa550_burn_s7                         0.1218
- marshall_jvm410h_chod1_spk     nano_marshall_jvm410h_chod1_spk_s7           0.1256
- marshall_jvm410h_chod1         nano_marshall_jvm410h_chod1_s7               0.1388
- tsa15                          nano_rf50_s7                                 0.1443
- tsa15                          nano_s7                                      0.1508
- tsa15                          nano_rf50_s11                                0.1856
- tsa15                          nano_s11                                     0.1982
