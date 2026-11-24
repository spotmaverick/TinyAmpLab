## a2lite (11-24 00:15) [done]
- 配置: 
- train: [train] corr +0.9697 NRMS 0.2457 | val: [val] corr +0.9638 NRMS 0.2704 | test: [test] corr +0.9668 NRMS 0.2589
- 产物: 2/2 个已回传 (权重+音频) | wall: 1231s

## lgw1 (11-24 00:15) [failed/no-final]
- 配置: 
- train:  | val:  | test: 
- 产物: 0/2 个已回传 (权重+音频) | wall: 1691s

## lgw1 (11-24 00:17) [done]
- 配置: 
- train: [train] corr +0.9771 NRMS 0.2152 | val: [val] corr +0.9737 NRMS 0.2287 | test: [test] corr +0.9751 NRMS 0.2231
- 产物: 2/2 个已回传 (权重+音频) | wall: 1849s

## lgw2 (11-24 00:20) [done]
- 配置: 
- train: [train] corr +0.9841 NRMS 0.1858 | val: [val] corr +0.9826 NRMS 0.1947 | test: [test] corr +0.9827 NRMS 0.1948
- 产物: 2/2 个已回传 (权重+音频) | wall: 1959s

## v_tcn812 (11-24 00:39) [done]
- 配置: [CFG] v_tcn812 ch=8 L=12 gate=none leaky=False skipgates=True dil_exp=7 head2=1 params=2221 RF=541(11.3ms) MACs/s=1968 state=8.4KB
- train: [train] corr +0.9774 NRMS 0.2158 | val: [val] corr +0.9758 NRMS 0.2194 | test: [test] corr +0.9762 NRMS 0.2198
- 产物: 2/2 个已回传 (权重+音频) | wall: 1380s

## v_classic814 (11-24 00:42) [done]
- 配置: [CFG] v_classic814 ch=8 L=14 gate=classic leaky=True skipgates=True dil_exp=7 head2=1 params=3583 RF=637(13.3ms) MACs/s=3168 state=9.9KB
- train: [train] corr +0.9787 NRMS 0.2083 | val: [val] corr +0.9766 NRMS 0.2187 | test: [test] corr +0.9780 NRMS 0.2124
- 产物: 2/2 个已回传 (权重+音频) | wall: 1576s

## v_sgate814 (11-24 00:42) [done]
- 配置: [CFG] v_sgate814 ch=8 L=14 gate=sgate leaky=True skipgates=True dil_exp=7 head2=1 params=3583 RF=637(13.3ms) MACs/s=3168 state=9.9KB
- train: [train] corr +0.9730 NRMS 0.2308 | val: [val] corr +0.9684 NRMS 0.2507 | test: [test] corr +0.9691 NRMS 0.2475
- 产物: 2/2 个已回传 (权重+音频) | wall: 1574s

## v_dual512 (11-24 01:42) [done]
- 配置: [CFG] v_dual512 ch=5 L=12 gate=dual leaky=True skipgates=True dil_exp=7 head2=0 params=2068 RF=541(11.3ms) MACs/s=1690 state=5.3KB
- train: [train] corr +0.9620 NRMS 0.2770 | val: [val] corr +0.9548 NRMS 0.3025 | test: [test] corr +0.9576 NRMS 0.2928
- 产物: 2/2 个已回传 (权重+音频) | wall: 1341s

## v_sgate614 (11-24 01:45) [done]
- 配置: [CFG] v_sgate614 ch=6 L=14 gate=sgate leaky=True skipgates=True dil_exp=7 head2=0 params=2133 RF=637(13.3ms) MACs/s=1776 state=7.5KB
- train: [train] corr +0.9622 NRMS 0.2769 | val: [val] corr +0.9539 NRMS 0.3016 | test: [test] corr +0.9584 NRMS 0.2880
- 产物: 2/2 个已回传 (权重+音频) | wall: 1540s

## v_dual418 (11-24 01:52) [done]
- 配置: [CFG] v_dual418 ch=4 L=18 gate=dual leaky=True skipgates=True dil_exp=7 head2=0 params=2119 RF=1027(21.4ms) MACs/s=1664 state=8.0KB
- train: [train] corr +0.9645 NRMS 0.2644 | val: [val] corr +0.9586 NRMS 0.2853 | test: [test] corr +0.9621 NRMS 0.2727
- 产物: 2/2 个已回传 (权重+音频) | wall: 1945s

## v_classic419 (11-24 01:53) [done]
- 配置: [CFG] v_classic419 ch=4 L=19 gate=classic leaky=True skipgates=True dil_exp=7 head2=0 params=1476 RF=1035(21.6ms) MACs/s=1148 state=8.1KB
- train: [train] corr +0.9359 NRMS 0.3646 | val: [val] corr +0.9146 NRMS 0.4167 | test: [test] corr +0.9276 NRMS 0.3953
- 产物: 2/2 个已回传 (权重+音频) | wall: 2028s

## v_rf814d9 (11-24 02:06) [done]
- 配置: [CFG] v_rf814d9 ch=8 L=14 gate=dual leaky=True skipgates=True dil_exp=9 head2=1 params=5599 RF=2077(43.3ms) MACs/s=4960 state=32.4KB
- train: [train] corr +0.9812 NRMS 0.1932 | val: [val] corr +0.9764 NRMS 0.2165 | test: [test] corr +0.9792 NRMS 0.2035
- 产物: 2/2 个已回传 (权重+音频) | wall: 6330s

## v_classic614 (11-24 02:08) [done]
- 配置: [CFG] v_classic614 ch=6 L=14 gate=classic leaky=True skipgates=True dil_exp=7 head2=0 params=2133 RF=637(13.3ms) MACs/s=1776 state=7.5KB
- train: [train] corr +0.9629 NRMS 0.2702 | val: [val] corr +0.9545 NRMS 0.2987 | test: [test] corr +0.9588 NRMS 0.2840
- 产物: 2/2 个已回传 (权重+音频) | wall: 1525s

## v_sgate419 (11-24 02:21) [done]
- 配置: [CFG] v_sgate419 ch=4 L=19 gate=sgate leaky=True skipgates=True dil_exp=7 head2=0 params=1476 RF=1035(21.6ms) MACs/s=1148 state=8.1KB
- train: [train] corr +0.9459 NRMS 0.3255 | val: [val] corr +0.9320 NRMS 0.3639 | test: [test] corr +0.9405 NRMS 0.3434
- 产物: 2/2 个已回传 (权重+音频) | wall: 2045s

## v_tcn812h1 (11-24 02:48) [done]
- 配置: [CFG] v_tcn812h1 ch=8 L=12 gate=none leaky=False skipgates=True dil_exp=7 head2=0 params=2149 RF=541(11.3ms) MACs/s=1840 state=8.4KB
- train: [train] corr +0.9788 NRMS 0.2051 | val: [val] corr +0.9775 NRMS 0.2115 | test: [test] corr +0.9782 NRMS 0.2089
- 产物: 2/2 个已回传 (权重+音频) | wall: 1325s

## v_tcn814 (11-24 02:49) [done]
- 配置: [CFG] v_tcn814 ch=8 L=14 gate=none leaky=False skipgates=True dil_exp=7 head2=0 params=2503 RF=637(13.3ms) MACs/s=2144 state=9.9KB
- train: [train] corr +0.9802 NRMS 0.1998 | val: [val] corr +0.9784 NRMS 0.2087 | test: [test] corr +0.9788 NRMS 0.2072
- 产物: 2/2 个已回传 (权重+音频) | wall: 1513s

## v_tcn814d9 (11-24 02:50) [done]
- 配置: [CFG] v_tcn814d9 ch=8 L=14 gate=none leaky=False skipgates=True dil_exp=9 head2=0 params=2503 RF=2077(43.3ms) MACs/s=2144 state=32.4KB
- train: [train] corr +0.9789 NRMS 0.2083 | val: [val] corr +0.9768 NRMS 0.2167 | test: [test] corr +0.9777 NRMS 0.2115
- 产物: 2/2 个已回传 (权重+音频) | wall: 1531s

## v_tcn814d9l (11-24 02:50) [done]
- 配置: [CFG] v_tcn814d9l ch=8 L=14 gate=none leaky=True skipgates=True dil_exp=9 head2=0 params=2503 RF=2077(43.3ms) MACs/s=2144 state=32.4KB
- train: [train] corr +0.9727 NRMS 0.2333 | val: [val] corr +0.9678 NRMS 0.2521 | test: [test] corr +0.9697 NRMS 0.2445
- 产物: 2/2 个已回传 (权重+音频) | wall: 1513s

## v_tcn812d9 (11-24 05:16) [done]
- 配置: [CFG] v_tcn812d9 ch=8 L=12 gate=none leaky=False skipgates=True dil_exp=9 head2=1 params=2221 RF=2053(42.8ms) MACs/s=1968 state=32.1KB
- train: [train] corr +0.9752 NRMS 0.2217 | val: [val] corr +0.9704 NRMS 0.2422 | test: [test] corr +0.9722 NRMS 0.2344
- 产物: 2/2 个已回传 (权重+音频) | wall: 9305s

## v_tcn713h1 (11-24 05:42) [done]
- 配置: [CFG] v_tcn713h1 ch=7 L=13 gate=none leaky=False skipgates=True dil_exp=7 head2=0 sched=const params=1855 RF=573(11.9ms) MACs/s=1561 state=7.8KB
- train: [train] corr +0.9765 NRMS 0.2184 | val: [val] corr +0.9750 NRMS 0.2255 | test: [test] corr +0.9756 NRMS 0.2246
- 产物: 2/2 个已回传 (权重+音频) | wall: 1421s

## v_tcn812h1_p3 (11-24 06:24) [done]
- 配置: [CFG] v_tcn812h1_p3 ch=8 L=12 gate=none leaky=False skipgates=True dil_exp=7 head2=0 sched=cosine params=2149 RF=541(11.3ms) MACs/s=1840 state=8.4KB
- train: [train] corr +0.9856 NRMS 0.1695 | val: [val] corr +0.9840 NRMS 0.1791 | test: [test] corr +0.9850 NRMS 0.1736
- 产物: 2/2 个已回传 (权重+音频) | wall: 3923s

## v_tcn713h1_p3 (11-24 07:38) [done]
- 配置: [CFG] v_tcn713h1_p3 ch=7 L=13 gate=none leaky=False skipgates=True dil_exp=7 head2=0 sched=cosine params=1855 RF=573(11.9ms) MACs/s=1561 state=7.8KB
- train: [train] corr +0.9845 NRMS 0.1759 | val: [val] corr +0.9834 NRMS 0.1823 | test: [test] corr +0.9838 NRMS 0.1801
- 产物: 2/2 个已回传 (权重+音频) | wall: 4205s

## v_tcn812h1_p3_s11 (11-24 07:41) [failed/no-final]
- 配置: [CFG] v_tcn812h1_p3 ch=8 L=12 gate=none leaky=False skipgates=True dil_exp=7 head2=0 sched=cosine params=2149 RF=541(11.3ms) MACs/s=1840 state=8.4KB
- train:  | val:  | test: 
- 产物: 0/2 个已回传 (权重+音频) | wall: s

