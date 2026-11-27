"""P0a 13 种子：预测2 机理分解（本地版，依赖 numpy）"""
import json
import numpy as np

d = json.load(open('runs_remote/p0a_1127/p4_proxy_bw2_13.json'))
c = d['causal']
S = sorted(c)


def sp(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def pe(x, y):
    return np.corrcoef(x, y)[0, 1]


def get(k):
    return np.array([c[s]['native']['v16'][k] if c[s]['native']['v16'].get(k) is not None else np.nan
                     for s in S])


t1 = get('amp_top1_frac')
eo = get('head_err_out_frac')
cr = get('head_crest')
tot = np.array([c[s]['total_drop_mean'] for s in S])
d48 = np.array([c[s]['native']['v48']['recov_pair_mean'] for s in S])
dinf = np.array([c[s]['native']['v100000']['recov_pair_mean'] for s in S])
rec = np.array([c[s]['native']['v48']['recovery_agg'] for s in S])
mono = get('max_err_out_other')

print('=' * 74)
print('13 种子：总退化与 head 收益的结构')
print('=' * 74)
print(f"\n{'seed':6}{'top1':>8}{'errOut':>9}{'d@48':>9}{'dInf':>9}{'tot':>9}{'rec@48':>8}{'maxOth':>9}")
for i, s in enumerate(S):
    print(f'{s:6}{t1[i]:8.4f}{eo[i]:9.4f}{d48[i]:9.4f}{dinf[i]:9.4f}{tot[i]:9.4f}{rec[i]:8.3f}{mono[i]:9.4f}')

print('\n--- 相关性矩阵（rho） ---')
print(f"{'':24}" + ''.join(f'{n:>10}' for n in ['top1', 'errOut', 'd@48', 'tot', 'rec@48']))
for n1, a1 in [('top1', t1), ('errOut', eo), ('d@48', d48), ('tot', tot), ('rec@48', rec)]:
    row = f'{n1:24}'
    for n2, a2 in [('top1', t1), ('errOut', eo), ('d@48', d48), ('tot', tot), ('rec@48', rec)]:
        row += f'{sp(a1, a2):10.3f}'
    print(row)

print('\n--- 预测2 原判据 vs 换指标 ---')
print(f'  top1    -> rec@V48  rho={sp(t1, rec):+0.4f}  r={pe(t1, rec):+0.4f}   (判据 rho>0.6)')
print(f'  errOut  -> rec@V48  rho={sp(eo, rec):+0.4f}  r={pe(eo, rec):+0.4f}')
print(f'  top1    -> d@V48    rho={sp(t1, d48):+0.4f}  r={pe(t1, d48):+0.4f}   (绝对收益)')
print(f'  top1    -> dInf     rho={sp(t1, dinf):+0.4f}  r={pe(t1, dinf):+0.4f}  (绝对收益, V->inf)')
print(f'  top1    -> tot      rho={sp(t1, tot):+0.4f}  r={pe(t1, tot):+0.4f}   (总退化)')

print('\n--- 斜率：分母比分子涨得更快？ ---')
for lbl, y in [('d@48 分子', d48), ('tot 分母', tot)]:
    b, a = np.polyfit(t1, y, 1)
    span = b * (t1.max() - t1.min())
    print(f'  {lbl:10} slope={b:+.4f}  全跨度变化={span:+.4f}')
print('  => 二者斜率接近，比值 rec 被稀释，相关性必然低于任一分量')
