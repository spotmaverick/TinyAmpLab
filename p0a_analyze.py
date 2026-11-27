"""P0A 13 种子扩展：两条预测的后处理检验。

预测1：top1 跨音频稳定 → 排序可直接跨种子比
预测2：top1 越高 head 收益越大 → 单调性在 13 种子延续

用法:
  python p0a_analyze.py pred1 <proxy.json>
  python p0a_analyze.py pred2 <causal.json>
  python p0a_analyze.py both  <proxy.json> <causal.json>
"""
import json, sys, itertools
import numpy as np
from scipy.stats import spearmanr, pearsonr

def load(p): return json.load(open(p))

def check_pred1(proxy_json):
    d = load(proxy_json)
    rows = []
    for seed, segs in sorted(d.get('proxy', {}).items()):
        vals = [segs[sg]['amp_top1_frac'] for sg in ('seg0','seg1','seg2')
                if segs.get(sg) and segs[sg].get('amp_top1_frac') is not None]
        if len(vals) < 3: continue
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        rows.append(dict(seed=seed, top1=m, sd=sd, rng=max(vals)-min(vals),
                         cv=sd/m if m else None, vals=vals))
    if not rows:
        print("预测1：无 proxy 数据（需要 --mode proxy 输出）"); return []
    top1s = np.array([r['top1'] for r in rows])
    seg_sds = np.array([r['sd'] for r in rows])
    seg_rngs = np.array([r['rng'] for r in rows])
    print("=== 预测1：top1 跨音频稳定性 ===")
    print(f"{'seed':6} {'top1均值':>9} {'段内σ':>8} {'段内极差':>9} {'CV':>7}")
    for r in rows:
        print(f"{r['seed']:6} {r['top1']:9.4f} {r['sd']:8.4f} {r['rng']:9.4f} {r['cv']:7.2%}")
    inter_sd = top1s.std(ddof=1) if len(top1s) > 1 else float('nan')
    print(f"\n种子间 top1 标准差 = {inter_sd:.4f}")
    print(f"段内 σ 中位数 = {np.median(seg_sds):.4f}  最大 = {seg_sds.max():.4f}")
    print(f"段内极差中位数 = {np.median(seg_rngs):.4f}  最大 = {seg_rngs.max():.4f}")
    ratio = np.median(seg_sds)/inter_sd if inter_sd and inter_sd>0 else float('nan')
    print(f"噪声/信号比 (段内σ中位 / 种子间σ) = {ratio:.3f}")
    # 段间排序一致性
    v = np.array([[r['vals'][i] for r in rows] for i in range(3)])
    rhos = []
    print("\n段间 top1 排序一致性 (Spearman ρ of top1 向量):")
    for i,j in itertools.combinations(range(3),2):
        rho,_ = spearmanr(v[i], v[j]); rhos.append(rho)
        print(f"  seg{i} vs seg{j}: ρ={rho:+.3f}")
    ok = (ratio < 0.30) and all(r > 0.9 for r in rhos)
    print(f"\n>>> 预测1判定: {'成立 ✓' if ok else '不成立 ✗'} (判据: 噪声/信号<0.30 且段间ρ>0.9)")
    return rows

def check_pred2(causal_json, threshold_lo=0.45, threshold_hi=0.60):
    d = load(causal_json)
    rows = []
    for seed, slot in sorted(d.get('causal', {}).items()):
        t1 = slot.get('top1')
        v48 = slot.get('native', {}).get('v48', {})
        rec = v48.get('recovery_agg')
        rec_mean = v48.get('recovery_mean')
        if t1 is None or rec is None: continue
        rows.append(dict(seed=seed, top1=t1, rec=rec, rec_mean=rec_mean,
                         band=('高' if t1>threshold_hi else '中' if t1>=threshold_lo else '低')))
    if not rows:
        print("预测2：无 causal 数据（需要 --mode causal 输出）"); return []
    print("\n=== 预测2：top1 vs head 收益单调性 ===")
    print(f"{'seed':6} {'top1':>7} {'档':>4} {'rec@V48':>9} {'rec_mean':>9}")
    for r in rows:
        rm = f"{r['rec_mean']:+.3f}" if r['rec_mean'] is not None else '-'
        print(f"{r['seed']:6} {r['top1']:7.4f} {r['band']:>4} {r['rec']:+9.3f} {rm:>9}")
    t1 = np.array([r['top1'] for r in rows]); rec = np.array([r['rec'] for r in rows])
    if len(rows) < 3:
        print(f"\nn={len(rows)} 样本不足，跳过相关性检验"); return rows
    rho, prho = spearmanr(t1, rec)
    r, pr = pearsonr(t1, rec)
    print(f"\nn={len(rows)}  Spearman ρ={rho:+.3f} (p={prho:.4f})  Pearson r={r:+.3f} (p={pr:.4f})")
    order_t1 = [r['seed'] for r in sorted(rows, key=lambda x:x['top1'], reverse=True)]
    order_rec = [r['seed'] for r in sorted(rows, key=lambda x:x['rec'], reverse=True)]
    print(f"top1 排序: {' > '.join(order_t1)}")
    print(f"收益排序: {' > '.join(order_rec)}")
    bands = {}
    for b in ('高','中','低'):
        vv=[r['rec'] for r in rows if r['band']==b]
        if vv: bands[b]=(np.mean(vv), len(vv))
    print("分档平均收益:", "  ".join(f"{b}档={m:+.3f}(n={n})" for b,(m,n) in bands.items()))
    ok = rho > 0.6 and prho < 0.05
    print(f"\n>>> 预测2判定: {'成立 ✓' if ok else '不成立 ✗'} (判据: ρ>0.6 且 p<0.05)")
    return rows

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    mode = sys.argv[1]
    if mode == 'pred1':
        check_pred1(sys.argv[2])
    elif mode == 'pred2':
        check_pred2(sys.argv[2])
    elif mode == 'both':
        check_pred1(sys.argv[2])
        check_pred2(sys.argv[3])
