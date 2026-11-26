"""模拟电路友好候选架构生成器（官方 NAM 配置表达，零新模型代码）

设计依据：docs/plans/ANALOG_A2_ALIGNED.md
物理成本排序（从贵到便宜）：延迟线抽头 > 交叉开关 > 累加链 > MACs > 参数

本脚本只产出官方 nam_full_configs 兼容的 JSON，A2 本身也用同一份官方配置，
确保「同一训练器、同一数据、同一评测口径」的公平对比。

用法：
    python src/tools/gen_analog_cfg.py --list
    python src/tools/gen_analog_cfg.py --out experiments/a2_official/cfg
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "github_mirrors" / "neural-amp-modeler" / "nam_full_configs" / "models"
A2_SRC = OFFICIAL / "wavenet_slimmable.json"

# A2 官方规格（与 wavenet_slimmable.json 完全一致，仅用于成本核算）
A2_CH = 12
A2_K = 6
A2_DIL = [1, 5, 29, 97, 227] * 3
A2_HEAD_SCALE = 0.01


def a2_layers():
    """A2 单层数组配置（官方）"""
    return [
        {
            "condition_size": 1,
            "input_size": 1,
            "channels": A2_CH,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "kernel_size": A2_K,
            "dilations": list(A2_DIL),
            "activation": "LeakyReLU",
        }
    ]


def cost(ch, k, dil, bottleneck=None, g_conv=1, g_mix=1, g_l1=1, head_out=1):
    """物理成本核算（严格按官方算子结构）

    官方每层: conv(ch -> mid, k, groups=g_conv) + input_mixer(1 -> mid)
              + layer1x1(bottleneck -> ch, groups=g_l1)
    mid = 2*bottleneck（配对激活）或 bottleneck
    """
    bn = bottleneck or ch
    mid = bn
    n = len(dil)

    xbar = 0   # 交叉开关（权重乘法器单元数）
    macs = 0   # 每采样乘法数
    chain = 0  # 单条求和链最长累加项
    rf = 1 + sum((k - 1) * d for d in dil)

    xbar += 1 * ch                                   # rechannel
    macs += ch
    for _ in range(n):
        cw = ch * mid * k // g_conv                  # conv 权重
        xbar += cw
        macs += cw
        chain = max(chain, mid * k // g_conv)        # 每个输出通道累加项

        mw = ch                                      # input_mixer (1 -> mid)*ch
        xbar += mw
        macs += mw

        lw = bn * ch // g_l1                         # layer1x1
        xbar += lw
        macs += lw
        chain = max(chain, bn // g_l1)
    xbar += ch * head_out                            # head_rechannel
    macs += ch * head_out

    params = xbar + ch * n + ch * head_out + bn * n  # + biases 近似

    return dict(params=params, xbar=xbar, macs=macs, chain=chain, rf=rf,
                delay_taps=rf, layers=n)


def build_variant(name):
    """返回 (config_dict, 说明, 成本)"""
    v = name.lower()
    base = {"channels": A2_CH, "kernel_size": A2_K, "dilations": list(A2_DIL),
            "bottleneck": None, "g_conv": 1, "g_mix": 1, "g_l1": 1,
            "activation": "LeakyReLU", "head_scale": A2_HEAD_SCALE}

    if v == "a2":
        desc = "官方 A2 Slimmable 原样（锚点）"
        cfg = json.load(open(A2_SRC))
        c = cost(A2_CH, A2_K, A2_DIL)
        return cfg, desc, c

    # ---- 族 A：官方配置内可表达 ----
    if v == "a-dw":
        base["g_conv"] = A2_CH
        desc = "A-dw: conv 改 depthwise（交叉开关 ch²k→ch·k）"
    elif v.startswith("a-grp"):
        g = int(v.replace("a-grp", ""))
        base["g_conv"] = g
        desc = f"A-grp{g}: conv 分组 g={g}"
    elif v.startswith("a-bneck"):
        bn = int(v.replace("a-bneck", ""))
        base["bottleneck"] = bn
        desc = f"A-bneck{bn}: bottleneck 收窄到 {bn}"
    elif v.startswith("a-k"):
        k = int(v.replace("a-k", ""))
        base["kernel_size"] = k
        desc = f"A-k{k}: kernel 3→{k}（乘法器与 RF 同比缩）"
    elif v.startswith("a-l1grp"):
        g = int(v.replace("a-l1grp", ""))
        base["g_l1"] = g
        desc = f"A-l1grp{g}: layer1x1 分组 g={g}（ch²→ch²/g）"
    elif v == "a-dw-ch16-l1grp4":
        base["channels"] = 16
        base["g_conv"] = 16
        base["g_l1"] = 4
        desc = "A-dw-ch16-l1grp4: depthwise ch16 + 1×1 分组 4"
    elif v == "a-dw-ch24":
        base["channels"] = 24
        base["g_conv"] = 24
        desc = "A-dw-ch24: depthwise + ch24（宽度补参数）"
    elif v.startswith("a-dw-ch"):
        ch = int(v.replace("a-dw-ch", ""))
        base["channels"] = ch
        base["g_conv"] = ch
        desc = f"A-dw-ch{ch}: depthwise + 通道 {ch}"
    elif v == "a-dw-l1grp4":
        base["g_conv"] = A2_CH
        base["g_l1"] = 4
        desc = "A-dw-l1grp4: depthwise + 1×1 分组 4（交叉开关双降）"
    elif v == "a-dw-l1grp3":
        base["g_conv"] = A2_CH
        base["g_l1"] = 3
        desc = "A-dw-l1grp3: depthwise + 1×1 分组 3（对齐 allowed_channels 3/12）"
    elif v == "a-dw-ch16":
        base["channels"] = 16
        base["g_conv"] = 16
        desc = "A-dw-ch16: depthwise + ch16（补回通道数换参数）"
    elif v == "a-grp6":
        base["g_conv"] = 6
        desc = "A-grp6: conv 分组 6（每组 2 通道）"
    elif v == "a-dw-k3":
        base["g_conv"] = A2_CH
        base["kernel_size"] = 3
        desc = "A-dw-k3: depthwise + kernel3（RF 与乘法器双减）"
    else:
        raise SystemExit(f"未知变体: {name}")

    cfg = _make_cfg(base)
    c = cost(base["channels"], base["kernel_size"], base["dilations"],
             base["bottleneck"], base["g_conv"], base["g_mix"], base["g_l1"])
    return cfg, desc, c


def _make_cfg(b):
    """按官方 wavenet_slimmable.json 结构生成配置（保留 slimmable 以支持 Lite/Full 共权重）"""
    ch = b["channels"]
    lc = {
        "condition_size": 1,
        "input_size": 1,
        "channels": ch,
        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
        "kernel_size": b["kernel_size"],
        "dilations": list(b["dilations"]),
        "activation": b["activation"],
    }
    if b["g_conv"] != 1:
        lc["groups_input"] = b["g_conv"]
    if b["bottleneck"]:
        lc["bottleneck"] = b["bottleneck"]
    if b["g_l1"] != 1:
        lc["layer_1x1_config"] = {"active": True, "groups": b["g_l1"]}

    return {
        "_notes": [
            "模拟电路友好候选（官方配置表达，无新模型代码）",
            f"conv groups={b['g_conv']} bottleneck={b['bottleneck'] or ch} "
            f"layer1x1 groups={b['g_l1']} kernel={b['kernel_size']}",
            "slimmable 保留，与 A2 同协议同数据对比",
        ],
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [lc],
                "head_scale": b["head_scale"],
            },
        },
        "optimizer": {"lr": 0.004},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }


VARIANTS = [
    "a2", "a-dw", "a-grp4", "a-grp6", "a-bneck6", "a-bneck4", "a-k3",
    "a-l1grp4", "a-l1grp3", "a-dw-l1grp4", "a-dw-l1grp3",
    "a-dw-ch16", "a-dw-ch16-l1grp4", "a-dw-ch24", "a-dw-k3",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出目录（不指定则只打印）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="逗号分隔的变体名")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else VARIANTS
    if args.list:
        print(f"{'变体':<20}{'参数':>7}{'交叉开关':>9}{'MACs':>7}{'累加链':>7}{'RF':>7}  说明")
        print("-" * 100)
        for nm in names:
            try:
                _, d, c = build_variant(nm)
            except SystemExit:
                continue
            print(f"{nm:<20}{c['params']:>7}{c['xbar']:>9}{c['macs']:>7}"
                  f"{c['chain']:>7}{c['rf']:>7}  {d}")
        return

    out = Path(args.out or (ROOT / "experiments/a2_official/cfg"))
    out.mkdir(parents=True, exist_ok=True)
    for nm in names:
        cfg, desc, c = build_variant(nm)
        p = out / f"model_{nm}.json"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"{p.name:<28} {desc}")
        print(f"{'':28} params={c['params']} xbar={c['xbar']} "
              f"macs={c['macs']} chain={c['chain']} rf={c['rf']}")


if __name__ == "__main__":
    main()
