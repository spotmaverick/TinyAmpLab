"""matplotlib.pyplot stub：任何绘图调用都显式报错，避免静默产出空图。"""


def _no(*a, **k):
    raise RuntimeError("matplotlib stub: P-1 不做绘图，请用训练侧环境")


def __getattr__(name):
    return _no
