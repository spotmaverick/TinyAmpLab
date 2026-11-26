"""wavio stub：nam/data.py 顶层的 write/read 入口，P-1 不走音频 IO。"""
__version__ = "0.0.0-stub"


def read(*a, **k):
    raise RuntimeError("wavio stub: P-1 用 soundfile 读 wav")


def write(*a, **k):
    raise RuntimeError("wavio stub: P-1 不写 wav")
