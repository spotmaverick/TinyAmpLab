"""P-1 最小 librosa stub。

用途：nam 包的 __init__ 会无条件 import data.py，而 data.py 顶层 import librosa。
P-1 只从 .nam 的 JSON 里读权重，完全不经过 data.py 的音频加载路径，
所以这里只需要让 import 成功，任何实际调用都显式报错。
"""
__version__ = "0.0.0-stub"


def load(*a, **k):
    raise RuntimeError("librosa stub: P-1 不加载音频文件，只读 .nam JSON")


def resample(*a, **k):
    raise RuntimeError("librosa stub: 用 soundfile 自行重采样")


def get_duration(*a, **k):
    raise RuntimeError("librosa stub")


class feature:
    @staticmethod
    def melspectrogram(*a, **k):
        raise RuntimeError("librosa stub")
