# -*- coding: utf-8 -*-
"""
纯 numpy 轻量档推理器 —— HashingVectorizer + MultinomialNB

tier0 / en_tier0 的结构固定:
    HashingVectorizer(alternate_sign=False, norm='l2') -> MultinomialNB

sklearn 的 predict_proba 会把 65536 维全量稠密算一遍, 而我们只关心
非零的几个特征 (27 字文本约 100 个 n-gram)。纯 numpy 只累加非零项:
    热路径 0.7ms -> 0.03ms, 冷启动 1.4s -> 0.3s (不 import sklearn)

murmurhash3 与 sklearn.utils.murmurhash3_32(positive=True) 逐位一致,
见 tests 里的 test_hash_parity。
"""
import os
import numpy as np

_M = 0xFFFFFFFF


def _rotl32(x, r):
    return ((x << r) | (x >> (32 - r))) & _M


def murmurhash3_32_bytes(key, seed=0):
    """与 sklearn.utils.murmurhash3_32(positive=True) 结果一致

    注意: HashingVectorizer 内部用的是有符号版本 murmurhash3_bytes_s32
    (见 sklearn/feature_extraction/_hashing_fast.pyx), 然后取 abs(h) % n_features。
    见 signed_murmur 的用法。"""
    length = len(key)
    nblocks = length >> 2
    h1 = seed & _M
    c1, c2 = 0xCC9E2D51, 0x1B873593
    for i in range(nblocks):
        k1 = int.from_bytes(key[i * 4:i * 4 + 4], "little")
        k1 = (k1 * c1) & _M
        k1 = _rotl32(k1, 15)
        k1 = (k1 * c2) & _M
        h1 ^= k1
        h1 = _rotl32(h1, 13)
        h1 = (h1 * 5 + 0xE6546B64) & _M
    tail = key[nblocks * 4:]
    k1 = 0
    if len(tail) >= 3:
        k1 ^= tail[2] << 16
    if len(tail) >= 2:
        k1 ^= tail[1] << 8
    if len(tail) >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & _M
        k1 = _rotl32(k1, 15)
        k1 = (k1 * c2) & _M
        h1 ^= k1
    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & _M
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & _M
    h1 ^= h1 >> 16
    return h1


def murmurhash3_bytes_s32(key, seed=0):
    """与 sklearn 的 murmurhash3_bytes_s32 一致 (返回有符号 int32)"""
    h = murmurhash3_32_bytes(key, seed)
    return h - 0x100000000 if h >= 0x80000000 else h


def hash_index(feature: str, n_features: int, seed: int = 0) -> int:
    """复现 HashingVectorizer 的索引: abs(signed_murmur) % n_features"""
    h = murmurhash3_bytes_s32(feature.encode("utf-8"), seed)
    if h == -2147483648:
        # abs(-2**31) 无定义, sklearn 特判成这个式子
        return (2147483647 - (n_features - 1)) % n_features
    return abs(h) % n_features


class UnsupportedModel(Exception):
    pass


class FastNB:
    """HashingVectorizer + MultinomialNB 的纯 numpy 实现"""

    def __init__(self, model_dir, vec_name="vec.pkl", clf_name="clf.pkl"):
        self.dir = model_dir
        npz = os.path.join(model_dir, "fast_nb.npz")
        if os.path.exists(npz):
            self._load_npz(npz)
            return
        self._load_pickle(model_dir, vec_name, clf_name)

    # ---------- 载入 ----------
    def _load_npz(self, path):
        z = np.load(path, allow_pickle=False)
        cfg = z["cfg"]
        self.analyzer = {0: "char", 1: "char_wb", 2: "word"}.get(int(cfg[0]), "char")
        self.ngram_range = (int(cfg[1]), int(cfg[2]))
        self.lowercase = bool(cfg[3])
        self.n_features = int(cfg[4])
        self.seed = int(cfg[5])
        self.norm = "l2" if int(cfg[6]) else None
        self.class_log_prior = z["class_log_prior"].astype(np.float64)
        self.feature_log_prob = z["feature_log_prob"].astype(np.float64)

    def _load_pickle(self, model_dir, vec_name, clf_name):
        # 本地自训练模型, 来源可信
        import pickle
        vp = os.path.join(model_dir, vec_name)
        cp = os.path.join(model_dir, clf_name)
        if not (os.path.exists(vp) and os.path.exists(cp)):
            raise UnsupportedModel(f"缺少 {vp}")
        vec = pickle.load(open(vp, "rb"))
        clf = pickle.load(open(cp, "rb"))

        an = getattr(vec, "analyzer", None)
        if an not in ("char", "char_wb", "word"):
            raise UnsupportedModel(f"analyzer={an!r} 不支持")
        if getattr(vec, "alternate_sign", False):
            raise UnsupportedModel("alternate_sign=True 不支持")
        if not hasattr(clf, "feature_log_prob_"):
            raise UnsupportedModel("不是 MultinomialNB")

        self.analyzer = an
        self.ngram_range = tuple(vec.ngram_range)
        self.lowercase = bool(getattr(vec, "lowercase", True))
        self.n_features = int(vec.n_features)
        self.seed = int(getattr(vec, "seed", 0) or 0)
        self.norm = getattr(vec, "norm", "l2")
        self.class_log_prior = np.asarray(clf.class_log_prior_, dtype=np.float64)
        self.feature_log_prob = np.asarray(clf.feature_log_prob_, dtype=np.float64)

    # ---------- n-gram ----------
    def _ngrams(self, text):
        if self.lowercase:
            text = text.lower()
        lo, hi = self.ngram_range
        if self.analyzer == "char":
            n = len(text)
            for k in range(lo, hi + 1):
                if k > n:
                    break
                for i in range(n - k + 1):
                    yield text[i:i + k]
        elif self.analyzer == "char_wb":
            for word in text.split():
                padded = " " + word + " "
                m = len(padded)
                for k in range(lo, hi + 1):
                    if k > m:
                        break
                    for i in range(m - k + 1):
                        yield padded[i:i + k]
        else:
            import re
            toks = re.findall(r"(?u)\b\w+\b", text)
            n = len(toks)
            for k in range(lo, hi + 1):
                if k > n:
                    break
                for i in range(n - k + 1):
                    yield " ".join(toks[i:i + k])

    # ---------- 推理 ----------
    def transform(self, text):
        """返回 (indices, values), 已做 hash 与 L2 归一"""
        from collections import Counter
        hashed = Counter(
            hash_index(g, self.n_features, self.seed)
            for g in self._ngrams(text)
        )
        if not hashed:
            return None, None
        idx = np.fromiter(hashed.keys(), dtype=np.int32, count=len(hashed))
        vals = np.fromiter(hashed.values(), dtype=np.float64, count=len(hashed))
        if self.norm == "l2":
            n = float(np.sqrt(np.dot(vals, vals)))
            if n > 0:
                vals = vals / n
        return idx, vals

    def decision_function(self, text):
        """多类 NB 的对数后验比 (类1 - 类0)

        与 sklearn 的 predict_log_proba 等价:
            jll = X @ feature_log_prob_.T + class_log_prior_
        稀疏 X 只对非零项求和 (X 是 L2 归一的 tf 权重), 零特征不参与。
        """
        idx, vals = self.transform(text)
        lp = self.class_log_prior
        flp = self.feature_log_prob
        if idx is None:
            d = np.zeros(flp.shape[0])
        else:
            d = np.array([float(np.dot(vals, flp[c][idx])) for c in range(flp.shape[0])])
        jll = d + lp
        # sklearn 用 logsumexp 归一后取差; 二类时差就是直接相减
        return float(jll[1] - jll[0])

    def predict_proba(self, text):
        """返回 P(ai)"""
        z = self.decision_function(text)
        # 二类 softmax
        return float(1.0 / (1.0 + np.exp(-z)))


def try_load(model_dir, **kw):
    try:
        return FastNB(model_dir, **kw)
    except Exception:
        return None
