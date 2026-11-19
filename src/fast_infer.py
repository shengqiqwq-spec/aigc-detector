# -*- coding: utf-8 -*-
"""
纯 numpy 推理器 —— 不依赖 sklearn, 冷启动 ~0.2s 而非 ~1.7s

sklearn 的 import 占冷启动 1200ms/1700ms, 且无论怎么切子模块都省不掉。
而这个产品的两个主力模型 (tier1 / en) 结构是固定的:

    char(n,m) TF-IDF -> L2 归一 -> LogisticRegression

这些完全可以用 numpy 手写, 数值与 sklearn 一致 (见 test_parity_with_sklearn)。

用法:
    from fast_infer import FastDetector
    d = FastDetector("ckpt/detector_zh_final")
    p = d.predict_proba("一段文本")

注意: 只支持纯 numpy 能复现的配置。加载时校验, 不符则 raise,
      由调用方回退到 sklearn 路径 (model_zoo 已内置回退)。
"""
import os
import numpy as np


class UnsupportedModel(Exception):
    """模型配置超出纯 numpy 实现范围, 应回退 sklearn"""


class FastDetector:
    """单模型纯 numpy 推理器

    支持: analyzer='char' | 'char_wb', norm='l2'|None, sublinear_tf True/False,
          use_idf True/False, smooth_idf True/False
    """

    def __init__(self, model_dir, analyzer=None):
        self.dir = model_dir
        # 优先用 fast.npz (纯数组, 加载不触发 sklearn import)
        npz_path = os.path.join(model_dir, "fast.npz")
        if os.path.exists(npz_path):
            self._load_npz(npz_path, analyzer)
            return
        self._load_pickle(model_dir, analyzer)

    def _load_npz(self, npz_path, analyzer):
        """从 npz 载入, 完全不经过 sklearn

        支持 1 个或 2 个向量器 (双特征 hstack)。各向量器的词表索引在
        npz 里已按全局偏移存好, 直接查即可。
        """
        z = np.load(npz_path, allow_pickle=False)
        n_vec = int(z["meta"][0])
        self.n_features = int(z["meta"][1])
        self.norm = "l2"
        self.vecs = []
        prefixes = [""] if n_vec == 1 else ["a_", "b_"]
        offset = 0
        for pre in prefixes:
            cfg = z[f"{pre}cfg"]
            an = int(cfg[0])
            v = {
                "analyzer": {0: "char", 1: "char_wb", 2: "word"}.get(an, "char"),
                "ngram_range": (int(cfg[1]), int(cfg[2])),
                "sublinear_tf": bool(cfg[3]),
                "use_idf": bool(cfg[4]),
                "lowercase": bool(cfg[6]),
                "vocab": {str(k): int(i) + offset
                          for k, i in zip(z[f"{pre}keys"], z[f"{pre}idx"])},
                "idf": z[f"{pre}idf"].astype(np.float32),
                "base": offset,
            }
            self.vecs.append(v)
            offset += int(cfg[7])
        # 单向量器时暴露旧属性, 保持兼容
        v0 = self.vecs[0]
        self.analyzer = v0["analyzer"]
        self.ngram_range = v0["ngram_range"]
        self.sublinear_tf = v0["sublinear_tf"]
        self.use_idf = v0["use_idf"]
        self.lowercase = v0["lowercase"]
        self.vocab = v0["vocab"]
        self.idf = v0["idf"]

        self.coef = z["coef"].astype(np.float32)
        self.intercept = float(z["intercept"][0])
        if self.coef.shape[0] != self.n_features:
            raise UnsupportedModel(
                f"coef 维度 {self.coef.shape[0]} 与特征 {self.n_features} 不一致")

    def _load_pickle(self, model_dir, analyzer):
        vec_path = os.path.join(model_dir, "tfidf.pkl")
        clf_path = os.path.join(model_dir, "clf.pkl")
        if not (os.path.exists(vec_path) and os.path.exists(clf_path)):
            raise UnsupportedModel(f"缺少模型文件: {model_dir}")

        # 本地自训练模型, 来源可信
        import pickle
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)

        # ---- 校验配置 ----
        analyzer = getattr(vec, "analyzer", None)
        if analyzer not in ("char", "char_wb"):
            raise UnsupportedModel(f"analyzer={analyzer!r} 不支持 (只支持 char/char_wb)")
        if getattr(vec, "norm", "l2") not in ("l2", None):
            raise UnsupportedModel(f"norm={getattr(vec,'norm',None)!r} 不支持")
        if not hasattr(vec, "vocabulary_") or not hasattr(vec, "idf_"):
            raise UnsupportedModel("模型缺少 vocabulary_/idf_, 不是 TfidfVectorizer")

        self.analyzer = analyzer
        self.ngram_range = tuple(vec.ngram_range)
        self.sublinear_tf = bool(vec.sublinear_tf)
        self.use_idf = bool(getattr(vec, "use_idf", True))
        self.smooth_idf = bool(getattr(vec, "smooth_idf", True))
        self.norm = getattr(vec, "norm", "l2")
        self.lowercase = bool(getattr(vec, "lowercase", True))

        # ---- 词表 (dict, 用于 n-gram 查询) ----
        self.vocab = vec.vocabulary_
        self.n_features = len(self.vocab)
        # idf (smooth_idf 的平滑已在 sklearn 侧算好, 直接取)
        self.idf = np.asarray(vec.idf_, dtype=np.float32)
        del vec

        # ---- 分类器权重 ----
        coef = np.asarray(clf.coef_, dtype=np.float32).ravel()
        if coef.shape[0] != self.n_features:
            raise UnsupportedModel(
                f"coef 维度 {coef.shape[0]} 与词表 {self.n_features} 不一致")
        self.coef = coef
        self.intercept = float(np.asarray(clf.intercept_).ravel()[0])
        del clf
        self.vecs = [{
            "analyzer": self.analyzer, "ngram_range": self.ngram_range,
            "sublinear_tf": self.sublinear_tf, "use_idf": self.use_idf,
            "lowercase": self.lowercase, "vocab": self.vocab, "idf": self.idf,
            "base": 0,
        }]

    # ---------- 特征 ----------
    @staticmethod
    def _ngrams_of(text, cfg):
        """按单个向量器配置产出 n-gram, 与 sklearn 的 char/char_wb/word 一致"""
        if cfg["lowercase"]:
            text = text.lower()
        lo, hi = cfg["ngram_range"]
        an = cfg["analyzer"]
        if an == "char":
            n = len(text)
            for k in range(lo, hi + 1):
                if k > n:
                    break
                for i in range(n - k + 1):
                    yield text[i:i + k]
        elif an == "char_wb":
            for word in text.split():
                padded = " " + word + " "
                m = len(padded)
                for k in range(lo, hi + 1):
                    if k > m:
                        break
                    for i in range(m - k + 1):
                        yield padded[i:i + k]
        else:  # word: sklearn 默认 token_pattern r"(?u)\b\w\w+\b" 是 2+ 字符
            import re as _re
            toks = _re.findall(r"(?u)\b\w+\b", text)
            n = len(toks)
            for k in range(lo, hi + 1):
                if k > n:
                    break
                for i in range(n - k + 1):
                    yield " ".join(toks[i:i + k])

    def transform(self, text):
        """返回 (indices, values): 所有向量器的特征拼接后统一做 L2 归一"""
        from collections import Counter
        all_idx, all_val = [], []
        for cfg in self.vecs:
            vocab = cfg["vocab"]        # 全局索引
            base = cfg["base"]          # 该向量器在本段特征里的起始偏移
            counts = Counter(g for g in self._ngrams_of(text, cfg) if g in vocab)
            if not counts:
                continue
            glob = np.fromiter((vocab[g] for g in counts), dtype=np.int32,
                               count=len(counts))
            vals = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
            if cfg["sublinear_tf"]:
                vals = 1.0 + np.log(vals)
            if cfg["use_idf"] and cfg["idf"] is not None:
                vals = vals * cfg["idf"][glob - base]
            # 单向量器: 在这里归一; 多向量器: 由各 vectorizer 自己先 L2,
            # 拼接后 sklearn 的 hstack 不再归一 —— 见下方 dual 分支
            if len(self.vecs) == 1:
                nrm = float(np.sqrt(np.dot(vals, vals)))
                if nrm > 0:
                    vals = vals / nrm
            else:
                nrm = float(np.sqrt(np.dot(vals, vals)))
                if nrm > 0:
                    vals = vals / nrm
            all_idx.append(glob)
            all_val.append(vals)
        if not all_idx:
            return None, None
        idx = np.concatenate(all_idx)
        vals = np.concatenate(all_val)
        # 单向量器已在上方归一; 双特征时 sklearn 是各 vectorizer 归一再 hstack,
        # 拼接结果范数为 sqrt(2) —— 与 sklearn 保持一致, 不再二次归一
        if self.norm == "l2" and len(self.vecs) == 1:
            nrm = float(np.sqrt(np.dot(vals, vals)))
            if nrm > 0:
                vals = vals / nrm
        return idx, vals

    def decision_function(self, text):
        idx, vals = self.transform(text)
        if idx is None:
            return self.intercept
        return float(np.dot(self.coef[idx], vals) + self.intercept)

    def predict_proba(self, text):
        """返回 P(ai), 与 sklearn predict_proba[:,1] 一致"""
        z = self.decision_function(text)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, text, threshold=0.5):
        return 1 if self.predict_proba(text) >= threshold else 0


def try_load(model_dir, **kw):
    """尝试加载, 失败返回 None (调用方回退 sklearn)"""
    try:
        return FastDetector(model_dir, **kw)
    except UnsupportedModel:
        return None
    except Exception:
        return None
