# -*- coding: utf-8 -*-
"""把训练好的 sklearn TF-IDF + LR 导出成纯 numpy 的 fast.npz

fast.npz 加载不触发 sklearn import, 冷启动 1.7s -> 0.3s。

支持:
  - 单特征: tfidf.pkl + clf.pkl
  - 双特征: v1.pkl + v2.pkl + clf.pkl (hstack, 英文模型)

用法:
    python scripts/export_fast.py ckpt/detector_zh_final
    python scripts/export_fast.py ckpt/detector_en_v5 --dual
"""
import os
import pickle
import argparse
import numpy as np


def _dump_vectorizer(vec, prefix=""):
    """把一个 fitted vectorizer 拆成可 npz 存的数组"""
    vocab = vec.vocabulary_
    keys = np.array(list(vocab.keys()), dtype=object)
    maxlen = max(len(k) for k in keys) if len(keys) else 1
    keys_u = np.array(list(vocab.keys()), dtype=f"U{maxlen}")
    idx = np.array([vocab[k] for k in keys_u], dtype=np.int32)
    order = np.argsort(idx)
    keys_u, idx = keys_u[order], idx[order]

    analyzer = {"char": 0, "char_wb": 1, "word": 2}.get(vec.analyzer, -1)
    return {
        f"{prefix}keys": keys_u,
        f"{prefix}idx": idx,
        f"{prefix}idf": np.asarray(vec.idf_, dtype=np.float32),
        f"{prefix}cfg": np.array([
            analyzer,
            vec.ngram_range[0], vec.ngram_range[1],
            1 if vec.sublinear_tf else 0,
            1 if getattr(vec, "use_idf", True) else 0,
            1 if getattr(vec, "smooth_idf", True) else 0,
            1 if getattr(vec, "lowercase", True) else 0,
            len(vocab),
        ], dtype=np.int32),
    }


def export(model_dir, dual=False):
    clf = pickle.load(open(os.path.join(model_dir, "clf.pkl"), "rb"))

    arrays = {}
    if dual:
        v1 = pickle.load(open(os.path.join(model_dir, "v1.pkl"), "rb"))
        v2 = pickle.load(open(os.path.join(model_dir, "v2.pkl"), "rb"))
        arrays.update(_dump_vectorizer(v1, "a_"))
        arrays.update(_dump_vectorizer(v2, "b_"))
        n_feat = arrays["a_cfg"][7] + arrays["b_cfg"][7]
        n_vec = 2
    else:
        vec = pickle.load(open(os.path.join(model_dir, "tfidf.pkl"), "rb"))
        arrays.update(_dump_vectorizer(vec, ""))
        n_feat = arrays["cfg"][7]
        n_vec = 1

    coef = np.asarray(clf.coef_, dtype=np.float32).ravel()
    if coef.shape[0] != n_feat:
        raise SystemExit(f"维度不符: coef {coef.shape[0]} vs 特征 {n_feat}")

    arrays["coef"] = coef
    arrays["intercept"] = np.asarray(clf.intercept_, dtype=np.float32)
    arrays["meta"] = np.array([n_vec, n_feat], dtype=np.int64)

    out = os.path.join(model_dir, "fast.npz")
    np.savez_compressed(out, **arrays)

    old = sum(
        os.path.getsize(os.path.join(model_dir, f))
        for f in os.listdir(model_dir) if f.endswith(".pkl")
    )
    new = os.path.getsize(out)
    print(f"✅ {out}")
    print(f"   特征维度 {n_feat:,} | 向量器 {n_vec} 个")
    print(f"   {old/1024/1024:.1f}MB (pkl) -> {new/1024/1024:.1f}MB (npz)")




def export_nb(model_dir):
    """Hashing+NB 档: vec.pkl + clf.pkl -> fast_nb.npz"""
    vec = pickle.load(open(os.path.join(model_dir, "vec.pkl"), "rb"))
    clf = pickle.load(open(os.path.join(model_dir, "clf.pkl"), "rb"))
    if not hasattr(clf, "feature_log_prob_"):
        raise SystemExit("不是 MultinomialNB, 请用默认导出模式")

    anm = {"char": 0, "char_wb": 1, "word": 2}.get(vec.analyzer, -1)
    if anm < 0:
        raise SystemExit(f"analyzer={vec.analyzer!r} 不支持")
    if getattr(vec, "alternate_sign", False):
        raise SystemExit("alternate_sign=True 不支持")

    cfg = np.array([
        anm, vec.ngram_range[0], vec.ngram_range[1],
        1 if vec.lowercase else 0,
        vec.n_features,
        int(getattr(vec, "seed", 0) or 0),
        1 if getattr(vec, "norm", "l2") == "l2" else 0,
    ], dtype=np.int64)

    out = os.path.join(model_dir, "fast_nb.npz")
    np.savez_compressed(
        out, cfg=cfg,
        class_log_prior=np.asarray(clf.class_log_prior_, dtype=np.float64),
        feature_log_prob=np.asarray(clf.feature_log_prob_, dtype=np.float32),
    )
    old = sum(os.path.getsize(os.path.join(model_dir, f))
              for f in ("vec.pkl", "clf.pkl")
              if os.path.exists(os.path.join(model_dir, f)))
    print(f"✅ {out}")
    print(f"   {old/1024/1024:.1f}MB (pkl) -> {os.path.getsize(out)/1024/1024:.1f}MB (npz)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--dual", action="store_true", help="v1.pkl + v2.pkl 双特征模型")
    ap.add_argument("--nb", action="store_true", help="HashingVectorizer + MultinomialNB 模型")
    a = ap.parse_args()
    if a.nb:
        export_nb(a.model_dir)
    else:
        export(a.model_dir, a.dual)
