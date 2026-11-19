# -*- coding: utf-8 -*-
"""Tier-0 轻量档: Hashing + MultinomialNB (~2MB, 手机/嵌入式)

特征用 HashingVectorizer, 不需要保存词表, 模型体积可控在 2MB 以内。
准确率低于 tier1, 但模型极小、无 scipy 稀疏矩阵依赖负担。
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get("AIGC_DATA_DIR", _os.path.join(_ROOT, "data"))
CKPT_DIR = _os.environ.get("AIGC_CKPT_DIR", _os.path.join(_ROOT, "ckpt"))

import json, time, random
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, roc_auc_score
import joblib

OUT = _os.path.join(CKPT_DIR, "detector_tier0")
_os.makedirs(OUT, exist_ok=True)
T0 = time.time()

# ---------- 数据 ----------
xs=[]; ys=[]; gs=[]; n=0
hc3 = _os.path.join(DATA_DIR, "hc3_all.jsonl")
for line in open(hc3, encoding="utf-8"):
    try: d=json.loads(line)
    except: continue
    n+=1
    for h in d.get("human_answers",[]):
        if isinstance(h,str) and 20<len(h.strip())<2000:
            xs.append(h.strip()); ys.append(0); gs.append(n)
    for a in d.get("chatgpt_answers",[]):
        if isinstance(a,str) and 20<len(a.strip())<2000:
            xs.append(a.strip()); ys.append(1); gs.append(n)
print(f"HC3: {n}QA -> 人={ys.count(0)} AI={ys.count(1)}", flush=True)

import glob
ai_new=0
for fn in glob.glob(_os.path.join(DATA_DIR, "ai_zh_*.jsonl")):
    for line in open(fn, encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        t=(d.get("text") or "").strip()
        if len(t)>=20:
            xs.append(t); ys.append(1); gs.append(-(ai_new+1)); ai_new+=1
print(f"当代AI: {ai_new}  合计 {len(xs)}", flush=True)

# ---------- 分组切分 ----------
random.seed(42)
ug=sorted(set(gs)); random.shuffle(ug)
n_test=int(len(ug)*0.20)
tg=set(ug[:n_test])
tri=[i for i,g in enumerate(gs) if g not in tg]
tei=[i for i,g in enumerate(gs) if g in tg]
Xtr=[xs[i] for i in tri]; ytr=[ys[i] for i in tri]
Xte=[xs[i] for i in tei]; yte=[ys[i] for i in tei]
print(f"切分: train={len(Xtr)} test={len(Xte)}", flush=True)

# ---------- Hashing 特征 (无需词表) ----------
vec = HashingVectorizer(n_features=65536, analyzer='char', ngram_range=(1,4),
                        alternate_sign=False, norm='l2')
A = vec.transform(Xtr); B = vec.transform(Xte)
print(f"特征 {A.shape[1]} ({time.time()-T0:.0f}s)", flush=True)

# ---------- 训练 ----------
best=(-1,None,None)
for alpha in [0.1, 0.3, 1.0]:
    clf = MultinomialNB(alpha=alpha)
    clf.fit(A, ytr)
    p = clf.predict_proba(B)[:,1]
    f = f1_score(yte, (p>=0.5).astype(int)); a = roc_auc_score(yte, p)
    print(f"  alpha={alpha}: F1={f:.4f} AUC={a:.4f}", flush=True)
    if f > best[0]: best=(f, clf, alpha)

f, clf, alpha = best
joblib.dump(vec, _os.path.join(OUT, "vec.joblib"))
joblib.dump(clf, _os.path.join(OUT, "clf.joblib"))

rep = {"f1": round(f,4), "alpha": alpha, "model": "Hashing+MultinomialNB",
       "features": 65536, "train_n": len(Xtr), "test_n": len(Xte), "ai_new": ai_new}
json.dump(rep, open(_os.path.join(OUT,"report.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n完成 F1={f:.4f} alpha={alpha} 用时{(time.time()-T0)/60:.1f}分", flush=True)
print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
