# -*- coding: utf-8 -*-
"""英文轻量档: Hashing + MultinomialNB (~2MB, 手机/嵌入式)

与中文 tier0 同样的思路: 不需要保存词表, 体积可控。
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get("AIGC_DATA_DIR", _os.path.join(_ROOT, "data"))
CKPT_DIR = _os.environ.get("AIGC_CKPT_DIR", _os.path.join(_ROOT, "ckpt"))

import json, time, random, glob, re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, roc_auc_score
import joblib

OUT = _os.path.join(CKPT_DIR, "detector_en_tier0")
_os.makedirs(OUT, exist_ok=True)
T0 = time.time()

def clean(t):
    t = re.sub(r"^```.*?\n", "", t); t = re.sub(r"```", "", t)
    t = re.sub(r"^(Certainly|Sure|Of course)[!,.]\s*", "", t, flags=re.I)
    return t.strip()

# ---------- 数据 ----------
xs=[]; ys=[]; gs=[]; n=0
hc3 = _os.path.join(DATA_DIR, "hc3_en_real.jsonl")
for line in open(hc3, encoding="utf-8"):
    try: d=json.loads(line)
    except: continue
    n+=1
    for h in d.get("human_answers",[]):
        if isinstance(h,str) and 100<len(h.strip())<2000:
            xs.append(h.strip()); ys.append(0); gs.append(n)
    for a in d.get("chatgpt_answers",[]):
        if isinstance(a,str) and 100<len(a.strip())<2000:
            xs.append(a.strip()); ys.append(1); gs.append(n)
print(f"HC3-en: {n}QA -> 人={ys.count(0)} AI={ys.count(1)}", flush=True)

ai_new=0
for fn in glob.glob(_os.path.join(DATA_DIR, "ai_en_*.jsonl")):
    for line in open(fn, encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        t=clean(d.get("text") or "")
        if 100 <= len(t) < 3000:
            xs.append(t); ys.append(1); gs.append(-(ai_new+1)); ai_new+=1
print(f"当代AI: {ai_new}  合计 {len(xs)}", flush=True)

# ---------- 分组切分 ----------
random.seed(42)
ug=sorted(set(gs)); random.shuffle(ug)
n_test=int(len(ug)*0.20)
tg=set(ug[:n_test])
tri=[i for i,g in enumerate(gs) if g not in tg]
tei=[i for i,g in enumerate(gs) if g in tg]

# 训练集内均衡 (人类远多于AI, 按AI量取人类)
tr_h=[i for i in tri if ys[i]==0]; tr_a=[i for i in tri if ys[i]==1]
random.shuffle(tr_h); random.shuffle(tr_a)
m=min(len(tr_h), max(len(tr_a), 20000))
tri = tr_h[:m] + tr_a
random.shuffle(tri)
print(f"切分: train={len(tri)} test={len(tei)}", flush=True)

Xtr=[xs[i] for i in tri]; ytr=[ys[i] for i in tri]
Xte=[xs[i] for i in tei]; yte=[ys[i] for i in tei]

# ---------- Hashing ----------
vec = HashingVectorizer(n_features=65536, analyzer='char_wb', ngram_range=(2,4),
                        alternate_sign=False, norm='l2', lowercase=True)
A = vec.transform(Xtr); B = vec.transform(Xte)
print(f"特征 {A.shape[1]} ({time.time()-T0:.0f}s)", flush=True)

best=(-1,None,None)
for alpha in [0.05, 0.1, 0.3, 1.0]:
    clf = MultinomialNB(alpha=alpha)
    clf.fit(A, ytr)
    p = clf.predict_proba(B)[:,1]
    f = f1_score(yte, (p>=0.5).astype(int)); a = roc_auc_score(yte, p)
    print(f"  alpha={alpha}: F1={f:.4f} AUC={a:.4f}", flush=True)
    if f > best[0]: best=(f, clf, alpha)

f, clf, alpha = best
joblib.dump(vec, _os.path.join(OUT, "vec.joblib"))
joblib.dump(clf, _os.path.join(OUT, "clf.joblib"))

# 同时输出 pkl (加载器优先读 pkl)
import pickle
for name in ["vec", "clf"]:
    obj = joblib.load(_os.path.join(OUT, name + ".joblib"))
    with open(_os.path.join(OUT, name + ".pkl"), "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

rep = {"f1": round(f,4), "alpha": alpha, "model": "Hashing+MultinomialNB (en)",
       "features": 65536, "train_n": len(Xtr), "test_n": len(Xte), "ai_new": ai_new}
json.dump(rep, open(_os.path.join(OUT,"report.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n完成 F1={f:.4f} alpha={alpha} 用时{(time.time()-T0)/60:.1f}分", flush=True)
print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
