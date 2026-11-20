
import os,json,glob,warnings,time
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
DATA=r"D:\ai_train\data"
def load(pfx,exclude=()):
    xs,ys=[],[]
    hc3=os.path.join(DATA,"hc3_all.jsonl" if pfx.startswith("zh") else "hc3_en_real.jsonl")
    if not os.path.exists(hc3):hc3=os.path.join(DATA,"hc3_en_clean.jsonl")
    for line in open(hc3,encoding="utf-8"):
        try:d=json.loads(line)
        except:continue
        for a in (d.get("human_answers") or []):
            if isinstance(a,str) and 20<len(a.strip())<2000:xs.append(a.strip());ys.append(0)
        for a in (d.get("chatgpt_answers") or []):
            if isinstance(a,str) and 20<len(a.strip())<2000:xs.append(a.strip());ys.append(1)
    pattern="ai_"+pfx[2:]+"_" if pfx.startswith("ai") else pfx+"_"
    realpfx="ai_zh_" if pfx.startswith("zh") else "ai_en_"
    for p in sorted(glob.glob(os.path.join(DATA,realpfx+"*.jsonl"))):
        b=os.path.basename(p)
        if b in exclude:continue
        if "functional" in b:continue
        for line in open(p,encoding="utf-8"):
            try:d=json.loads(line)
            except:continue
            t=(d.get("text") or "").strip()
            if len(t)>=20:xs.append(t);ys.append(1)
    return xs,np.array(ys)

def holdout(pfx,fn):
    Xte,yte=[],[]
    for line in open(os.path.join(DATA,fn),encoding="utf-8"):
        try:d=json.loads(line)
        except:continue
        t=(d.get("text") or "").strip()
        if len(t)>=20:Xte.append(t);yte.append(1)
    Xtr,ytr=load(pfx,exclude=(fn,))
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,max_features=300000,sublinear_tf=True)
    A=v.fit_transform(Xtr);B=v.transform(Xte)
    clf=LogisticRegression(C=30,class_weight="balanced",max_iter=1500,random_state=42)
    clf.fit(A,ytr);pred=clf.predict(B)
    return float((pred==np.array(yte)).mean()),len(Xte)

for fn in ("ai_zh_modern2.jsonl","ai_zh_short2.jsonl","ai_zh_vl30b2.jsonl","ai_en_npu2.jsonl"):
    pfx="zh" if "_zh_" in fn else "en"
    acc,n=holdout(pfx,fn)
    print(f"{fn:26s} 二分类话题外准确率 {acc*100:5.1f}% (n={n})",flush=True)
