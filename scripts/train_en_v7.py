# -*- coding: utf-8 -*-
import os as _os
# 可移植路径: 默认用仓库下的 data/ 和 ckpt/, 可用环境变量覆盖
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get("AIGC_DATA_DIR", _os.path.join(_ROOT, "data"))
CKPT_DIR = _os.environ.get("AIGC_CKPT_DIR", _os.path.join(_ROOT, "ckpt"))
"""英文v7 最终版: HC3-en原始QA + 当代AI + 分组防泄漏切分"""
import os, json, time, random, glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from scipy.sparse import hstack
import joblib

OUT = CKPT_DIR + r"\detector_en_v7"
os.makedirs(OUT, exist_ok=True)
T0=time.time()

# ---------- 数据 ----------
xs=[];ys=[];gs=[];n=0
for line in open(DATA_DIR + r"\hc3_en_real.jsonl",encoding="utf-8"):
    try: d=json.loads(line)
    except: continue
    n+=1
    for h in d.get("human_answers",[]):
        if isinstance(h,str) and 200<len(h.strip())<3000:
            xs.append(h.strip()); ys.append(0); gs.append(n)
    for a in d.get("chatgpt_answers",[]):
        if isinstance(a,str) and 200<len(a.strip())<3000:
            xs.append(a.strip()); ys.append(1); gs.append(n)
print(f"HC3-en: {n}QA -> 人={ys.count(0)} 旧AI={ys.count(1)}", flush=True)

import re
def clean(t):
    t=re.sub(r"^```.*?\n","",t); t=re.sub(r"```","",t)
    t=re.sub(r"^(Certainly|Sure|Of course)[!,.]\s*","",t,flags=re.I)
    return t.strip()

ai_new=[]
for fn in glob.glob(DATA_DIR + r"\ai_en_*.jsonl"):
    c=0
    for line in open(fn,encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        t=clean(d.get("text") or "")
        if 200<len(t)<3000: ai_new.append(t); c+=1
    print(f"  {os.path.basename(fn)}: +{c}", flush=True)

for i,t in enumerate(ai_new):
    xs.append(t); ys.append(1); gs.append(-(i+1))
print(f"当代AI: {len(ai_new)}  合计{len(xs)}", flush=True)

# ---------- 分组切分 ----------
random.seed(42)
ug=sorted(set(gs)); random.shuffle(ug)
n_test=int(len(ug)*0.20); n_ind=int((len(ug)-n_test)*0.18)
tg=set(ug[:n_test]); ig=set(ug[n_test:n_test+n_ind])
tri=[];tei=[];ini=[]
for i,g in enumerate(gs):
    if g in tg: tei.append(i)
    elif g in ig: ini.append(i)
    else: tri.append(i)
Xtr=[xs[i] for i in tri]; ytr=[ys[i] for i in tri]
Xte=[xs[i] for i in tei]; yte=[ys[i] for i in tei]
Xin=[xs[i] for i in ini]; yin=[ys[i] for i in ini]
print(f"切分: train={len(Xtr)} test={len(Xte)} ind={len(Xin)}", flush=True)

# ---------- 训练: char_wb + word 双特征 ----------
v1=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,max_features=200000,sublinear_tf=True)
v2=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=2,max_features=100000,
                   sublinear_tf=True,token_pattern=r'(?u)\b\w+\b')
A=hstack([v1.fit_transform(Xtr),v2.fit_transform(Xtr)]).tocsr()
B=hstack([v1.transform(Xte),v2.transform(Xte)]).tocsr()
C_=hstack([v1.transform(Xin),v2.transform(Xin)]).tocsr()
print(f"特征 {A.shape[1]} ({time.time()-T0:.0f}s)", flush=True)

best=(-1,None,None)
for c in [3,5,10]:
    clf=LogisticRegression(C=c,max_iter=1500,solver='liblinear',random_state=42)
    clf.fit(A,ytr)
    p=clf.predict_proba(B)[:,1]; f=f1_score(yte,(p>=0.5).astype(int)); a=roc_auc_score(yte,p)
    pi=clf.predict_proba(C_)[:,1]; fi=f1_score(yin,(pi>=0.5).astype(int))
    print(f"  C={c}: 测试F1={f:.4f} AUC={a:.4f} | 独立F1={fi:.4f}", flush=True)
    if fi>best[0]: best=(fi,clf,c)

fi,clf,C=best
joblib.dump(v1,os.path.join(OUT,'v1.joblib')); joblib.dump(v2,os.path.join(OUT,'v2.joblib'))
joblib.dump(clf,os.path.join(OUT,'clf.joblib'))

pi=clf.predict_proba(C_)[:,1]
bk={"短(<500)":[],"中(500-1500)":[],"长(>1500)":[]}
for t,y,p in zip(Xin,yin,pi):
    k="短(<500)" if len(t)<500 else ("中(500-1500)" if len(t)<=1500 else "长(>1500)")
    bk[k].append((y,int(p>=0.5)))
rep={"f1":round(fi,4),"C":C,"train_n":len(Xtr),"ind_n":len(Xin),"ai_new":len(ai_new),"model":"en_v7"}
for k,v in bk.items():
    if v: rep[k]={"f1":round(f1_score([a for a,b in v],[b for a,b in v]),4),"n":len(v)}
json.dump(rep,open(os.path.join(OUT,'report.json'),'w',encoding="utf-8"),ensure_ascii=False,indent=2)
print(f"\n完成 独立F1={fi:.4f} C={C} 用时{(time.time()-T0)/60:.1f}分", flush=True)
print(json.dumps(rep,ensure_ascii=False,indent=2), flush=True)
