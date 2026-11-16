# -*- coding: utf-8 -*-
"""英文v4: 合并两批当代AI数据(500条) + HC3人类, 更稳"""
import os, json, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib

OUT = r"D:\ai_train\ckpt\detector_en_v4"
os.makedirs(OUT, exist_ok=True)

def clean_ai(t):
    t = re.sub(r"^As (Qwen|an AI|a language model|an assistant|an AI assistant)[^.]*\.\s*", "", t, flags=re.I)
    t = re.sub(r"^(Certainly|Sure|Of course|Absolutely)[!,.]\s*", "", t, flags=re.I)
    t = re.sub(r"^Here (are|is)[^:]*:\s*", "", t, flags=re.I)
    return t.strip()

ai=[]
for fn in ["ai_en_npu.jsonl","ai_en_npu2.jsonl"]:
    p=os.path.join(r"D:\ai_train\data",fn)
    if not os.path.exists(p): continue
    for line in open(p,encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        t=clean_ai(d.get("text",""))
        if len(t)>100: ai.append(t)

human=[]
for line in open(r"D:\ai_train\data\hc3_en_clean.jsonl",encoding="utf-8"):
    try: d=json.loads(line)
    except: continue
    for h in d.get("human_answers",[]):
        if isinstance(h,str) and 100<len(h.strip())<2000: human.append(h.strip())

print(f"AI(当代,清洗)={len(ai)}  人类={len(human)}")
# AI少, 人类取等量(多次采样)
random.seed(42); random.shuffle(ai); random.shuffle(human)
n=len(ai)
human_use=human[:n]
texts=human_use+ai; labels=[0]*n+[1]*n
idx=list(range(len(texts))); random.shuffle(idx)
texts=[texts[i] for i in idx]; labels=[labels[i] for i in idx]
print(f"均衡: {len(texts)}")

tr_x,va_x,tr_y,va_y=train_test_split(texts,labels,test_size=0.25,random_state=42,stratify=labels)
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,max_features=200000,sublinear_tf=True)
Xtr=vec.fit_transform(tr_x); Xva=vec.transform(va_x)
best=(-1,None,None)
for C in [2,5,10]:
    clf=LogisticRegression(C=C,max_iter=1000,solver='liblinear',random_state=42)
    clf.fit(Xtr,tr_y)
    p=clf.predict_proba(Xva)[:,1]
    f=f1_score(va_y,(p>=0.5).astype(int)); a=roc_auc_score(va_y,p)
    print(f"  C={C}: F1={f:.4f} AUC={a:.4f}")
    if f>best[0]: best=(f,C,clf)

joblib.dump(vec,os.path.join(OUT,'tfidf.joblib'))
joblib.dump(best[2],os.path.join(OUT,'clf.joblib'))
json.dump({"f1":round(best[0],4),"C":best[1],"ai_n":len(ai),"total":len(texts)},
          open(os.path.join(OUT,'report.json'),'w'),ensure_ascii=False,indent=2)
print(f"完成 F1={best[0]:.4f} -> "+OUT)
