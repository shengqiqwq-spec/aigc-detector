# -*- coding: utf-8 -*-
"""中文检测器 v5: HC3历史 + Qwen当代AI数据混合
解决: 对当代新式AI文本漏判(HC3是2022年数据)
"""
import os, json, random, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib

OUT = r"D:\ai_train\ckpt\detector_zh_v5"
os.makedirs(OUT, exist_ok=True)

def clean(t):
    t = re.sub(r"^\s*(作为|我是)?(一个)?(AI|人工智能|语言模型|助手)[^，。]*[，。]\s*", "", t)
    t = re.sub(r"^\s*(好的|当然|没问题)[，。！]\s*", "", t)
    return t.strip()

print(f"[{time.strftime('%H:%M:%S')}] 加载数据", flush=True)

# HC3 历史数据
texts, labels = [], []
for line in open(r"D:\ai_train\data\hc3_aug.jsonl", encoding="utf-8"):
    try: d=json.loads(line)
    except: continue
    texts.append(d["text"]); labels.append(d["label"])
hc3_n = len(texts)
print(f"HC3: {hc3_n}", flush=True)

# 当代 AI 数据
modern=[]
for fn in ["ai_zh_modern.jsonl","ai_zh_modern2.jsonl"]:
    p=os.path.join(r"D:\ai_train\data",fn)
    if not os.path.exists(p): continue
    for line in open(p,encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        t=clean(d.get("text",""))
        if len(t)>100: modern.append(t)
print(f"当代AI: {len(modern)}", flush=True)

# 合并: HC3均衡 + 当代AI全部(重复采样放大权重)
hi=[i for i,l in enumerate(labels) if l==0]; ai=[i for i,l in enumerate(labels) if l==1]
random.seed(42); random.shuffle(hi); random.shuffle(ai)
n=min(len(hi),len(ai))
base_x=[texts[i] for i in hi[:n]]+[texts[i] for i in ai[:n]]
base_y=[0]*n+[1]*n

# 当代AI放大3倍权重
for _ in range(3):
    base_x += modern; base_y += [1]*len(modern)

idx=list(range(len(base_x))); random.shuffle(idx)
base_x=[base_x[i] for i in idx]; base_y=[base_y[i] for i in idx]
print(f"合并: {len(base_x)} (人{base_y.count(0)}/AI{base_y.count(1)})", flush=True)

tr_x,va_x,tr_y,va_y=train_test_split(base_x,base_y,test_size=0.15,random_state=42,stratify=base_y)
vec=TfidfVectorizer(analyzer='char',ngram_range=(1,5),min_df=2,max_features=250000,sublinear_tf=True)
Xtr=vec.fit_transform(tr_x); Xva=vec.transform(va_x)
print(f"特征 {Xtr.shape[1]}", flush=True)

best=(-1,None)
for C in [4,8,12]:
    clf=LogisticRegression(C=C,max_iter=1500,solver='liblinear',random_state=42)
    clf.fit(Xtr,tr_y)
    p=clf.predict_proba(Xva)[:,1]
    f=f1_score(va_y,(p>=0.5).astype(int)); a=roc_auc_score(va_y,p)
    print(f"  C={C}: F1={f:.4f} AUC={a:.4f}", flush=True)
    if f>best[0]: best=(f,clf)

joblib.dump(vec,os.path.join(OUT,'tfidf.joblib'))
joblib.dump(best[1],os.path.join(OUT,'clf.joblib'))
json.dump({"f1":round(best[0],4),"hc3":hc3_n,"modern_ai":len(modern),"total":len(base_x)},
          open(os.path.join(OUT,'report.json'),'w'),ensure_ascii=False,indent=2)
print(f"完成 F1={best[0]:.4f} -> "+OUT, flush=True)
