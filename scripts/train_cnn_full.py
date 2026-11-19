# -*- coding: utf-8 -*-
"""Tier-3 CharTextCNN - 极速推理档 (需 torch, ~5MB, 单次 ~20ms)

与 tier1 共用同一套数据管线: HC3 原始 QA 对 + 当代 AI 文本,
并按原问题分组切分 (同一问题的 Q/A 不跨训练集/测试集)。
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get("AIGC_DATA_DIR", _os.path.join(_ROOT, "data"))
CKPT_DIR = _os.environ.get("AIGC_CKPT_DIR", _os.path.join(_ROOT, "ckpt"))

import json, time, random, glob, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from collections import Counter

torch.set_num_threads(10)
random.seed(42); np.random.seed(42); torch.manual_seed(42)

OUT = _os.path.join(CKPT_DIR, "detector_cnn_full")
_os.makedirs(OUT, exist_ok=True)
MAX_LEN=192; VOCAB=8000; BATCH=64; EPOCHS=3

# ---------- 数据 (与 tier1 同源) ----------
print(f"[{time.strftime('%H:%M:%S')}] 加载数据", flush=True)
xs=[]; ys=[]; gs=[]; n=0
hc3 = _os.path.join(DATA_DIR, "hc3_all.jsonl")
for line in open(hc3, encoding="utf-8"):
    try: d=json.loads(line)
    except: continue
    n+=1
    for h in d.get("human_answers",[]):
        if isinstance(h,str) and 20<len(h.strip())<1200:
            xs.append(h.strip()); ys.append(0); gs.append(n)
    for a in d.get("chatgpt_answers",[]):
        if isinstance(a,str) and 20<len(a.strip())<1200:
            xs.append(a.strip()); ys.append(1); gs.append(n)
print(f"HC3: {n}QA -> 人={ys.count(0)} AI={ys.count(1)}", flush=True)

ai_new=0
for fn in glob.glob(_os.path.join(DATA_DIR, "ai_zh_*.jsonl")):
    for line in open(fn, encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        t=(d.get("text") or "").strip()
        if 20 <= len(t) < 1200:
            xs.append(t); ys.append(1); gs.append(-(ai_new+1)); ai_new+=1
print(f"当代AI: {ai_new}  合计 {len(xs)}", flush=True)

# ---------- 分组切分 ----------
random.seed(42)
ug=sorted(set(gs)); random.shuffle(ug)
n_test=int(len(ug)*0.20)
tg=set(ug[:n_test])
tri=[i for i,g in enumerate(gs) if g not in tg]
tei=[i for i,g in enumerate(gs) if g in tg]

# 均衡: 训练集内人类/AI 取较小方
tr_h=[i for i in tri if ys[i]==0]; tr_a=[i for i in tri if ys[i]==1]
random.shuffle(tr_h); random.shuffle(tr_a)
m=min(len(tr_h), len(tr_a))
tri = tr_h[:m] + tr_a[:m]
random.shuffle(tri)
print(f"切分: train={len(tri)} (均衡{m*2}) test={len(tei)}", flush=True)

Xtr=[xs[i] for i in tri]; ytr=[ys[i] for i in tri]
Xte=[xs[i] for i in tei]; yte=[ys[i] for i in tei]

# ---------- 词表 ----------
cnt=Counter()
for t in Xtr[:60000]: cnt.update(t)
chars=[c for c,_ in cnt.most_common(VOCAB)]
c2i={c:i+1 for i,c in enumerate(chars)}
print(f"词表: {len(c2i)}", flush=True)

def encode(t):
    ids=[c2i.get(c,0) for c in t[:MAX_LEN]]
    return ids+[0]*(MAX_LEN-len(ids))

class DS(Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(encode(self.X[i]), dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.float))

class CNN(nn.Module):
    def __init__(self, vocab, dim=64):
        super().__init__()
        self.emb=nn.Embedding(vocab+1, dim, padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(dim, 96, k) for k in (2,3,4,5)])
        self.drop=nn.Dropout(0.3)
        self.fc=nn.Linear(96*4, 1)
    def forward(self, x):
        e=self.emb(x).transpose(1,2)
        h=[torch.relu(c(e)).max(dim=2)[0] for c in self.convs]
        return self.fc(self.drop(torch.cat(h,1))).squeeze(1)

dev="cuda" if torch.cuda.is_available() else "cpu"
model=CNN(len(c2i)).to(dev)
opt=torch.optim.Adam(model.parameters(), lr=1e-3)
lossf=nn.BCEWithLogitsLoss()
print(f"设备: {dev}, 参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

dl=DataLoader(DS(Xtr,ytr), batch_size=BATCH, shuffle=True, num_workers=0)
dlv=DataLoader(DS(Xte,yte), batch_size=128, shuffle=False, num_workers=0)

best_f1=-1
for ep in range(1, EPOCHS+1):
    model.train(); tot=0
    for i,(xb,yb) in enumerate(dl):
        xb,yb=xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss=lossf(model(xb), yb); loss.backward(); opt.step()
        tot+=loss.item()
        if (i+1)%300==0:
            print(f"  ep{ep} {i+1}/{len(dl)} loss={tot/(i+1):.4f}", flush=True)
    model.eval(); ps=[]; ts=[]
    with torch.no_grad():
        for xb,yb in dlv:
            ps += model(xb.to(dev)).cpu().numpy().tolist()
            ts += yb.numpy().tolist()
    p=np.array(ps); t=np.array(ts)
    pr=(p>=0).astype(int)
    f=f1_score(t,pr); a=roc_auc_score(t,p)
    print(f"[{time.strftime('%H:%M:%S')}] ep{ep} F1={f:.4f} AUC={a:.4f}", flush=True)
    if f>best_f1:
        best_f1=f
        torch.save({"state_dict":model.state_dict(),"c2i":c2i,"max_len":MAX_LEN,"vocab_size":len(c2i)},
                   _os.path.join(OUT,"model.pt"))

rep={"f1":round(best_f1,4),"params_m":round(sum(p.numel() for p in model.parameters())/1e6,2),
     "train_n":len(Xtr),"test_n":len(Xte),"epochs":EPOCHS,"ai_new":ai_new,
     "model":"CharTextCNN","split":"grouped"}
json.dump(rep, open(_os.path.join(OUT,"report.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n完成 best F1={best_f1:.4f}", flush=True)
print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
