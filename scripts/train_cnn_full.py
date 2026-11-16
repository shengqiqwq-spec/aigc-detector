# -*- coding: utf-8 -*-
"""Tier-3 CNN 全量数据训练 - 冲击更高F1"""
import os, json, time, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from collections import Counter

torch.set_num_threads(10)   # CPU 10线程榨干
random.seed(42); np.random.seed(42); torch.manual_seed(42)

OUT=r"D:\ai_train\ckpt\detector_cnn_full"
os.makedirs(OUT,exist_ok=True)
DATA=r"D:\ai_train\data\hc3_aug.jsonl"
MAX_LEN=192; VOCAB=8000; BATCH=64; EPOCHS=4

print(f"[{time.strftime('%H:%M:%S')}] 加载全量数据", flush=True)
texts,labels=[],[]
for line in open(DATA,encoding="utf-8"):
    try:d=json.loads(line)
    except:continue
    texts.append(d["text"]); labels.append(d["label"])
hi=[i for i,l in enumerate(labels) if l==0]; ai=[i for i,l in enumerate(labels) if l==1]
random.shuffle(hi); random.shuffle(ai)
n=min(len(hi),len(ai))   # 全量均衡,不截断
idx=hi[:n]+ai[:n]; random.shuffle(idx)
texts=[texts[i] for i in idx]; labels=[labels[i] for i in idx]
print(f"全量均衡: {len(texts)}", flush=True)

cnt=Counter()
for t in texts[:60000]: cnt.update(t)   # 用6万条建词表
chars=[c for c,_ in cnt.most_common(VOCAB)]
c2i={c:i+1 for i,c in enumerate(chars)}

def encode(t):
    ids=[c2i.get(c,0) for c in t[:MAX_LEN]]
    return ids+[0]*(MAX_LEN-len(ids))

class DS(Dataset):
    def __init__(self,ts,ls): self.x=[encode(t) for t in ts]; self.y=ls
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return torch.tensor(self.x[i]),torch.tensor(self.y[i])

tr_x,va_x,tr_y,va_y=train_test_split(texts,labels,test_size=0.12,random_state=42,stratify=labels)
print(f"训练{len(tr_x)} 验证{len(va_x)}, 编码中...", flush=True)
tr_dl=DataLoader(DS(tr_x,tr_y),batch_size=BATCH,shuffle=True,num_workers=0)
va_dl=DataLoader(DS(va_x,va_y),batch_size=BATCH,shuffle=False,num_workers=0)

class TextCNN(nn.Module):
    def __init__(self,v,embed=160):
        super().__init__()
        self.emb=nn.Embedding(v+1,embed,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(embed,128,k) for k in [2,3,4,5,6]])
        self.fc=nn.Linear(640,2); self.drop=nn.Dropout(0.35)
    def forward(self,x):
        e=self.emb(x).transpose(1,2)
        p=[torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(p,dim=1)))

model=TextCNN(len(c2i))
print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
crit=nn.CrossEntropyLoss()
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*len(tr_dl))

best=0
for ep in range(EPOCHS):
    model.train(); t0=time.time(); ls=0
    for i,(x,y) in enumerate(tr_dl):
        opt.zero_grad(); o=model(x); l=crit(o,y); l.backward(); opt.step(); sched.step(); ls+=l.item()
        if (i+1)%300==0: print(f"  ep{ep+1} {i+1}/{len(tr_dl)} loss={ls/(i+1):.4f} {time.time()-t0:.0f}s",flush=True)
    model.eval(); pr=[];pb=[];gt=[]
    with torch.no_grad():
        for x,y in va_dl:
            o=model(x);p=torch.softmax(o,dim=1)
            pb.extend(p[:,1].numpy());pr.extend(p.argmax(1).numpy());gt.extend(y.numpy())
    f1=f1_score(gt,pr);auc=roc_auc_score(gt,pb);acc=accuracy_score(gt,pr)
    print(f"[{time.strftime('%H:%M:%S')}] ep{ep+1} F1={f1:.4f} AUC={auc:.4f} ACC={acc:.4f}",flush=True)
    if f1>best:
        best=f1
        torch.save(model.state_dict(),os.path.join(OUT,"model.pt"))
        json.dump({"c2i":c2i,"MAX_LEN":MAX_LEN},open(os.path.join(OUT,"vocab.json"),"w",encoding="utf-8"))

json.dump({"f1":round(best,4),"params_m":round(sum(p.numel() for p in model.parameters())/1e6,2),
    "train_n":len(tr_x),"val_n":len(va_x),"epochs":EPOCHS},
    open(os.path.join(OUT,"report.json"),"w"),ensure_ascii=False,indent=2)
print(f"完成 best F1={best:.4f}",flush=True)
