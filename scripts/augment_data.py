# -*- coding: utf-8 -*-
"""
数据增强: 句子级切分, 补短文本样本
问题: 训练集文本中位154字, 短文本(<80字)是分布外, 所有模型低估
方案: 把长文本按句拆成短样本, 连同原文一起训练
"""
import json, re, random

SRC=r"D:\ai_train\data\hc3_all.jsonl"
OUT=r"D:\ai_train\data\hc3_aug.jsonl"
random.seed(42)

def split_sents(t, min_len=20, max_len=120):
    parts=[s.strip() for s in re.split(r"[。！？!?；;\n]", t) if s.strip()]
    res=[]
    for p in parts:
        if min_len<=len(p)<=max_len: res.append(p)
        elif len(p)>max_len:
            # 太长再按逗号切
            sub=[x.strip() for x in re.split(r"[，,]",p) if x.strip()]
            cur=""
            for x in sub:
                if len(cur)+len(x)<=max_len: cur+=x+"，"
                else:
                    if len(cur)>=min_len: res.append(cur.rstrip("，"))
                    cur=x+"，"
            if len(cur)>=min_len: res.append(cur.rstrip("，"))
    return res

rows=[]
n_orig=0;n_aug=0
for line in open(SRC,encoding="utf-8"):
    try:d=json.loads(line)
    except:continue
    for h in d.get("human_answers",[]):
        if not(isinstance(h,str) and len(h.strip())>15):continue
        rows.append({"text":h.strip(),"label":0});n_orig+=1
        for s in split_sents(h):
            rows.append({"text":s,"label":0});n_aug+=1
    for c in d.get("chatgpt_answers",[]):
        if not(isinstance(c,str) and len(c.strip())>15):continue
        rows.append({"text":c.strip(),"label":1});n_orig+=1
        for s in split_sents(c):
            rows.append({"text":s,"label":1});n_aug+=1

random.shuffle(rows)
with open(OUT,"w",encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

short=sum(1 for r in rows if len(r["text"])<80)
print(f"原始:{n_orig} 句子增强:{n_aug} 总计:{len(rows)}")
print(f"短文本(<80字):{short} ({short/len(rows)*100:.1f}%)")
print(f"保存到 {OUT}")
