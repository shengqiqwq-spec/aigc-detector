# -*- coding: utf-8 -*-
"""
英文HC3数据清洗: 去除预处理格式泄漏
问题: HC3英文中ChatGPT答案被加了 " ." " ," 空格(空格+标点), 人类答案没有,
      模型学到这个格式差异 → F1=1.0假象
修复: 标准化空格, 统一标点前无空格
"""
import json, re

SRC = r"D:\ai_train\data\hc3_en_real.jsonl"
OUT = r"D:\ai_train\data\hc3_en_clean.jsonl"

def clean(t):
    if not isinstance(t, str): return None
    # 标点前空格去除: " ." -> ".", " ," -> ",", " ?" -> "?", " !" -> "!"
    t = re.sub(r"\s+([.,?!;:])", r"\1", t)
    # 括号内去多余空格
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    # 多个空格合一
    t = re.sub(r" {2,}", " ", t)
    # 引号统一
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return t.strip()

n=0; kept=0
with open(OUT,"w",encoding="utf-8") as fo:
    for line in open(SRC,encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        n+=1
        h=[clean(x) for x in d.get("human_answers",[])]
        c=[clean(x) for x in d.get("chatgpt_answers",[])]
        h=[x for x in h if x and len(x)>20]
        c=[x for x in c if x and len(x)>20]
        if h and c:
            kept+=1
            fo.write(json.dumps({"question":d.get("question",""),"human_answers":h,"chatgpt_answers":c},ensure_ascii=False)+"\n")
print(f"处理{n}行, 保留{kept}行 -> {OUT}")
