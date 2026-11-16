# -*- coding: utf-8 -*-
"""
降AI自动迭代工具 (降AI矩阵第三档)
流程: 检测AI概率 -> 若>目标则LLM改写 -> 再检测 -> 循环直到达标
需要: llama-server运行(127.0.0.1:8089) + model_zoo检测器
用法:
  python humanize_auto.py "文本" --target 0.3 --max-loop 3
  python humanize_auto.py --batch in.txt out.txt
"""
import sys, os, json, urllib.request
sys.path.insert(0, r"D:\ai_train")
from model_zoo import detect as ai_detect

API = "http://127.0.0.1:8089/v1/chat/completions"
SYSTEM = ("你是中文写作润色助手。把用户给的AI文字改写成普通人自然说话的口语，"
          "保留全部原意，用短句、口语词、语气词，去掉书面套话。只输出改写后文字，不要解释。")

def server_ok():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=5) as r:
            return json.loads(r.read()).get("status")=="ok"
    except: return False

def rewrite(text, stronger=False):
    sysp = SYSTEM + (" 这次要改得更口语、更随意，像朋友聊天。" if stronger else "")
    data=json.dumps({"model":"local","messages":[{"role":"system","content":sysp},
        {"role":"user","content":text}],"max_tokens":400,"temperature":0.85}).encode()
    req=urllib.request.Request(API,data=data,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=180) as r:
        res=json.loads(r.read())
    out=res["choices"][0]["message"]["content"].strip()
    return out.split("改写：")[-1].split("---")[-1].strip()

def humanize_until(text, target=0.3, max_loop=3, tier="tier3"):
    history=[]
    cur=text
    p0=ai_detect(cur,tier=tier)["ai_probability"]
    history.append({"round":0,"prob":p0,"text":cur})
    for i in range(1,max_loop+1):
        if p0<=target: break
        cur=rewrite(cur,stronger=(i>=2))
        p=ai_detect(cur,tier=tier)["ai_probability"]
        history.append({"round":i,"prob":p,"text":cur})
        print(f"  第{i}轮改写: AI概率 {p0:.3f} -> {p:.3f}",file=sys.stderr)
        p0=p
    return {"initial_prob":history[0]["prob"],"final_prob":history[-1]["prob"],
            "target":target,"rounds":len(history)-1,"final_text":cur,"history":history}

if __name__=="__main__":
    if not server_ok():
        print("错误: llama-server未运行, 先双击start_llama_server.bat");sys.exit(1)
    args=sys.argv[1:]
    if not args:
        print("用法: python humanize_auto.py \"文本\" [--target 0.3] [--max-loop 3]");sys.exit(1)
    target=0.3; maxloop=3
    if "--target" in args: target=float(args[args.index("--target")+1])
    if "--max-loop" in args: maxloop=int(args[args.index("--max-loop")+1])
    text=args[0]
    print(f"原文: {text[:60]}...",file=sys.stderr)
    r=humanize_until(text,target,maxloop)
    print("\n===== 最终结果 =====")
    print(f"AI概率: {r['initial_prob']:.3f} -> {r['final_prob']:.3f} (目标<{target}, {r['rounds']}轮)")
    print(f"改写文本: {r['final_text']}")
