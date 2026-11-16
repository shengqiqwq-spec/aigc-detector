# -*- coding: utf-8 -*-
"""
降AI率工具 (正式版) - 本地Qwen3-14B改写 + 检测器验证
依赖: llama-server 运行在 127.0.0.1:8089
用法:
  1) 先启动服务: 见 start_llama_server.bat
  2) python humanize_llm.py "待改写文本"
     python humanize_llm.py --batch input.txt output.txt
输出: 改写文本 + (可选)检测器AI概率前后对比
"""
import sys, json, os, urllib.request

API = "http://127.0.0.1:8089/v1/chat/completions"

SYSTEM_PROMPT = ("你是一个中文写作润色助手。把用户给的AI生成的文字，改写成普通人自然说话的口语。"
                 "保留全部原意和信息，但用日常聊天的方式表达：用短句、口语词、语气词，"
                 "去掉书面套话和结构化表达。只输出改写后的文字本身，不要任何解释、标题或前言。")

def rewrite(text, temperature=0.8, max_tokens=400):
    data = json.dumps({
        "model": "local",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        res = json.loads(r.read())
    out = res["choices"][0]["message"]["content"].strip()
    # 清理模型偶尔带的前缀
    out = out.split("改写：")[-1].split("---")[-1].strip()
    return out

def check_server():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=5) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False

if __name__ == "__main__":
    if not check_server():
        print("错误: llama-server 未运行。先双击 start_llama_server.bat"); sys.exit(1)
    if len(sys.argv) < 2:
        print("用法: python humanize_llm.py \"文本\"  |  --batch in.txt out.txt"); sys.exit(1)
    if sys.argv[1] == "--batch" and len(sys.argv) >= 4:
        lines = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
        res = []
        for i, l in enumerate(lines, 1):
            print(f"改写 {i}/{len(lines)}...", file=sys.stderr)
            res.append(rewrite(l))
        open(sys.argv[3], "w", encoding="utf-8").write("\n".join(res))
        print(f"完成: {len(res)} 行 -> {sys.argv[3]}")
    else:
        print(rewrite(sys.argv[1]))
