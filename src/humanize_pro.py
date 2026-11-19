# -*- coding: utf-8 -*-
"""
降AI工具 (LLM 版) - 检测→改写→再检测 自动迭代

需要本地 llama-server。效果与模型大小强相关 (实测):
  - Qwen3-14B Q4:  单轮可把 0.89 降到 0.18, 但本机 ~0.7 tok/s, 一段话要 50s+
  - Qwen2.5-1.5B:  1.4s 出结果, 但改写力度不足, 有时反而升分

因此: 没有大模型时请用 humanize2.py (规则改写) 或 evasion_engine.py
(程序化引擎), 两者零依赖且已修好。本脚本适合有 GPU/大显存的机器。

后端端口按顺序探测: 8089 → 8090 → 8091

用法:
  python src/humanize_pro.py "文本"
  python src/humanize_pro.py "文本" --target 0.3
  python src/humanize_pro.py --batch in.txt out.txt
"""
import sys, os, json, urllib.request
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import model_zoo as M

# 支持多个 LLM 后端, 按顺序探测第一个可用的
BACKENDS = [
    ("http://127.0.0.1:8089", "llama-server #1"),
    ("http://127.0.0.1:8090", "llama-server #2"),
    ("http://127.0.0.1:8091", "llama-server #3"),
]
_API_BASE = None
API = None

SYS_BASE = ("把下面的AI生成文字改写成普通人自然说话的口语，保留全部原意，"
            "用短句、口语词、语气词，去掉书面套话和结构化表达。只输出改写后的文字，不要解释。")
SYS_STRONG = ("把下面的文字彻底改写成朋友聊天的口语，越自然越好。保留核心信息，"
              "但句式要随意、有语气。只输出改写后文字。")

def detect_prob(text):
    return M._ALL_PREDICTORS["tier1"](text)

def rewrite(text, strong=False):
    if not find_backend():
        raise RuntimeError("没有可用的 LLM 后端 (需要 llama-server)")
    data = json.dumps({"model":"local","messages":[
        {"role":"system","content": SYS_STRONG if strong else SYS_BASE},
        {"role":"user","content": text}],
        "max_tokens":800,"temperature":0.88}).encode()
    req = urllib.request.Request(API, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    return out.split("改写：")[-1].split("---")[-1].strip()

def find_backend():
    """探测可用的 LLM 后端, 返回 (base_url, name) 或 None"""
    global _API_BASE, API
    if _API_BASE:
        return (_API_BASE, "cached")
    for base, name in BACKENDS:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    _API_BASE = base
                    API = base + "/v1/chat/completions"
                    return (base, name)
        except Exception:
            continue
    return None

def server_ok():
    return find_backend() is not None

def humanize(text, target=0.3, max_rounds=3):
    history = [{"round":0, "prob":detect_prob(text), "text":text}]
    cur = text
    for i in range(1, max_rounds+1):
        cur_p = history[-1]["prob"]
        if cur_p <= target:
            break
        cur = rewrite(cur, strong=(i>=2))
        p = detect_prob(cur)
        history.append({"round":i, "prob":p, "text":cur})
        print(f"  第{i}轮: {cur_p:.3f} -> {p:.3f}", file=sys.stderr)
    return {
        "initial_prob": round(history[0]["prob"],4),
        "final_prob": round(history[-1]["prob"],4),
        "rounds": len(history)-1,
        "final_text": cur,
        "reached_target": history[-1]["prob"] <= target,
    }

if __name__ == "__main__":
    backend = find_backend()
    if not backend:
        print("错误: 没有可用的 LLM 后端")
        print("需要先启动 llama-server, 例如:")
        print("  llama-server -m <模型.gguf> --port 8089 -ngl 99")
        print("支持的端口: 8089 / 8090 / 8091")
        sys.exit(1)
    print(f"使用后端: {backend[1]} ({backend[0]})", file=sys.stderr)
    args = sys.argv[1:]
    if not args:
        print("用法: python humanize_pro.py \"文本\" [--target 0.3]"); sys.exit(1)
    target = 0.3
    if "--target" in args:
        target = float(args[args.index("--target")+1])
    if args[0] == "--batch" and len(args)>=3:
        lines=[l.strip() for l in open(args[1],encoding="utf-8") if l.strip()]
        out=[]
        for i,l in enumerate(lines,1):
            print(f"处理 {i}/{len(lines)}", file=sys.stderr)
            out.append(humanize(l,target)["final_text"])
        open(args[2],"w",encoding="utf-8").write("\n".join(out))
        print(f"完成 -> {args[2]}")
    else:
        r = humanize(args[0], target)
        print(json.dumps({k:v for k,v in r.items() if k!="final_text"}, ensure_ascii=False))
        print("\n改写结果:")
        print(r["final_text"])
