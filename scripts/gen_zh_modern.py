# -*- coding: utf-8 -*-
"""GPU生成当代中文AI文本 - 用于提升中文检测器对新一代模型的识别"""
import json, random, os, time
import openvino_genai as ovg

OUT = r"D:\ai_train\data\ai_zh_modern.jsonl"
TARGET = 400
MODEL = r"D:\models\ov_qwen1.5b"

# 多种文体: 论述/问答/科普/分析
PROMPTS = [
 "请论述{t}的影响。","{t}是什么？请简要说明。","分析{t}的现状和趋势。",
 "谈谈你对{t}的看法。","总结{t}的主要特点。","{t}有哪些优势？",
 "为什么{t}很重要？","如何理解{t}？","请评价{t}。","{t}面临哪些挑战？",
]
TOPICS = ["人工智能对就业","远程办公","在线教育","新能源汽车","短视频平台",
"共享经济","元宇宙","区块链技术","云计算","大数据分析","5G通信","物联网",
"数字化转型","智能制造","智慧城市","医疗AI","自动驾驶","基因编辑","可再生能源",
"碳中和","网络隐私","算法推荐","社交媒体","电子支付","无人配送","虚拟现实",
"量子计算","开源软件","网络安全","数据治理","零工经济","人口老龄化","城乡差距",
"教育公平","心理健康","食品安全","公共卫生","垃圾分类","城市交通","住房政策"]

def gen(q):
    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = 220
    cfg.do_sample = True
    cfg.temperature = 0.92
    cfg.top_p = 0.95
    return str(pipe.generate(q, cfg)).strip()

print("加载GPU模型...", flush=True)
t0=time.time()
pipe = ovg.LLMPipeline(MODEL, "GPU")
print(f"加载 {time.time()-t0:.1f}s", flush=True)

qs=[]
for t in TOPICS:
    for p in PROMPTS:
        qs.append(p.format(t=t))
random.seed(11); random.shuffle(qs)

existing=set()
if os.path.exists(OUT):
    for line in open(OUT,encoding="utf-8"):
        try: existing.add(json.loads(line).get("question",""))
        except: pass
qs=[q for q in qs if q not in existing]
print(f"待生成 {len(qs)}, 目标 {TARGET}", flush=True)

done=len(existing); t0=time.time(); fail=0
with open(OUT,"a",encoding="utf-8") as f:
    for q in qs:
        if done>=TARGET: break
        try:
            a=gen(q)
            if len(a)>80:
                f.write(json.dumps({"text":a,"label":1,"source":"qwen2.5-gpu-zh","question":q},ensure_ascii=False)+"\n")
                f.flush(); done+=1
            if done%20==0:
                el=time.time()-t0
                print(f"[{time.strftime('%H:%M')}] {done}/{TARGET} ({(done-len(existing))/max(el,1)*60:.0f}/分)",flush=True)
        except Exception as e:
            fail+=1
            if fail%10==0: print(f"fail{fail}: {type(e).__name__}",flush=True)
            if fail>25: break
print(f"完成 {done} -> {OUT}", flush=True)
