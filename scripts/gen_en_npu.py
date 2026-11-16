# -*- coding: utf-8 -*-
"""英文AI数据生成 - NPU版(Qwen2.5-1.5B int4)"""
import json, random, os, time
import openvino_genai as ovg

OUT = r"D:\ai_train\data\ai_en_npu.jsonl"
TARGET = 300
MODEL = r"D:\models\ov_qwen1.5b"

TOPICS = ["climate change mitigation","AI in healthcare","remote work","online learning",
"space exploration","social media impact","electric vehicles","universal basic income",
"cryptocurrency regulation","mental health","sustainable agriculture","gene therapy",
"quantum computing","cybersecurity","smart cities","food delivery","gaming addiction",
"data privacy","job automation","AI tutoring","self-driving cars","virtual reality",
"blockchain","telemedicine","biometric security","open source software","cloud computing",
"machine translation","recommendation systems","NLP applications","computer vision",
"warehouse robotics","drone regulations","IoT security","5G deployment","edge computing",
"serverless architecture","container orchestration","API design","database tuning",
"caching strategies","load balancing","microservices","devops culture","continuous integration",
"test automation","code review","technical debt","software estimation","pair programming"]

PREFIXES = ["Explain the main benefits of","Discuss the future of","What should people know about",
    "Analyze the impact of","Give your perspective on","Summarize the debate around",
    "Outline the key aspects of","Describe the importance of"]

print("加载NPU模型...", flush=True)
t0=time.time()
pipe = ovg.LLMPipeline(MODEL, "GPU")
print(f"加载完成 {time.time()-t0:.1f}s", flush=True)

existing=set()
if os.path.exists(OUT):
    for line in open(OUT,encoding="utf-8"):
        try: existing.add(json.loads(line).get("question",""))
        except: pass
print(f"已有 {len(existing)}", flush=True)

qs=[]
for t in TOPICS:
    for p in PREFIXES:
        qs.append(f"{p} {t}?")
random.seed(7); random.shuffle(qs)
qs=[q for q in qs if q not in existing]

cfg = ovg.GenerationConfig()
cfg.max_new_tokens = 110
cfg.do_sample = True
cfg.temperature = 0.9

done=len(existing); t0=time.time(); fail=0
with open(OUT,"a",encoding="utf-8") as f:
    for q in qs:
        if done>=TARGET: break
        try:
            res = pipe.generate(q, cfg)
            a = str(res).strip()
            if len(a)>60:
                f.write(json.dumps({"text":a,"label":1,"source":"qwen2.5-npu","question":q},ensure_ascii=False)+"\n")
                f.flush(); done+=1
            if done%20==0:
                el=time.time()-t0
                print(f"[{time.strftime('%H:%M')}] {done}/{TARGET} ({(done-len(existing))/max(el,1)*60:.0f}/分)",flush=True)
        except Exception as e:
            fail+=1
            if fail%10==0: print(f"fail{fail}: {type(e).__name__}",flush=True)
            if fail>30: break
print(f"完成 {done} -> {OUT}", flush=True)
