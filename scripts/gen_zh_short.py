# -*- coding: utf-8 -*-
import os as _os
# 可移植路径: 默认用仓库下的 data/ 和 ckpt/, 可用环境变量覆盖
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get("AIGC_DATA_DIR", _os.path.join(_ROOT, "data"))
CKPT_DIR = _os.environ.get("AIGC_CKPT_DIR", _os.path.join(_ROOT, "ckpt"))
"""专攻短文本AI样本 (30-150字), 补齐长度分布短板"""
import sys, os, json, time, random, re
import openvino_genai as ovg

OUT = DATA_DIR + r"\ai_zh_short2.jsonl"
TARGET = int(sys.argv[1]) if len(sys.argv)>1 else 500
MODEL = os.environ.get("AIGC_GEN_MODEL", "models/ov_qwen1.5b")

# 短文本场景: 问答回复、评论、说明、建议
PROMPTS = [
 "用一两句话回答：什么是{topic}？",
 "简短说明一下{topic}，控制在50字以内。",
 "关于{topic}，简单说说你的看法，50字左右。",
 "一句话概括{topic}。",
 "用三四十个字解释{topic}。",
 "{topic}是什么意思？简短回答。",
 "简短评论一下{topic}，不超过80字。",
 "用100字左右介绍一下{topic}。",
 "说说{topic}的三个要点，每个一句。",
 "对{topic}给个简短建议，50字内。",
 "如果朋友问你{topic}，你会怎么简单回答？",
 "简要说明{topic}的作用，30字左右。",
 "用口语化的方式解释{topic}，60字左右。",
 "简单谈谈{topic}的优缺点，80字内。",
 "一句话说说{topic}为什么重要。",
]

TOPICS = ["区块链","元宇宙","碳中和","预制菜","直播带货","短视频","人工智能","大模型",
 "新能源汽车","充电桩","共享单车","外卖平台","网购退货","在线问诊","电子发票",
 "云存储","密码管理","双因素认证","开源软件","版本控制","单元测试","代码审查",
 "敏捷开发","微服务","容器化","负载均衡","缓存策略","数据库索引","消息队列","分布式锁",
 "机器学习","深度学习","神经网络","过拟合","正则化","交叉验证","特征工程","模型压缩",
 "时间管理","番茄工作法","深度工作","刻意练习","费曼学习法","复盘","目标管理","精力管理",
 "断舍离","极简主义","情绪管理","正念冥想","睡眠质量","有氧运动","力量训练","间歇性断食"]

random.seed()
tasks=[]
for t in TOPICS:
    for p in PROMPTS: tasks.append((t,p))
random.shuffle(tasks)

def clean(t):
    t=re.sub(r"^```.*?\n","",t); t=re.sub(r"```","",t)
    t=re.sub(r"^(好的|当然|没问题)[，,。\s]*","",t)
    t=re.sub(r"^(以下|下面)(是|为)[^：:]{0,30}[：:]\s*","",t)
    t=re.sub(r"^[。，、\s\*]+","",t)
    t=re.sub(r"^\*\*?[^\n]{0,30}\*\*?\s*\n","",t)
    t=re.sub(r"^\s*【[^】]{0,30}】\s*\n?","",t)
    t=re.sub(r"^\s*\d+[\.、]\s*","",t)
    return t.strip()

def bad(t):
    if not t or len(t)<25 or len(t)>400: return True
    for k in range(4,min(len(t),60)):
        if t.count(t[:k])>3: return True
    seg=t.split("，")
    for i in range(len(seg)-2):
        if len(set(seg[i:i+2]))==1: return True
    for m in ["好的，以下","以下是","希望对你有帮助","这有点难","你可能想知道","作为AI","我是一个语言模型"]:
        if m in t[:80]: return True
    cjk=sum(1 for c in t if '\u4e00'<=c<='\u9fff')
    return cjk/max(len(t),1) < 0.5

print("加载模型...", flush=True)
pipe = ovg.LLMPipeline(MODEL, "GPU")
cfg = pipe.get_generation_config()
cfg.max_new_tokens = 220
cfg.temperature = 0.9
cfg.top_p = 0.92
cfg.repetition_penalty = 1.08
cfg.do_sample = True
print("就绪, 目标", TARGET, flush=True)

start=time.time(); ok=0; tried=0
with open(OUT,"a",encoding="utf-8") as f:
    for topic,prompt in tasks:
        if ok>=TARGET: break
        tried+=1
        try: raw=str(pipe.generate(prompt.format(topic=topic), cfg))
        except Exception: continue
        txt=clean(raw)
        if bad(txt): continue
        f.write(json.dumps({"text":txt,"topic":topic,"model":"Qwen2.5-1.5B-OV-GPU","style":"short"},ensure_ascii=False)+"\n")
        f.flush(); ok+=1
        if ok%25==0:
            el=time.time()-start
            print(f"[{ok}/{TARGET}] {el/60:.1f}分 {ok/el*60:.1f}条/分 通过{ok/max(tried,1)*100:.0f}%", flush=True)
print(f"完成{ok}条 {(time.time()-start)/60:.1f}分", flush=True)
