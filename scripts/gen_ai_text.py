# -*- coding: utf-8 -*-
import os as _os
# 可移植路径: 默认用仓库下的 data/ 和 ckpt/, 可用环境变量覆盖
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get("AIGC_DATA_DIR", _os.path.join(_ROOT, "data"))
CKPT_DIR = _os.environ.get("AIGC_CKPT_DIR", _os.path.join(_ROOT, "ckpt"))
"""OpenVINO 1.5B GPU 大规模生成 中/英文当代AI文本"""
import sys, os, json, time, random, re
import openvino_genai as ovg

LANG = sys.argv[1]           # zh / en
TARGET = int(sys.argv[2]) if len(sys.argv)>2 else 300
OUT = os.path.join(DATA_DIR, f"ai_{LANG}_ov.jsonl")
MODEL = os.environ.get("AIGC_GEN_MODEL", "models/ov_qwen1.5b")

TOPICS_ZH = ["人工智能与就业","在线教育","城市交通治理","短视频对青少年","新能源汽车",
 "心理健康","远程医疗","食品安全","气候变暖","老龄化社会","乡村振兴","数字经济",
 "传统文化传承","网络隐私","职场压力","亲子关系","旅游产业","体育竞技","艺术创作",
 "太空探索","基因编辑","元宇宙","直播电商","共享经济","社交媒体","电子竞技",
 "养老产业","环保政策","科研评价","语言学习","城市规划","公共卫生","消费降级",
 "创业环境","人工智能伦理","数据安全","自动驾驶","区块链","量子计算","生物医药",
 "教育改革","医疗保障","住房问题","就业形势","人工智能监管","数字鸿沟","青少年教育"]

TOPICS_EN = ["artificial intelligence and jobs","online education","urban traffic","social media and teens",
 "electric vehicles","mental health","telemedicine","food safety","climate change","aging population",
 "digital economy","online privacy","workplace stress","tourism industry","space exploration",
 "gene editing","live commerce","sharing economy","renewable energy","cybersecurity",
 "remote work","sustainable farming","public health","language learning","smart cities",
 "startup culture","AI ethics","data protection","autonomous driving","quantum computing",
 "biotechnology","education reform","healthcare access","housing crisis","digital divide",
 "youth education","media literacy","cultural heritage","urban planning","food delivery"]

STYLES_ZH = [
 "请用流畅自然的中文写一段论述，约200字，围绕{topic}。",
 "以{topic}为主题，写一段分析性文字，约200字，语气客观。",
 "就{topic}这个话题，写一段有观点的评论，约200字。",
 "写一篇关于{topic}的短文，约200字，举例说明。",
 "结合现实，谈谈{topic}，约200字。",
 "{topic}——请写一段说明性文字，约200字。",
 "关于{topic}，写一段分析，约200字，可以提到挑战和机遇。",
 "谈谈你对{topic}的看法，约200字。",
]
STYLES_EN = [
 "Write a clear paragraph of about 150 words discussing {topic}.",
 "Write an analytical paragraph of about 150 words about {topic}.",
 "Write a short essay paragraph of about 150 words on {topic}.",
 "Discuss {topic} in about 150 words, mentioning benefits and challenges.",
 "Write an informative paragraph about {topic}, around 150 words.",
 "Give your view on {topic} in about 150 words.",
 "Write a paragraph explaining {topic} to a general audience, about 150 words.",
 "Compose a short piece about {topic}, around 150 words.",
]

def clean_zh(t):
    t = re.sub(r"^```.*?\n","",t); t = re.sub(r"```","",t)
    t = re.sub(r"^(好的|当然|没问题)[，,。\s]*","",t)
    t = re.sub(r"^(以下|下面)(是|为)[^：:]{0,30}[：:]\s*","",t)
    t = re.sub(r"^[。，、\s]+","",t)
    t = re.sub(r"^\*\*?[^\n]{0,40}\*\*?\s*\n","",t)
    t = re.sub(r"^\s*【[^】]{0,30}】\s*\n?","",t)
    # 《标题》单独一行
    t = re.sub(r"^\s*《[^》]{2,40}》\s*\n+","",t)
    t = re.sub(r"^\s*《[^》]{2,40}》\s*","",t)
    # 冒号标题行 "标题：xxx\n"
    t = re.sub(r"^[^\n：:]{2,20}[：:]\s*[^\n]{2,50}\n+","",t)
    t = re.sub(r"^\s*#+\s*[^\n]{0,40}\n","",t)
    return t.strip()

def clean_en(t):
    t = re.sub(r"^```.*?\n","",t); t = re.sub(r"```","",t)
    t = re.sub(r"^(Certainly|Sure|Of course|Absolutely)[!,.]\s*","",t,flags=re.I)
    t = re.sub(r"^Here (is|are)[^:]{0,30}:\s*","",t,flags=re.I)
    t = re.sub(r"^(As an AI|I'm an AI|As a language model)[^.]*\.\s*","",t,flags=re.I)
    return t.strip()

def bad_zh(t):
    if not t or len(t)<120 or len(t)>1500: return True
    for k in range(4,min(len(t),80)):
        if t.count(t[:k])>4: return True
    seg=t.split("，")
    for i in range(len(seg)-3):
        if len(set(seg[i:i+3]))==1: return True
    for m in ["好的，以下","以下是","希望对你有帮助","这有点难","你可能想知道","作为AI","我是一个语言模型"]:
        if m in t[:100]: return True
    cjk=sum(1 for c in t if '\u4e00'<=c<='\u9fff')
    return cjk/len(t) < 0.5

def bad_en(t):
    if not t or len(t)<400 or len(t)>3000: return True
    for k in range(5,min(len(t),100)):
        if t.count(t[:k])>4: return True
    asc=sum(1 for c in t if ord(c)<128)
    if asc/len(t)<0.9: return True
    for m in ["I cannot","I'm sorry","As an AI"]:
        if m in t[:100]: return True
    return False

topics = TOPICS_ZH if LANG=="zh" else TOPICS_EN
styles = STYLES_ZH if LANG=="zh" else STYLES_EN
clean = clean_zh if LANG=="zh" else clean_en
isbad = bad_zh if LANG=="zh" else bad_en

tasks=[]
for t in topics:
    for s in styles: tasks.append((t,s))
random.seed(); random.shuffle(tasks)

print("加载模型 (GPU)...", flush=True)
pipe = ovg.LLMPipeline(MODEL, "GPU")
cfg = pipe.get_generation_config()
cfg.max_new_tokens = 320
cfg.temperature = 0.85
cfg.top_p = 0.92
cfg.repetition_penalty = 1.12
cfg.do_sample = True
print("就绪, 目标", TARGET, flush=True)

start=time.time(); ok=0; tried=0
with open(OUT,"a",encoding="utf-8") as f:
    for topic,style in tasks:
        if ok>=TARGET: break
        tried+=1
        try:
            raw = pipe.generate(style.format(topic=topic), cfg)
        except Exception:
            continue
        txt = clean(str(raw))
        if isbad(txt): continue
        f.write(json.dumps({"text":txt,"topic":topic,"model":"Qwen2.5-1.5B-OV-GPU"},ensure_ascii=False)+"\n")
        f.flush(); ok+=1
        if ok%25==0:
            el=time.time()-start
            print(f"[{ok}/{TARGET}] {el/60:.1f}分 {ok/el*60:.1f}条/分 通过率{ok/max(tried,1)*100:.0f}%", flush=True)
print(f"完成{ok}条 用{(time.time()-start)/60:.1f}分 -> {OUT}", flush=True)
