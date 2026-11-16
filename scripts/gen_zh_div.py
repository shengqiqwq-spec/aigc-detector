# -*- coding: utf-8 -*-
"""多样化中文AI生成 - 更丰富的问题模板"""
import json, random, os, time
import openvino_genai as ovg

OUT = r"D:\ai_train\data\ai_zh_div.jsonl"
MODEL = r"D:\models\ov_qwen1.5b"
TARGET = 600

# 更多样的问法
TEMPLATES = [
 "请写一段关于{t}的短文。","用三句话解释{t}。","{t}的优缺点分别是什么？",
 "如果有人问你{t}，你会怎么回答？","写一段介绍{t}的文字。",
 "{t}对我们的日常生活有什么影响？","未来{t}会怎么发展？",
 "用简单的例子说明{t}。","{t}该怎么入门？","评价一下{t}的现状。",
 "写一篇关于{t}的科普。","{t}存在哪些误区？","如何向小学生解释{t}？",
 "{t}的关键技术有哪些？","{t}为什么受到关注？","{t}会带来哪些改变？",
]
TOPICS = ["人工智能","机器学习","深度学习","神经网络","自然语言处理","计算机视觉",
"语音识别","推荐系统","智能客服","自动驾驶","无人机","机器人","3D打印","虚拟现实",
"增强现实","区块链","比特币","数字货币","智能合约","云原生","微服务","容器技术",
"边缘计算","量子计算","生物识别","人脸识别","数据挖掘","数据可视化","商业智能",
"数字化转型","智能制造","工业互联网","数字孪生","智慧农业","冷链物流","跨境电商",
"社区团购","直播电商","私域流量","用户增长","产品设计","敏捷开发","代码审查"]

def gen(q):
    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = 200
    cfg.do_sample = True
    cfg.temperature = 0.95
    cfg.top_p = 0.92
    return str(pipe.generate(q, cfg)).strip()

print("加载GPU...", flush=True)
pipe = ovg.LLMPipeline(MODEL, "GPU")

qs=[]
for t in TOPICS:
    for tmpl in TEMPLATES:
        qs.append(tmpl.format(t=t))
random.seed(31415); random.shuffle(qs)

existing=set()
if os.path.exists(OUT):
    for line in open(OUT,encoding="utf-8"):
        try: existing.add(json.loads(line).get("question",""))
        except: pass
qs=[q for q in qs if q not in existing]
print(f"可用问题 {len(qs)}", flush=True)

done=len(existing); t0=time.time()
with open(OUT,"a",encoding="utf-8") as f:
    for q in qs:
        if done>=TARGET: break
        try:
            a=gen(q)
            if len(a)>80:
                f.write(json.dumps({"text":a,"label":1,"source":"qwen-div","question":q},ensure_ascii=False)+"\n")
                f.flush(); done+=1
            if done%30==0:
                print(f"[{time.strftime('%H:%M')}] {done}/{TARGET}",flush=True)
        except Exception:
            pass
print(f"完成 {done}", flush=True)
