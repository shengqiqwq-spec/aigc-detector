# -*- coding: utf-8 -*-
"""生成中文 AI 文体数据 —— 专补实测发现的 6 个盲区

实测(现场手写 10 类 AI 文本喂给检测器):
  识别: 议论文 0.745 / 技术解析 0.807 / 新闻稿 0.611 / 说明书 0.736
  漏判: 产品介绍 0.113 / 学习方法 0.131 / 求职信 0.257
        学术摘要 0.167 / 影评 0.102 / 旅游攻略 0.131

这 6 类恰恰是 ChatGPT 最高频的输出类型。训练集里没有当代 AI 写的样本,
所以模型完全看不到它们的特征。

用法: python gen_zh_blindspots.py [每类目标条数]
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

# 14B 只有 6.5 tok/s(50s/条), 1.5B 有 46.7 tok/s(7s/条),
# 文风质量对"训练判别特征"这个用途已经够, 所以用 1.5B。
SERVER = "http://127.0.0.1:8092/completion"
OUT = os.path.join(os.environ.get("AIGC_DATA_DIR") or "data", "ai_zh_blindspots.jsonl")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 35

# 每个文体 6 个具体主题, 避免全是同一个话题
GENRES = {
    "产品介绍": [
        "一款降噪耳机", "一款扫地机器人", "一款电动牙刷", "一款空气炸锅",
        "一款智能门锁", "一款便携投影仪",
    ],
    "学习方法": [
        "如何背诵英语单词", "如何准备期末考试", "如何读完一本厚书",
        "如何整理课堂笔记", "如何提升阅读理解", "如何准备面试",
    ],
    "求职信": [
        "应聘前端开发工程师", "应聘市场专员", "应聘产品经理",
        "应聘财务助理", "应聘数据分析师", "应聘幼儿园教师",
    ],
    "学术摘要": [
        "社交媒体对青少年心理健康的影响", "城市绿地与居民幸福感的关系",
        "在线教育的学习效果评估", "新能源汽车的消费者接受度",
        "慢性病患者的服药依从性", "方言使用与身份认同",
    ],
    "影评": [
        "一部戛纳获奖剧情片", "一部国产科幻片", "一部经典黑白电影",
        "一部纪录片", "一部青春校园片", "一部悬疑犯罪片",
    ],
    "旅游攻略": [
        "去成都玩三天", "去西安看古迹", "去杭州周边自驾",
        "去厦门度周末", "去云南徒步", "去青岛看海",
    ],
}

# Qwen3 的 thinking 模式会输出推理链("好的，我现在需要…"), 必须用
# /no_think 关掉, 否则生成的全是思考过程而不是正文。
PROMPT = "写一段{genre}，主题是「{topic}」。200-350字，正式书面语。"


def call(prompt, n_predict=340, temp=0.9):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "stream": False,
        "temperature": temp, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.15,
        "stop": ["</s>", "<|im_end|>"],
    }).encode()
    req = urllib.request.Request(SERVER, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r).get("content", "")


def clean(t):
    t = t.strip()
    # 去掉 thinking 模式的残留: 思考内容 + 封闭标签
    t = re.sub(r"^.*?</think>", "", t, flags=re.S).strip()
    t = re.sub(r"</?think>", "", t).strip()
    # 去掉 markdown 加粗
    t = re.sub(r"\*\*", "", t)
    # 去掉字数尾巴
    t = re.sub(r"[（(]\s*约?\s*\d+\s*字\s*[）)]\s*$", "", t)
    # 去掉开头可能的"以下是/这是"等元话语
    t = re.sub(r"^(以下是|下面是|这是)[^。\n]{0,20}[：:]\s*", "", t)
    # 去掉首行标题(短且无句末标点)
    lines = t.split("\n")
    if lines and len(lines) > 1 and len(lines[0]) < 30 and not re.search(r"[。！？]", lines[0]):
        lines = lines[1:]
    t = "\n".join(lines).strip()
    # 压缩三连以上空行
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done[json.loads(line)["genre"]] = done.get(json.loads(line)["genre"], 0) + 1
                except Exception:
                    pass
        print(f"已有: {done}", flush=True)

    t0 = time.time()
    total = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for genre, topics in GENRES.items():
            have = done.get(genre, 0)
            need = TARGET - have
            if need <= 0:
                print(f"{genre}: 已够({have})", flush=True)
                continue
            got = 0
            tries = 0
            while got < need and tries < need * 4:
                tries += 1
                topic = random.choice(topics)
                try:
                    raw = call(PROMPT.format(genre=genre, topic=topic))
                except Exception as e:
                    print(f"  请求失败 {type(e).__name__}", flush=True)
                    time.sleep(3)
                    continue
                txt = clean(raw)
                # 质量门: 长度合理、主要是中文、没有指令复述
                if not (120 < len(txt) < 900):
                    continue
                cjk = sum(1 for c in txt if "\u4e00" <= c <= "\u9fff")
                if cjk / max(len(txt), 1) < 0.5:
                    continue
                low = txt.lower()
                if any(k in low for k in ("以下是", "要求：", "主题是", "字左右")):
                    continue
                f.write(json.dumps({"text": txt, "genre": genre, "topic": topic},
                                   ensure_ascii=False) + "\n")
                f.flush()
                got += 1
                total += 1
                el = time.time() - t0
                rate = total / el * 60 if el > 0 else 0
                print(f"  [{genre}] {got}/{need} ({len(txt)}字) {rate:.1f} 条/分", flush=True)
            print(f"{genre}: 完成 {got} 条 (试了 {tries} 次)", flush=True)

    print(f"\n共生成 {total} 条, 用时 {(time.time()-t0)/60:.1f} 分", flush=True)


if __name__ == "__main__":
    main()
