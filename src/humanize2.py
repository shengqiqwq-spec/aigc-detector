# -*- coding: utf-8 -*-
"""
降AI率改写器 v2 - 强口语化
v1只换套话效果弱(9.7%); v2针对检测器n-gram特征, 结构性改写:
  - 长句切短 + 口语连接
  - 注入主观判断/反问/语气词(人类的核心特征)
  - 打散书面语结构
"""
import sys, re, random

# 主观化前缀模板(人类写作核心特征: 有"我"的观点)
SUBJ_PREFIX = [
    "我觉得", "说实话", "在我看来", "怎么说呢", "讲真", "个人感觉",
    "我记得", "我之前也碰到过", "老实说", "要我说",
]
# 语气/不确定表达(人类特征)
HEDGE = ["大概", "可能吧", "说实话我也不太确定", "应该是", "感觉", "好像"]
# 口语连接词
CONNECT = ["然后", "不过", "还有就是", "话说回来", "对了", "反正"]

def split_sentences(text):
    text = re.sub(r"\n+", " ", text)
    parts = re.split(r"([。！？；])", text)
    sents = []
    cur = ""
    for p in parts:
        cur += p
        if p in "。！？；":
            if cur.strip(): sents.append(cur.strip())
            cur = ""
    if cur.strip(): sents.append(cur.strip())
    return sents

def humanize_v2(text, strength=1.0):
    sents = split_sentences(text)
    out = []
    for i, s in enumerate(sents):
        if not s: continue
        # 套话替换(保留v1核心)
        for pat, rep in [
            (r"首先[，,]?", ""), (r"其次[，,]?", ""), (r"最后[，,]?", "然后"),
            (r"总的来说[，,]?", "反正"), (r"综上所述[，,]?", "说白了"),
            (r"需要注意的是[，,]?", "对了"), (r"众所周知[，,]?", "大家都知道"),
            (r"因此[，,]", "所以"), (r"然而[，,]", "不过"), (r"此外[，,]", "还有"),
            (r"并且", "而且"), (r"用于", "用来"), (r"包括", "像"),
            (r"是一种", "算是"), (r"通常", "一般"), (r"能够", "能"),
            (r"提供", "给"), (r"使用", "用"), (r"如果您", "如果你"),
            (r"建议您", "建议你"), (r"我们应该", "咱们得"),
            (r"具有重要意义", "挺重要的"),
        ]:
            s = re.sub(pat, rep, s)
        # 注入主观前缀(每隔几句)
        if strength > 0 and i % 3 == 0 and len(s) > 15:
            s = random.choice(SUBJ_PREFIX) + "，" + s[0].lower() + s[1:] if s[0].isupper() else random.choice(SUBJ_PREFIX) + "，" + s
        # 注入口语连接/语气
        if i > 0 and random.random() < 0.5 * strength:
            s = random.choice(CONNECT) + "，" + s
        if random.random() < 0.3 * strength and len(s) > 20:
            pos = random.randint(5, len(s)//2)
            s = s[:pos] + random.choice(HEDGE) + s[pos:]
        out.append(s)
    result = "".join(out)
    # 粘连段落
    result = result.replace("\n", "")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python humanize2.py \"文本\""); sys.exit(1)
    print(humanize_v2(sys.argv[1]))
