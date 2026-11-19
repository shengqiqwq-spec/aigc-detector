# -*- coding: utf-8 -*-
"""
降AI率改写器 v3 - 分语言/分文体, 保护结构

v2 针对 n-gram 特征做结构改写, 但有四个真 bug:
  1. 无语言判断 —— 英文文本开头被塞「个人感觉，」
  2. 插入点选在冒号/顿号后 —— 「各部门：为」变成「各部门：，为」
  3. 删除所有换行 —— 公文/邮件的段落结构被打烂
  4. 套话表太窄 —— 只覆盖 13 条, 大量书面连接词漏网

v3 修复以上四点, 套话表扩充到 40+ 条。
"""
import sys, re, random

def is_english(text):
    """英文判定: ASCII 字母占字母总数的绝大多数"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isascii()) / len(letters) > 0.85

# ---------- 中文 ----------
# 主观化前缀模板(人类写作核心特征: 有"我"的观点)
SUBJ_PREFIX = [
    "我觉得", "说实话", "在我看来", "讲真", "老实说", "要我说", "我倒是觉得",
]
# 语气/不确定表达(人类特征)
HEDGE = ["大概", "应该是", "感觉", "好像", "也许"]
# 口语连接词
CONNECT = ["然后", "不过", "还有就是", "话说回来", "对了", "反正", "其实"]

# 括号补充说明 (人类写作的强特征: 边写边补注)
ZH_PAREN = ["（这点其实挺关键）", "（说实话有点麻烦）", "（我们试过）", "（效果一般）",
            "（具体还得看情况）", "（这个不好说）", "（大概是这样）"]
EN_PAREN = [" (which is honestly a pain)", " (we tried this)",
            " (not sure it always works)", " (ymmv)",
            " (still working on it)"]

ZH_SLOGAN = [
    (r"首先[，,]?", ""), (r"其次[，,]?", ""), (r"最后[，,]?", "然后"),
    (r"总的来说[，,]?", "反正"), (r"综上所述[，,]?", "说白了"),
    (r"总而言之[，,]?", "总之"),
    (r"需要注意的是[，,]?", "对了"), (r"值得一提的是[，,]?", "顺带说"),
    (r"众所周知[，,]?", "大家都知道"),
    (r"值得注意的是[，,]?", "有趣的是"),
    (r"与此同时[，,]?", "同时"), (r"在此基础上[，,]?", "在这之上"),
    (r"由此可见[，,]?", "看得出来"),
    (r"因此[，,]", "所以"), (r"然而[，,]", "不过"),
    (r"此外[，,]", "还有"), (r"并且", "而且"),
    (r"从而", "这样就"), (r"进而", "接着"),
    (r"不仅", "不光"), (r"以及", "和"),
    (r"用于", "用来"),
    (r"是一种", "算是"), (r"通常", "一般"), (r"能够", "能"),
    (r"提供", "给"), (r"使用", "用"), (r"进行", "做"),
    (r"如果您", "如果你"), (r"建议您", "建议你"),
    (r"我们应该", "咱们得"), (r"我们必须", "我们得"),
    (r"具有重要意义", "挺重要的"), (r"起到了重要作用", "起了大作用"),
    (r"发挥着重要作用", "起着大作用"), (r"具有重要价值", "很有价值"),
    (r"深入探讨", "好好聊聊"), (r"进一步分析", "再往下看"),
    (r"全面了解", "搞清楚"), (r"充分认识", "明白"),
    (r"高度重视", "看重"), (r"切实履行", "真做到"),
    (r"认真组织", "好好安排"),
]

# ---------- 英文 ----------
EN_SUBJ = ["honestly", "i think", "in my experience", "to be fair",
           "personally", "from what i've seen"]
EN_HEDGE = ["probably", "i guess", "sort of", "maybe", "roughly"]
EN_CONNECT = ["though", "anyway", "also", "still", "plus", "but then"]

EN_SLOGAN = [
    (r"\bit is important to note that\b", "worth noting,"),
    (r"\bit should be noted that\b", "note that"),
    (r"\bin conclusion\b", "so"),
    (r"\bto summarize\b", "in short"),
    (r"\bfurthermore\b", "also"), (r"\bmoreover\b", "plus"),
    (r"\bhowever\b", "but"), (r"\btherefore\b", "so"),
    (r"\bconsequently\b", "which means"),
    (r"\badditionally\b", "also"), (r"\bin addition\b", "also"),
    (r"\bnevertheless\b", "still"), (r"\bnonetheless\b", "still"),
    (r"\bthus\b", "so"), (r"\bhence\b", "so"),
    (r"\butiliz(e|es|ed|ing)\b", "use"), (r"\bleverag(e|es|ed|ing)\b", "use"),
    (r"\bfacilitat(e|es|ed|ing)\b", "help"),
    (r"\bimplement(s|ed|ing)?\b", "do"),
    (r"\bnumerous\b", "many"), (r"\ba significant number of\b", "a lot of"),
    (r"\bit is crucial to\b", "you have to"),
    (r"\bplays? an? (important|crucial|vital|significant) role\b", "matters a lot"),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bwith regard to\b", "about"), (r"\bin terms of\b", "for"),
    (r"\bcomprehensive\b", "full"),
    # 合成一条, 避免 "transformative paradigm shift" 被替换成 "big big change"
    (r"\btransformative paradigm shift\b", "big change"),
    (r"\btransformative\b", "big"),
    (r"\bparadigm shift\b", "big change"),
]

def split_sentences(text):
    parts = re.split(r"([。！？；!?;])", text)
    sents = []
    cur = ""
    for p in parts:
        cur += p
        if p in "。！？；!?;":
            if cur.strip(): sents.append(cur)
            cur = ""
    if cur.strip(): sents.append(cur)
    return sents

def _safe_insert_pos(s, puncts):
    """找安全插入点: 标点之后, 且后面不接英文词/数字, 也不连续堆标点"""
    safe = []
    for m in re.finditer(f"[{puncts}]", s):
        p = m.start()
        nxt = s[p + 1] if p + 1 < len(s) else ""
        if nxt and nxt.isascii() and nxt.isalnum():
            continue          # 不劈开英文单词/缩写
        if nxt in "，、；：,:;":
            continue          # 不连续堆两个标点
        safe.append(p)
    return safe

def humanize_v2(text, strength=1.0, lang=None):
    """改写入口 (参数名与 v2 兼容)

    lang: "zh" / "en" / None(自动判断)
    """
    if lang is None:
        lang = "en" if is_english(text) else "zh"
    if lang == "en":
        slogans, subjects, hedges, connects = EN_SLOGAN, EN_SUBJ, EN_HEDGE, EN_CONNECT
        parens = EN_PAREN
        puncts, sep = ".,;:", " "
    else:
        slogans, subjects, hedges, connects = ZH_SLOGAN, SUBJ_PREFIX, HEDGE, CONNECT
        parens = ZH_PAREN
        puncts, sep = "，、；：", "，"

    # 保留原始段落结构
    out_paras = []
    for para in text.split("\n"):
        if not para.strip():
            out_paras.append("")
            continue
        stripped = para.strip()
        # 列表行/短标题行原样保留 (破坏它们是格式 bug)
        if (re.match(r"^(\d+[\.\)]|[-*•]|[一二三四五六七八九十]+[、.])", stripped)
                or len(stripped) < 12):
            out_paras.append(para)
            continue

        sents = split_sentences(para)
        injected = 0
        # 注入上限随强度变化: strength=1 时每 3 句 1 处, 2 时每句都允许
        per = max(1, int(3 / max(strength, 0.5)))
        max_inject = max(1, len(sents) // per) if strength <= 1 else max(1, int(len(sents) * (strength / 3)))
        new_sents = []

        for i, s in enumerate(sents):
            # 1) 套话替换
            for pat, rep in slogans:
                s = re.sub(pat, rep, s) if lang == "zh" else re.sub(pat, rep, s, flags=re.I)

            # 2) 注入口语标记, 每句最多一处 (叠加会变结巴, 必须互斥)
            if strength >= 2 and len(s) > 40:
                # 长句拆短: 在中间逗号处断句(人类特征)
                mids = [m.start() for m in re.finditer(r"[，；,;]", s)]
                if mids and random.random() < 0.35 * (strength - 1):
                    cut = mids[len(mids) // 2]
                    s = s[:cut] + ("。" if lang == "zh" else ". ") + s[cut + 1:].lstrip()
            if strength > 0 and len(s) > 14 and injected < max_inject:
                roll = random.random()
                # 括号补充: 实测降幅最大(0.804 -> 0.244), 优先
                if roll < min(0.75, 0.35 * strength) and len(s) > 20:
                    body = s.rstrip("。！？；!?;.")
                    end = s[len(body):] or ("。" if lang == "zh" else ".")
                    s = body + random.choice(parens) + end
                    injected += 1
                elif roll < min(0.9, 0.22 * strength) and (i % 3 == 0 or strength > 1):
                    if lang == "zh":
                        s = random.choice(subjects) + sep + s
                    else:
                        # 英文: "Honestly, Artificial ..."
                        s = random.choice(subjects) + ", " + s
                        s = s[0].upper() + s[1:]
                    injected += 1
                elif roll < min(0.9, 0.38 * strength) and i > 0:
                    if lang == "zh":
                        s = random.choice(connects) + sep + s
                    else:
                        s = random.choice(connects) + ", " + s
                        s = s[0].upper() + s[1:]
                    injected += 1
                elif roll < min(0.95, 0.55 * strength):
                    safe = _safe_insert_pos(s, puncts)
                    if safe:
                        pos = random.choice(safe)
                        nxt = s[pos + 1:]
                        if lang == "zh":
                            s = s[:pos + 1] + random.choice(hedges) + sep + nxt.lstrip()
                        else:
                            s = s[:pos + 1] + " " + random.choice(hedges) + sep + nxt.lstrip()
                        injected += 1
            new_sents.append(s)

        out_paras.append("".join(new_sents))

    result = "\n".join(out_paras).strip()
    if lang == "en":
        # 标点后补空格 + 压缩多余空格
        result = re.sub(r"([.!?,;:])(?=[A-Za-z])", r"\1 ", result)
        result = re.sub(r"[ \t]{2,}", " ", result)
        # 句末标点后首字母大写 (英文语法要求)
        result = re.sub(r"(^|[.!?]\s+)([a-z])",
                        lambda m: m.group(1) + m.group(2).upper(), result)
        result = result.lstrip()
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python humanize2.py \"文本\""); sys.exit(1)
    print(humanize_v2(sys.argv[1]))
