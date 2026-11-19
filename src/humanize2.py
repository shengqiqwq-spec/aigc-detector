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

# 正式文书特征词: 命中足够多就走"正式"模式, 不做口语化替换
FORMAL_MARKERS = ("各部门", "各单位", "现将", "有关事项", "通知如下", "经研究决定",
                  "特此通知", "请遵照执行", "贯彻落实", "高度重视", "认真组织",
                  "切实做好", "进一步加强", "现将有关事项", "现予公布", "此致",
                  "敬礼", "请示", "批复", "函", "纪要", "决议", "章程")


def is_formal(text):
    """判断是否为公文/正式文书

    正式文书里"高度重视"改成"看重"、"认真组织"改成"好好安排",
    读起来像小学生模仿公文 —— 语域崩塌比留着 AI 味更糟。
    """
    hits = sum(1 for m in FORMAL_MARKERS if m in text)
    # 短文本里 3 个正式词就够判; 长文本按密度
    if len(text) < 120:
        return hits >= 3
    return hits >= 4 or (hits >= 2 and "通知" in text)


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
# 正式文书的括注: 不能带口语腔, 否则公文变闲聊
ZH_PAREN_FORMAL = ["（详见附件）", "（按现行规定执行）", "（具体以正式文件为准）",
                   "（后续另行说明）", "（相关部门已确认）"]
EN_PAREN_FORMAL = [" (see appendix)", " (per current policy)", " (details to follow)"]

EN_PAREN = [" (which is honestly a pain)", " (we tried this)",
            " (not sure it always works)", " (ymmv)",
            " (still working on it)"]

ZH_SLOGAN = [
    # ---- 按模型 coef 排序的高权重 AI 特征 (实测: 这些词最能拉高 AI 分) ----
    # 「是指」族
    (r"是指", "说的是"), (r"是指在", "说的是在"), (r"是一种", "算是一种"),
    (r"是一种(?=[^。，]{0,6}的)", "算是"),
    # 「用于」族
    (r"常用于", "经常用来"), (r"用于", "用来"), (r"用来", "拿来"),
    # 「确保」族
    (r"确保", "保证"), (r"以保证", "好让"),
    # 「通常」族
    (r"通常情况下", "一般情况"), (r"通常", "一般"),
    # 「可能」族
    (r"可能会", "可能会"), (r"可能", "也许"),
    # 「包括」族
    (r"包括", "里面有"), (r"以及其他", "还有别的"),
    # 「帮助」族
    (r"帮助", "帮着"), (r"有助于", "能帮着"),
    # 「许多」族
    (r"许多", "很多"), (r"这些都是", "这些都是"),
    # 书面连接词
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
    (r"是一种", "算是"), (r"能够", "能"),
    (r"提供", "给"), (r"使用", "用"), (r"进行", "做"),
    (r"如果您", "如果你"), (r"建议您", "建议你"),
    (r"我们应该", "咱们得"), (r"我们必须", "我们得"),
    (r"具有重要意义", "挺重要的"), (r"起到了重要作用", "起了大作用"),
    (r"发挥着重要作用", "起着大作用"), (r"具有重要价值", "很有价值"),
    (r"深入探讨", "好好聊聊"), (r"进一步分析", "再往下看"),
    (r"全面了解", "搞清楚"), (r"充分认识", "明白"),
    (r"高度重视", "看重"), (r"切实履行", "真做到"),
    (r"认真组织", "好好安排"),
    # ---- 针对「。它」「。这」「，并」这类高权重结构 ----
    (r"。它", "。这东西"), (r"。这种", "。这类"),
    (r"，并且", "，而且"), (r"，并", "，还"),
    # ---- 注入人类高频特征 (coef 最负) ----
    (r"是(?=[^。，]{2,8}的)", "算是"),
]

# ---------- 英文 ----------
EN_SUBJ = ["honestly", "i think", "in my experience", "to be fair",
           "personally", "from what i've seen", "look"]
EN_HEDGE = ["probably", "i guess", "sort of", "maybe", "basically", "pretty much"]
EN_CONNECT = ["though", "anyway", "also", "still", "plus", "but then"]

EN_SLOGAN = [
    # 整段删除的套话开头 (实测: 删掉比替换降幅大得多, 0.966->0.706)
    (r"\bit is important to note that\s*", ""),
    (r"\bit should be noted that\s*", ""),
    (r"\bit is worth noting that\s*", ""),
    (r"\bwe must carefully consider\b", "we gotta think about"),
    (r"\bcarefully consider\b", "really think about"),
    (r"\btransformative paradigm shift\b", "big change"),
    (r"\bparadigm shift\b", "big change"),
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
    formal = is_formal(text) if lang == "zh" else False
    if lang == "en":
        slogans, subjects, hedges, connects = EN_SLOGAN, EN_SUBJ, EN_HEDGE, EN_CONNECT
        parens = EN_PAREN
        puncts, sep = ".,;:", " "
    elif formal:
        # 公文模式: 不换套话(改了语域就崩), 不加油腔腔主语/衔接词,
        # 只用正式括注做"信息补充", 这是唯一不破坏语域的手段。
        slogans, subjects, hedges, connects = [], [], [], []
        parens = ZH_PAREN_FORMAL
        puncts, sep = "，、；：", "，"
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

        # 正式文书: 随机注入会让人有一大半时候"改完和原文一样",
        # 用户会以为功能坏了。这里改成确定性: 首句必插一个正式括注。
        # strength 控制插几个: 1.0 → 1 个, 2.0 → 2 个, 3.0 → 3 个(不超过句数)。
        injected_forced = 0
        if formal and sents:
            n_target = max(1, min(len(sents), int(round(strength))))
            order = sorted(range(len(sents)), key=lambda k: -len(sents[k]))
            used = 0
            for _i in order:
                if used >= n_target:
                    break
                _s = sents[_i]
                body = _s.rstrip("。！？；!?;.")
                end = _s[len(body):] or "。"
                if len(body) > 12:
                    sents[_i] = body + random.choice(parens) + end
                    used += 1
            injected_forced = used
        # 强制插过的要占用注入配额, 否则随机分支会再插一个 → 双括注
        injected = injected_forced

        for i, s in enumerate(sents):
            # 1) 套话替换
            for pat, rep in slogans:
                s = re.sub(pat, rep, s) if lang == "zh" else re.sub(pat, rep, s, flags=re.I)

            # 2) 注入口语标记, 每句最多一处 (叠加会变结巴, 必须互斥)
            if strength >= 2 and len(s) > 40 and not formal:
                # 长句拆短: 在中间逗号处断句(人类特征)
                # 但不能切在刚替换出来的衔接短语上 —— 否则会得到
                # "Sort of worth noting. We must..." 这种语义断裂
                BAD_CUTS = ("worth noting", "note that", "in short", "which means",
                            "big change", "in addition", "说白了", "反正", "顺带说",
                            "看得出来", "在这之上", "好好聊聊", "再往下看")
                mids = [m.start() for m in re.finditer(r"[，；,;]", s)
                        if not any(
                            s[max(0, m.start() - 20):m.start() + 25].lstrip().lower().startswith(b)
                            or b in s[max(0, m.start() - 20):m.start() + 25].lower()
                            for b in BAD_CUTS)]
                if mids and random.random() < 0.35 * (strength - 1):
                    cut = mids[len(mids) // 2]
                    s = s[:cut] + ("。" if lang == "zh" else ". ") + s[cut + 1:].lstrip()
            # 正式文书的注入已在上面确定性做过一次, 这里整体跳过 ——
            # 否则 subjects/hedges/connects 是空列表, random.choice([]) 会崩。
            if not formal and strength > 0 and len(s) > 14 and injected < max_inject:
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
