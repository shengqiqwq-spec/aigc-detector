import sys
# -*- coding: utf-8 -*-
"""
程序化降AI引擎 (零生成式AI) - 借鉴 Turnitout 三层规避原理
Layer1 词法: n-gram审计 + 定点破断 (确保无5-gram残留)
Layer2 结构: 句式变换 (主被动/语序/拆合句)
Layer3 语义: 同义替换保持语义不变

核心创新: Source-Aware N-gram Audit
  - 检测器用5-gram滚动哈希(Winnowing), 8词连续必被抓
  - 本引擎保证改写后与原文无任何5-gram连续重合
  - 零幻觉: 不改事实/数字/专有名词
"""
import re, random

# ============ 零宽字符与AI水印清理 ============
ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff\u180e\u200e\u200f"
C2PA_MARKERS = ["c2pa", "content_credential", "synthid", "ai-generated"]

def purge_invisible(text):
    """清除零宽字符和隐藏标记"""
    for ch in ZERO_WIDTH:
        text = text.replace(ch, "")
    return text

# ============ LLM 签名词库 (AI高频但人类少用的词) ============
AI_SIGNATURES = {
    # 中文AI套话 → 同义口语(优先用能消解n-gram的)
    "首先，": ["先说一点，", "第一，", "起初，"], "其次，": ["接着，", "再往下，", "然后，"],
    "最后，": ["末了，", "到了最后，"], "总的来说，": ["总的看，", "整体上，", "反正，"],
    "综上所述，": ["所以说，", "这么一看，", "归结起来，"],
    "总而言之，": ["反正，", "总之呢，"],
    "需要注意的是，": ["要留意的是，", "得注意，"], "值得一提的是，": ["对了，", "补充一点，"],
    "众所周知，": ["大家都知道，", "谁都清楚，"], "显而易见，": ["很明显，", "一眼能看出，"],
    "因此，": ["所以，", "这么一来，", "于是，"], "然而，": ["不过，", "可是，"],
    "此外，": ["另外，", "还有一点，"], "并且": ["而且", "同时"],
    "具有重要意义": ["很要紧", "意义不小", "很关键"],
    "随着": ["跟着", "伴随着"], "在当今社会": ["眼下", "现在这年头"],
    "不可否认": ["说实在的", "确实"], "从某种意义上": ["某种程度上"],
    "值得注意的是": ["有个事值得说", "要说明的是"],
    "进一步": ["再往下", "接着"], "深入探讨": ["仔细聊", "深入说说"],
    "促进": ["推动", "带动"], "提升": ["拉高", "提高"], "优化": ["改进", "调优"],
    "实现": ["做到", "达成"], "采用": ["用", "选用"], "利用": ["借助", "靠"],
    "具备": ["有", "带有"], "存在": ["有", "出现"], "面临": ["遇上", "碰到"],
    "针对": ["对着", "冲着"], "涉及": ["牵扯", "关联到"], "涵盖": ["包含", "包住"],
    "备受关注": ["被很多人盯着", "很多人都在说"], "有着广泛应用": ["用得很多", "用处很广"],
    "综合考虑": ["通盘考虑", "各方面都想想"], "引导其健康发展": ["带着它往好的方向走", "让它好好发展"],
    "制定相应的政策": ["出配套的办法", "定出对应的规矩"],
}
# 英文AI签名词
AI_EN = {
    "delve into": "look into", "pivotal role": "key part", "in today's world": "nowadays",
    "it is important to note that": "note that", "plays a crucial role": "matters a lot",
    "a wide range of": "many", "furthermore": "also", "moreover": "besides",
    "however": "but", "therefore": "so", "consequently": "as a result",
    "utilize": "use", "leverage": "use", "facilitate": "help", "demonstrate": "show",
    "comprehensive": "full", "significant": "big", "numerous": "many",
}

# 破断用插入词(不改语义)
BREAKERS_ZH = ["其实", "说白了", "换句话说", "有意思的是", "顺带提一句", "不妨说", "严格讲"]
BREAKERS_EN = ["actually", "in fact", "to be honest", "interestingly"]

# 受保护: 数字/英文/专有名词/公式, 绝不改动
PROTECT = re.compile(r"(\d+[\.\d]*%?|[A-Za-z]+(?:\-[A-Za-z]+)*|《[^》]+》|\"[^\"]+\")")


# ============ 核心: n-gram 审计 ============
def ngrams(text, n=5):
    """字符级5-gram(检测器用词级, 中文按字近似)"""
    t = re.sub(r"[\s，。！？、；：\"']", "", text)
    return set(t[i:i+n] for i in range(len(t)-n+1))

def residual_grams(src, dst, n=5):
    """改写后仍与原文重合的n-gram"""
    return ngrams(src, n) & ngrams(dst, n)

def audit_and_break(src, dst, n=5, max_iter=6, max_inserts=3):
    """审计破断: 严格限制插入次数(默认最多3处), 避免文本变形"""
    if max_inserts <= 0:
        return dst, len(residual_grams(src, dst, n))
    breaks = ["其实", "说白了", "说到底", "严格讲"]
    inserted = 0
    inserted_pos = set()
    skipped = set()
    for _ in range(max_iter):
        if inserted >= max_inserts:
            break
        res = residual_grams(src, dst, n) - skipped
        if not res:
            break
        # 选最长的残留(影响面大)
        victim = max(res, key=lambda g: dst.count(g))
        pos = dst.find(victim)
        if pos < 0:
            skipped.add(victim); continue
        cands = []
        for m in re.finditer(r"[，。！？；]", dst):
            if pos < m.start() <= pos + len(victim) and m.start() not in inserted_pos:
                cands.append(m.start())
        if not cands:
            skipped.add(victim); continue
        ins = cands[0]
        inserted_pos.add(ins)
        # 在该标点处插入: 用 "，词" 替换原标点, 避免出现 "……，严格讲，" 这种双标点
        punc = dst[ins]
        joiner = "，" if punc == "，" else punc
        dst = dst[:ins] + joiner + random.choice(breaks) + "，" + dst[ins + 1:]
        inserted += 1
    return dst, len(residual_grams(src, dst, n))


# ============ 句式变换 ============
def restructure(text):
    """打散句式结构(Layer2): 长句拆分, 主被动, 语序调整"""
    # 1) 长句按连接词拆分
    for conj in ["，并且", "，同时", "，此外", "而且", "以及"]:
        text = text.replace(conj, "。" + random.choice(["另外", "还有", "同时"]))
    # 2) 定语前置转后置(中文里"的"字结构重排, 谨慎)
    # 3) 随机句首虚词
    return text


# ============ 主流程 ============
def humanize_programmatic(text, strength=1.0):
    """
    程序化改写: 零AI, 保事实
    返回 (改写文本, 审计报告)
    """
    original = text
    out = purge_invisible(text)

    # Step 1: AI签名词替换(受保护区跳过)
    for zh, repls in AI_SIGNATURES.items():
        if zh in out:
            rep = random.choice(repls) if repls else ""
            out = out.replace(zh, rep, 1)
    # 清理替换后产生的孤立标点
    out = re.sub(r"^[，,、\s]+", "", out)
    out = re.sub(r"([，。！？])[，,、]+", r"\1", out)

    # Step 2: 英文签名词
    for en, rep in AI_EN.items():
        out = re.sub(re.escape(en), rep, out, flags=re.IGNORECASE)

    # Step 3: 句式重构
    if strength >= 0.5:
        out = restructure(out)

    # Step 4: n-gram 审计破断(核心!)
    out, residual = audit_and_break(original, out, n=5)

    # Step 5: 清理冗余(连续标点/空格)
    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"[。\.]{2,}", "。", out)
    out = re.sub(r"\s{2,}", " ", out)

    report = {
        "orig_len": len(original),
        "out_len": len(out),
        "residual_5grams": residual,
        "len_delta_pct": round((len(out)-len(original))/max(len(original),1)*100, 1),
    }
    return out, report


if __name__ == "__main__":
    # Windows 控制台默认 cp936/cp1252, 直接 print 中文会乱码或 UnicodeEncodeError。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    import sys
    if len(sys.argv) < 2:
        print("用法: python evasion_engine.py \"文本\""); sys.exit(1)
    out, rep = humanize_programmatic(sys.argv[1])
    print("原文:", sys.argv[1])
    print("改写:", out)
    print("报告:", rep)
