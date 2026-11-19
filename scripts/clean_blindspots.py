# -*- coding: utf-8 -*-
"""清洗盲区数据

抽查发现的问题:
  1. 元话语: "好的，请看以下的产品介绍。" / "好的，请提供纪录片的相关信息"
  2. 反问:   "好的，请告诉我您想评论的电影名称"
  3. 标题行: "这是一篇关于如何快速阅读一本书的指导文章。"
  4. markdown: "### 空气炸锅的特点：" / "---"
  5. 分节标记: "【影片简介】："
  6. 具体作品事实错误(模型编造获奨记录) —— 检测不了, 只能靠genre门控

策略: 严格清洗, 宁可少几条也不留脏数据。
"""
import json
import os
import re

_D = os.environ.get("AIGC_DATA_DIR") or "data"
SRC = os.path.join(_D, "ai_zh_blindspots.jsonl")
OUT = os.path.join(_D, "ai_zh_blindspots_clean.jsonl")

# 元话语/模型自述: 出现在开头就丢弃整条(它污染了整段的语域)
META_START = re.compile(
    # "好的，请看" 是元话语; "好的学习方法" 是形容词, 不能误伤
    r"^\s*好的[，,。：:]|^\s*当然[，,。：:]|^\s*当然可以[，,。：:]|"
    r"^\s*(以下是|下面是|这是一篇|请看以下|以下是一|"
    r"介绍如何|关于.{0,10}的(介绍|文章|指南))"
)
# 反问/请求补充信息: 出现就丢
META_ASK = re.compile(
    r"(请提供|请告诉我|请稍后|以便我|为您撰写|我为您|需要您提供|"
    r"如果没有具体|您想评论的是)"
)


def clean_one(t):
    """返回清洗后的文本, 或 None 表示应丢弃"""
    if not t:
        return None

    # 1) 开头是元话语 → 整条丢。这类句子会把"AI 助手口气"带进训练集,
    #    而我们要教的恰恰是"正文文风", 不是"助手口气"。
    if META_START.match(t):
        return None
    if META_ASK.search(t):
        return None

    # 2) 去掉 markdown 结构
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)      # ### 标题
    t = re.sub(r"^\s*-{3,}\s*$", "", t, flags=re.M)   # --- 分隔线
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)            # **粗体**
    t = re.sub(r"^【[^】]{1,12}】[：:]?\s*", "", t, flags=re.M)  # 【小节】

    # 3) 去掉字数标注
    t = re.sub(r"[（(]\s*约?\s*\d+\s*字(左右)?\s*[）)]", "", t)

    # 4) 去掉首行标题 —— 但只在它是"总标题"时。
    #    如果首行是"三、主要特点"这种中途小节标题, 说明前面的内容被切掉了,
    #    正文从半截开始, 整条必须丢弃(实测出现过从"三、主要特点"开始的产品介绍)。
    lines = [x for x in t.split("\n")]
    if lines and lines[0].strip():
        head = lines[0].strip()
        # "三、主要特点" 这种末尾小节: 序号是"三"及以上, 说明前面两节被吃了
        m_num = re.match(r"^([一二三四五六七八九十]+)[、.]", head)
        if m_num and len(head) < 20 and m_num.group(1) not in ("一",):
            return None
        while lines and lines[0].strip():
            head = lines[0].strip()
            if len(head) < 30 and not re.search(r"[。！？]", head):
                lines = lines[1:]
            else:
                break
    t = "\n".join(lines)

    # 5) 压缩空行
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # 6) 质量门
    if len(t) < 150:
        return None
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if cjk / max(len(t), 1) < 0.6:
        return None
    # 残留的元话语(句中)
    if META_ASK.search(t[:200]):
        return None
    return t


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    print(f"读入 {len(rows)} 条", flush=True)

    kept = {}
    dropped = {}
    for r in rows:
        g = r.get("genre", "?")
        t = clean_one(r.get("text", ""))
        if t is None:
            dropped[g] = dropped.get(g, 0) + 1
            continue
        kept.setdefault(g, []).append({**r, "text": t})

    with open(OUT, "w", encoding="utf-8") as f:
        total = 0
        for g, items in kept.items():
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                total += 1

    print(f"\n保留 {total} 条 → {OUT}")
    for g in sorted(set(list(kept) + list(dropped))):
        k = len(kept.get(g, []))
        d = dropped.get(g, 0)
        flag = " ⚠️" if k < 15 else ""
        print(f"  {g:8s} 保留 {k:3d}, 丢弃 {d:3d}{flag}")


if __name__ == "__main__":
    main()
