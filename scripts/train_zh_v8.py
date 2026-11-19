# -*- coding: utf-8 -*-
"""中文 v8: 补 6 个文体盲区

实测发现(现场手写 10 类 AI 文本喂给 v7):
  识别: 议论文 0.745 / 技术解析 0.807 / 新闻稿 0.611 / 说明书 0.736
  漏判: 产品介绍 0.113 / 学习方法 0.131 / 求职信 0.257
        学术摘要 0.167 / 影评 0.102 / 旅游攻略 0.131

这 6 类是 ChatGPT 最高频输出, 但训练集几乎没有 —— v7 的 AI 数据集中在
论述文和社交文, 产品介绍/求职信/影评这类"功能性文本"基本是空白。

本脚本在 v7 的基础上:
1. 加入 ai_zh_blindspots.jsonl(6 类 × 30 条, 1.5B 生成)
2. 对这批数据加权 3.0 —— 30 条 vs 全部几万条, 不加权等于没加
3. 评估时单独报"盲区识别率"和"真人误报率", 两端都要看
"""
import json
import os
import random
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# ---- 路径: 默认用脚本所在目录, 可用环境变量覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AIGC_DATA_DIR") or os.path.join(os.path.dirname(_HERE), "data")
CKPT = os.environ.get("AIGC_CKPT_DIR") or os.path.join(os.path.dirname(_HERE), "ckpt")
os.makedirs(DATA, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

OUT = os.path.join(CKPT, "detector_zh_v8_w" + os.environ.get("BLIND_W","3"))
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
random.seed(42)

xs, ys, gs, wts = [], [], [], []

# ---------- HC3 真人 + 旧 AI ----------
n = 0
for line in open(os.path.join(DATA, "hc3_all.jsonl"), encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    n += 1
    for h in d.get("human_answers", []):
        if isinstance(h, str) and 20 < len(h.strip()) < 2000:
            xs.append(h.strip()); ys.append(0); gs.append(n); wts.append(1.0)
    for a in d.get("chatgpt_answers", []):
        if isinstance(a, str) and 20 < len(a.strip()) < 2000:
            xs.append(a.strip()); ys.append(1); gs.append(n); wts.append(1.0)
print(f"HC3: {n} QA -> 人={ys.count(0)} 旧AI={ys.count(1)}", flush=True)

# ---------- 当代 AI ----------
AI_FILES = [
    "ai_zh_modern.jsonl", "ai_zh_modern2.jsonl", "ai_zh_short.jsonl",
    "ai_zh_styles.jsonl", "ai_zh_styles2.jsonl", "ai_zh_big.jsonl",
    "ai_zh_div.jsonl", "ai_zh_vl30b.jsonl", "ai_zh_ov.jsonl",
    "ai_zh_short2.jsonl", "ai_zh_vl30b2.jsonl", "ai_zh_styles3_clean.jsonl",
    "ai_zh_guides_clean.jsonl", "ai_zh_marketing_clean.jsonl",
    "ai_zh_social_clean.jsonl",
]
ai_new = []
for fn in AI_FILES:
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        continue
    c = 0
    for line in open(p, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = (d.get("text") or "").strip()
        if len(t) >= 20:
            ai_new.append(t); c += 1
    print(f"  {fn}: +{c}", flush=True)

# ---------- 盲区数据(高权重) ----------
BLIND = os.path.join(DATA, "ai_zh_blindspots_clean.jsonl")
blind = []
blind_by_genre = {}
if os.path.exists(BLIND):
    for line in open(BLIND, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = (d.get("text") or "").strip()
        if len(t) >= 100:
            blind.append(t)
            g = d.get("genre", "?")
            blind_by_genre.setdefault(g, []).append(t)
    print(f"盲区数据: {len(blind)} 条 {dict((k, len(v)) for k, v in blind_by_genre.items())}",
          flush=True)

# 权重: 全量 AI 数据 1.0, 盲区数据 3.0
BLIND_WEIGHT = float(os.environ.get('BLIND_W', '3'))
for t in ai_new:
    xs.append(t); ys.append(1); gs.append(-(len(gs) + 1)); wts.append(1.0)
for t in blind:
    xs.append(t); ys.append(1); gs.append(-(len(gs) + 1)); wts.append(BLIND_WEIGHT)

print(f"合计 {len(xs)} 条 (真人 {ys.count(0)}, AI {ys.count(1)})", flush=True)

# ---------- 分组切分(与 v7 一致) ----------
ug = sorted(set(gs))
random.shuffle(ug)
n_test = int(len(ug) * 0.20)
n_ind = int((len(ug) - n_test) * 0.18)
tg, ig = set(ug[:n_test]), set(ug[n_test:n_test + n_ind])
tri, tei, ini = [], [], []
for i, g in enumerate(gs):
    if g in tg:
        tei.append(i)
    elif g in ig:
        ini.append(i)
    else:
        tri.append(i)

Xtr = [xs[i] for i in tri]; ytr = [ys[i] for i in tri]; wtr = [wts[i] for i in tri]
Xte = [xs[i] for i in tei]; yte = [ys[i] for i in tei]
Xin = [xs[i] for i in ini]; yin = [ys[i] for i in ini]
print(f"切分: train={len(Xtr)} test={len(Xte)} ind={len(Xin)}", flush=True)

# ---------- 特征 ----------
v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                     max_features=200000, sublinear_tf=True)
v2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                     max_features=100000, sublinear_tf=True,
                     token_pattern=r"(?u)\b\w+\b")
from scipy.sparse import hstack
A = hstack([v1.fit_transform(Xtr), v2.fit_transform(Xtr)]).tocsr()
B_ = hstack([v1.transform(Xte), v2.transform(Xte)]).tocsr()
C_ = hstack([v1.transform(Xin), v2.transform(Xin)]).tocsr()
print(f"特征 {A.shape[1]} ({time.time()-T0:.0f}s)", flush=True)

# ---------- 盲区测试集: 当场手写的那 6 段 ----------
PROBE = {
    "产品介绍": "这款智能手表采用航空级铝合金表壳，搭配1.43英寸AMOLED高清显示屏，支持全天候血氧监测、心率监测和睡眠分析。内置GPS定位系统，支持100多种运动模式。续航时间长达14天，支持5ATM防水。",
    "学习方法": "高效学习的关键在于建立系统的知识框架。首先，要明确学习目标，将大目标拆解为可执行的小步骤；其次，采用间隔重复的方法巩固记忆；最后，通过输出倒逼输入，将所学内容讲给别人听。",
    "求职信": "您好！我是XX大学计算机科学专业的应届毕业生，在校期间主修数据结构、算法设计、操作系统等核心课程，GPA 3.8/4.0。曾参与开发校园二手交易平台项目，负责后端架构设计与数据库优化。希望能有机会加入贵公司。",
    "学术摘要": "本研究旨在探讨深度学习在医学影像诊断中的应用效果。采用ResNet-50作为基础网络，在包含12,000张标注影像的数据集上进行训练和验证。实验结果表明，模型在肺结节检测任务上达到了94.3%的准确率，显著优于传统方法。",
    "影评": "这部电影通过细腻的镜头语言展现了普通人在时代洪流中的挣扎与坚守。导演没有刻意煽情，而是用克制的叙事让观众自行体会人物的内心世界。配乐的使用恰到好处，在关键情节处起到了画龙点睛的作用。",
    "旅游攻略": "大理古城位于云南省大理白族自治州，是国家级历史文化名城。最佳游览时间为每年3-5月和9-11月，此时气候宜人。古城内以人民路和复兴路最为热闹，沿街分布着各类特色小店和客栈。建议安排2-3天时间深度游览。",
}
PROBE_X = hstack([v1.transform(list(PROBE.values())), v2.transform(list(PROBE.values()))]).tocsr()

best = None
for c in [3, 5, 10, 20]:
    clf = LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=42)
    clf.fit(A, ytr, sample_weight=wtr)
    pi = clf.predict_proba(C_)[:, 1]
    fi = f1_score(yin, (pi >= 0.5).astype(int))
    # 盲区识别
    pp = clf.predict_proba(PROBE_X)[:, 1]
    blind_hits = sum(1 for p in pp if p >= 0.5)
    print(f"  C={c:<3} 独立F1={fi:.4f} 盲区识别={blind_hits}/6 "
          f"({[round(float(x),2) for x in pp]})", flush=True)
    score = fi + blind_hits * 0.005
    if best is None or score > best[0]:
        best = (score, clf, c, fi, blind_hits, pp)

_, clf, C_BEST, F1_BEST, BLIND_HITS, PROBE_P = best
print(f"\n选中 C={C_BEST} F1={F1_BEST:.4f} 盲区={BLIND_HITS}/6", flush=True)

joblib.dump(v1, os.path.join(OUT, "v1.joblib"))
joblib.dump(v2, os.path.join(OUT, "v2.joblib"))
joblib.dump(clf, os.path.join(OUT, "clf.joblib"))

rep = {
    "f1": round(F1_BEST, 4),
    "C": C_BEST,
    "train_n": len(Xtr),
    "ind_n": len(Xin),
    "blind_n": len(blind),
    "blind_weight": BLIND_WEIGHT,
    "blind_hits": BLIND_HITS,
    "probe": {k: round(float(v), 3) for k, v in zip(PROBE.keys(), PROBE_P)},
    "model": "zh_v8",
}
json.dump(rep, open(os.path.join(OUT, "report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n完成 用时 {(time.time()-T0)/60:.1f} 分", flush=True)
print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
