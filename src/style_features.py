# -*- coding: utf-8 -*-
"""风格特征函数(独立模块, 避免import时触发训练)"""
import re, numpy as np

AI_MARKERS = ["首先","其次","再次","最后","总的来说","综上所述","总而言之","需要注意","值得一提的是",
    "众所周知","显而易见","不可否认","在当今","随着","从某种意义","换句话说","一方面","另一方面",
    "因此","然而","此外","并且","用于","包括","例如","是一种","通常","可能","能够","提供","使用",
    "帮助","具有重要意义","我们应该","我们需要","建议您","如果您","我无法","总之","可见"]
HU_MARKERS = ["就是","其实","觉得","还是","一般","说白了","反正","哈哈","额","呃","吧","啊","呢",
    "我觉得","说实话","个人感觉","不知道","好像","可能吧","随便","什么的","之类的","挺","蛮"]

def style_features(texts):
    feats = []
    for t in texts:
        n = max(len(t), 1)
        sents = [s for s in re.split(r"[。！？!?；;]", t) if s.strip()]
        avg_sent = sum(len(s) for s in sents)/max(len(sents),1)
        comma = t.count("，")+t.count(",")
        period = t.count("。")+t.count(".")
        excl = t.count("！")+t.count("!")
        ques = t.count("？")+t.count("?")
        uniq = len(set(t))/n
        ai_m = sum(t.count(w) for w in AI_MARKERS)/n*100
        hu_m = sum(t.count(w) for w in HU_MARKERS)/n*100
        digits = sum(c.isdigit() for c in t)/n
        latin = sum(c.isalpha() and ord(c)<128 for c in t)/n
        newline = t.count("\n")/n*100
        feats.append([n, avg_sent, len(sents), comma/n*100, period/n*100,
                      excl/n*100, ques/n*100, uniq, ai_m, hu_m, digits, latin, newline])
    return np.array(feats, dtype=np.float32)
