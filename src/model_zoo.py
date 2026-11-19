# -*- coding: utf-8 -*-
"""
AIGC 检测器 - 统一模型矩阵框架
按机型/性能自动选择最合适的模型档位，也可手动指定

档位:
  tier1  - TF-IDF+LR        (9MB,  F1~0.977, 手机/4GB, 毫秒级)
  tier2  - TFIDF-SVD+SVC    (187MB,F1~0.970, 主流PC, 需sklearn)
  tier2b - FastText         (3.7MB,词向量, 嵌入式/低配)
  tier3  - CharTextCNN      (3.3MB,F1~0.973, 主流PC, 需torch)
  en     - 英文TF-IDF       (demo级, 待真实数据)

用法:
  python model_zoo.py "文本"                 # 自动选档
  python model_zoo.py "文本" --tier tier3    # 指定档位
  python model_zoo.py --benchmark            # 跑各档基准对比
  python model_zoo.py --info                 # 显示机型检测结果和推荐档位
"""
import sys, os, json, time, platform
import joblib

BASE = r"D:\ai_train\ckpt"
# 自动定位项目根目录(兼容任意部署路径)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE) if os.path.basename(_HERE) == "src" else _HERE
BASE = os.path.join(_PROJ, "ckpt")
sys.path.insert(0, _HERE)

def detect_profile():
    """检测机型性能, 返回推荐档位"""
    import psutil
    cores = psutil.cpu_count(logical=False) or 4
    mem_gb = psutil.virtual_memory().total / 2**30
    # 有无torch
    try:
        import torch; has_torch = True
    except ImportError:
        has_torch = False
    if mem_gb < 4 or cores <= 2:
        rec = "tier1"
    elif mem_gb >= 8:
        rec = "tier1"        # 基准测试显示tier1增强版综合最优
    else:
        rec = "tier1"
    return {"cores": cores, "mem_gb": round(mem_gb, 1),
            "has_torch": has_torch, "recommended": rec,
            "os": platform.system()}

# ---- 各档推理实现 (带缓存) ----
_CACHE = {}

def _load_joblib(*parts):
    key = tuple(parts)
    if key not in _CACHE:
        _CACHE[key] = joblib.load(os.path.join(*parts))
    return _CACHE[key]

def pred_tier1(text):
    d = os.path.join(BASE, "detector_zh_final")        # v7: F1=0.9812(分组防泄漏), 2146条当代数据
    vec = _load_joblib(d, "tfidf.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def pred_tier1o(text):
    d = os.path.join(BASE, "detector_zh_tfidf")       # 原版
    vec = _load_joblib(d, "tfidf.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def pred_tier2(text):
    d = os.path.join(BASE, "detector_tier2b")
    clf = _load_joblib(d, "clf.joblib")
    vec = _load_joblib(d, "tfidf.joblib")
    svd = _load_joblib(d, "svd.joblib")
    scaler = _load_joblib(d, "scaler.joblib")
    from style_features import style_features
    import numpy as np
    sf = scaler.transform(style_features([text]))
    X = svd.transform(vec.transform([text]))
    F = np.hstack([X, sf])
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(F)[0]
        return float(1/(1+2.718281828**(-s)))
    return float(clf.predict_proba(F)[0, 1])

def _make_cnn_predictor(ckpt_dir):
    import torch, torch.nn as nn
    d = os.path.join(BASE, ckpt_dir)
    vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
    c2i, MAX_LEN = vocab["c2i"], vocab["MAX_LEN"]
    n_conv = len([k for k in torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=True) if k.startswith("convs.") and k.endswith(".weight")])
    class TextCNN(nn.Module):
        def __init__(self, v, embed=160, nk=5):
            super().__init__()
            self.emb = nn.Embedding(v+1, embed, padding_idx=0)
            ks = [2,3,4,5,6][:nk]
            self.convs = nn.ModuleList([nn.Conv1d(embed, 128, k) for k in ks])
            self.fc = nn.Linear(128*nk, 2); self.drop = nn.Dropout(0.35)
        def forward(self, x):
            e = self.emb(x).transpose(1,2)
            p = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
            return self.fc(self.drop(torch.cat(p, dim=1)))
    m = TextCNN(len(c2i), nk=max(n_conv,1))
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=True))
    m.eval()
    def _pred(text):
        ids = [c2i.get(c, 0) for c in text[:MAX_LEN]]
        ids += [0]*(MAX_LEN-len(ids))
        with torch.no_grad():
            return float(torch.softmax(m(torch.tensor([ids])), dim=1)[0, 1].item())
    return _pred

def pred_tier3(text):
    if "cnn_aug" not in _CACHE:
        _CACHE["cnn_aug"] = _make_cnn_predictor("detector_cnn_full")   # 全量训练版
    return _CACHE["cnn_aug"](text)

def pred_tier3o(text):
    if "cnn_orig" not in _CACHE:
        _CACHE["cnn_orig"] = _make_cnn_predictor("detector_tier3_cnn")  # 原版
    return _CACHE["cnn_orig"](text)

PREDICTORS = {"tier1": pred_tier1, "tier3": pred_tier3}

def pred_tier0(text):
    d = os.path.join(BASE, "detector_tier0")
    vec = _load_joblib(d, "vec.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def pred_ensemble(text):
    """集成: 严格评估显示 tier1 权重1.0+tier3 0.2 最优(F1=0.9691)"""
    return float(0.83 * pred_tier1(text) + 0.17 * pred_tier3(text))

def pred_tier2_light(text):
    """Tier2轻量版(41MB, F1=0.962)"""
    d = os.path.join(BASE, "detector_tier2_light")
    clf = _load_joblib(d, "clf.joblib")
    vec = _load_joblib(d, "tfidf.joblib")
    svd = _load_joblib(d, "svd.joblib")
    scaler = _load_joblib(d, "scaler.joblib")
    from style_features import style_features
    import numpy as np
    sf = scaler.transform(style_features([text]))
    X = svd.transform(vec.transform([text]))
    F = np.hstack([X, sf])
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(F)[0]
        return float(1/(1+2.718281828**(-s)))
    return float(clf.predict_proba(F)[0, 1])

def pred_en(text):
    """英文检测器 v7: 分组防泄漏切分 F1=0.9917, char_wb+word 双特征"""
    d = os.path.join(BASE, "detector_en_v5")
    clf = _load_joblib(d, "clf.joblib")
    # 双特征模型 (v1=char_wb, v2=word); 回退兼容单特征旧模型
    try:
        v1 = _load_joblib(d, "v1.joblib")
        v2 = _load_joblib(d, "v2.joblib")
        from scipy.sparse import hstack
        X = hstack([v1.transform([text]), v2.transform([text])])
    except Exception:
        vec = _load_joblib(d, "tfidf.joblib")
        X = vec.transform([text])
    return float(clf.predict_proba(X)[0, 1])

# 扩展档位 (已清理失效档位: tier1o/tier2/tier3o 对应模型已删除)
_ALL_PREDICTORS = dict(PREDICTORS)
_ALL_PREDICTORS["en"] = pred_en
_ALL_PREDICTORS["tier0"] = pred_tier0
_ALL_PREDICTORS["ensemble"] = pred_ensemble

def bucket(p):
    return "高" if p >= 0.7 else ("中" if p >= 0.3 else "低")

def detect(text, tier="auto"):
    lang = "zh"
    try:
        sys.path.insert(0, r"D:\ai_train")
        from lang_router import detect_lang_detail
        lang = detect_lang_detail(text)["lang"]
    except Exception:
        pass
    prof = detect_profile()
    # 英文自动路由到英文模型
    if lang == "en" and tier == "auto":
        chosen = "en"
    else:
        chosen = prof["recommended"] if tier == "auto" else tier
    if chosen not in _ALL_PREDICTORS:
        chosen = "tier1"
    t0 = time.time()
    p = _ALL_PREDICTORS[chosen](text)
    # 按长度给出置信度提示
    n = len(text)
    if n >= 150: confidence = "高(长文本~99.6%)"
    elif n >= 80: confidence = "中(80-150字~98.6%)"
    else: confidence = "低(<80字,仅供参考)"
    return {"language": lang, "tier": chosen, "ai_probability": round(p, 4),
            "bucket": bucket(p), "confidence": confidence, "text_len": n,
            "latency_ms": round((time.time()-t0)*1000, 1),
            "profile": prof if tier == "auto" else "manual"}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--info" in args:
        print(json.dumps(detect_profile(), ensure_ascii=False, indent=2)); sys.exit(0)
    if "--benchmark" in args:
        sample = "首先，人工智能的发展是一个重要话题。其次，从多个角度看都有意义。总的来说需要综合考虑各种因素。"
        for t in _ALL_PREDICTORS:
            try:
                r = detect(sample, tier=t)
                print(f"{t:8s} AI概率={r['ai_probability']:.3f} 耗时={r['latency_ms']}ms")
            except Exception as e:
                print(f"{t:8s} 失败: {type(e).__name__} {str(e)[:80]}")
        sys.exit(0)
    if not args:
        print("用法: python model_zoo.py \"文本\" [--tier tier1|tier2|tier3] [--benchmark] [--info]"); sys.exit(1)
    tier = "auto"
    if "--tier" in args:
        tier = args[args.index("--tier")+1]
    print(json.dumps(detect(args[0], tier), ensure_ascii=False, indent=2))
