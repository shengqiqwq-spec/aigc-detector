# -*- coding: utf-8 -*-
"""
AIGC 检测器 - 统一模型矩阵框架
按机型/性能自动选择最合适的模型档位，也可手动指定

档位:
  tier1  - TF-IDF+LR        (9MB,  F1~0.977, 手机/4GB, 毫秒级)
  tier2  - TFIDF-SVD+SVC    (187MB,F1~0.970, 主流PC, 需sklearn)
  tier2b - FastText         (3.7MB,词向量, 嵌入式/低配)
  tier3  - CharTextCNN      (5.1MB, 需torch, 无torch环境会自动跳过)
  en     - 英文TF-IDF       (demo级, 待真实数据)

用法:
  python model_zoo.py "文本"                 # 自动选档
  python model_zoo.py "文本" --tier tier0    # 指定档位
  python model_zoo.py --benchmark            # 跑各档基准对比
  python model_zoo.py --info                 # 显示机型检测结果和推荐档位
"""
import sys, os, json, time, platform
import joblib

# 自动定位项目根目录(兼容任意部署路径)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE) if os.path.basename(_HERE) == "src" else _HERE
BASE = os.path.join(_PROJ, "ckpt")
sys.path.insert(0, _HERE)

_PROFILE_CACHE = None

def detect_profile():
    """检测机型性能, 返回推荐档位 (结果缓存, 首次~1.7s, 后续0ms)"""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return dict(_PROFILE_CACHE)
    cores, mem_gb, has_torch = 4, 8.0, False
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or 4
        mem_gb = psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    # torch 只在需要时探测, 避免昂贵导入拖慢启动
    try:
        import importlib.util
        has_torch = importlib.util.find_spec("torch") is not None
    except Exception:
        has_torch = False
    if mem_gb < 4 or cores <= 2:
        rec = "tier1"
    elif mem_gb >= 8:
        rec = "tier1"        # 基准测试显示tier1增强版综合最优
    else:
        rec = "tier1"
    _PROFILE_CACHE = {"cores": cores, "mem_gb": round(mem_gb, 1),
                      "has_torch": has_torch, "recommended": rec,
                      "os": platform.system()}
    return dict(_PROFILE_CACHE)

# ---- 各档推理实现 (带缓存) ----
_CACHE = {}

def _load_joblib(*parts):
    """加载模型文件。优先 .pkl (pickle, 快17倍), 回退 .joblib。"""
    key = tuple(parts)
    if key not in _CACHE:
        path = os.path.join(*parts)
        # 同名 .pkl 优先
        pkl_path = path[:-len(".joblib")] + ".pkl" if path.endswith(".joblib") else path + ".pkl"
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, "rb") as f:
                _CACHE[key] = pickle.load(f)
        else:
            _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def warm_up(tiers=("tier1", "en")):
    """预加载模型到内存, 消除首次调用的冷启动延迟 (~2.7s -> 0)。
    可在服务启动时后台线程调用, 不阻塞主流程。"""
    _PROBE = "这是一个用于预热模型的测试句子，内容足够长以便模型能够完成判断。"
    # 一并预热硬件画像 (psutil 首次 import ~1.4s)
    try:
        detect_profile()
    except Exception:
        pass
    loaded = []
    for t in tiers:
        fn = _ALL_PREDICTORS.get(t)
        if fn is None:
            continue
        try:
            fn(_PROBE)
            loaded.append(t)
        except Exception:
            pass
    return loaded

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

def bucket(p):
    return "高" if p >= 0.7 else ("中" if p >= 0.3 else "低")

def detect(text, tier="auto"):
    # 输入校验: 空/纯空白/过短无意义输入直接返回明确结果, 不做无意义的概率猜测
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    # 有效字符: 中英文/数字
    meaningful = sum(1 for c in stripped if c.isalnum() or '\u4e00' <= c <= '\u9fff')
    if meaningful == 0:
        return {"language": "unknown", "tier": tier, "ai_probability": 0.0,
                "bucket": "低", "confidence": "无有效内容(纯符号/空白)",
                "text_len": len(text), "latency_ms": 0.0,
                "profile": "skipped", "error": "no_meaningful_content"}

    lang = "zh"
    try:
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
    # 分桶F1来自 report.json 的实测评测(分组防泄漏切分)
    if n >= 200: confidence = "高(长文本 F1~0.99)"
    elif n >= 80: confidence = "中(80-200字 F1~0.98)"
    elif n >= 40: confidence = "中低(40-80字 F1~0.95)"
    else: confidence = "低(<40字,仅供参考)"
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
        print(f"{'档位':<10}{'AI概率':>8}{'首次':>10}{'热推理(中位)':>14}")
        print("-" * 44)
        for t in _ALL_PREDICTORS:
            try:
                t0 = time.time()
                first = _ALL_PREDICTORS[t](sample)
                cold = (time.time() - t0) * 1000
                # 热身后取5次中位数, 才是真实推理速度
                times = []
                for _ in range(5):
                    t1 = time.time()
                    _ALL_PREDICTORS[t](sample)
                    times.append((time.time() - t1) * 1000)
                times.sort()
                print(f"{t:<10}{first:>8.3f}{cold:>9.0f}ms{times[2]:>12.2f}ms")
            except Exception as e:
                print(f"{t:<10} 失败: {type(e).__name__} {str(e)[:50]}")
        print("\n注: '首次'含模型加载/import开销, '热推理'是实际单次耗时")
        sys.exit(0)
    if not args:
        print("用法: python model_zoo.py \"文本\" [--tier tier1|tier2|tier3] [--benchmark] [--info]"); sys.exit(1)
    tier = "auto"
    if "--tier" in args:
        tier = args[args.index("--tier")+1]
    print(json.dumps(detect(args[0], tier), ensure_ascii=False, indent=2))
