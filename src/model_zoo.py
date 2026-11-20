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
import sys, os, json, time, platform, ctypes

# joblib 只被 sklearn 回退路径用到。快路径 (fast.npz / fast_nb.npz) 不需要它,
# 所以改成惰性导入 —— 否则一个只装了 numpy 的环境连快路径都起不来。

# 自动定位项目根目录(兼容任意部署路径)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE) if os.path.basename(_HERE) == "src" else _HERE
BASE = os.path.join(_PROJ, "ckpt")
sys.path.insert(0, _HERE)

_PROFILE_CACHE = None

def _mem_total_gb():
    """不依赖 psutil 的内存总量探测, 拿不到时返回 8.0(保守)"""
    try:
        if sys.platform == "win32":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / 2**30
        else:
            page = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return page * pages / 2**30
    except Exception:
        pass
    return 8.0


def detect_profile():
    """检测机型性能, 返回推荐档位 (结果缓存, 首次~1.7s, 后续0ms)"""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return dict(_PROFILE_CACHE)
    # 没有 psutil 时用标准库兜底: 只写死 4 核/8GB 会把 32GB 机器
    # 误判成低配, 进而给出错误的档位推荐。
    cores = os.cpu_count() or 4
    mem_gb = _mem_total_gb()
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or cores
        mem_gb = psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    has_torch = False
    # torch 只在需要时探测, 避免昂贵导入拖慢启动
    try:
        import importlib.util
        has_torch = importlib.util.find_spec("torch") is not None
    except Exception:
        has_torch = False
    # 低配机型推荐轻量档; 常规机型用主力档
    if mem_gb < 4 or cores <= 2:
        rec = "tier0"          # 手机/嵌入式: 用 4MB 轻量档
    else:
        rec = "tier1"          # 常规机型: tier1 综合最优
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
            import joblib          # 惰性: 只有没有 .pkl 时才需要
            _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def warm_up(tiers=("tier1", "en"), profile=False):
    """预加载模型到内存, 消除首次调用的冷启动延迟 (~1.3s -> 0)。

    可在服务启动时后台线程调用, 不阻塞主流程。

    profile=False (默认) 跳过硬件画像: psutil 首次 import 约 1.4s, 而纯检测
    不需要它。只有要展示逐档性能建议时才传 profile=True。
    """
    _PROBE = "这是一个用于预热模型的测试句子，内容足够长以便模型能够完成判断。"
    if profile:
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

# ---- 纯 numpy 快路径 (冷启动 1.7s -> 0.3s) ----
# 优先用 fast.npz; 没有 npz 或配置不支持时回退 sklearn。
_FAST_CACHE = {}


def _fast(model_dir, analyzer=None):
    """取缓存的 FastDetector, 不可用返回 None"""
    key = (model_dir, analyzer)
    if key in _FAST_CACHE:
        return _FAST_CACHE[key]
    try:
        from fast_infer import FastDetector
        d = FastDetector(model_dir, analyzer)
    except Exception:
        d = None
    _FAST_CACHE[key] = d
    return d


def _fast_nb(model_dir):
    """取缓存的 FastNB (Hashing+NB 快路径), 不可用返回 None"""
    key = ("nb", model_dir)
    if key in _FAST_CACHE:
        return _FAST_CACHE[key]
    try:
        from fast_nb import FastNB
        d = FastNB(model_dir)
    except Exception:
        d = None
    _FAST_CACHE[key] = d
    return d


def pred_tier1(text):
    d = os.path.join(BASE, "detector_zh_final")        # 分组防泄漏 F1=0.9802
    fd = _fast(d, "char")
    if fd is not None:
        return float(fd.predict_proba(text))
    vec = _load_joblib(d, "tfidf.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def pred_tier1o(text):
    d = os.path.join(BASE, "detector_zh_tfidf")       # 原版
    vec = _load_joblib(d, "tfidf.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def _make_cnn_predictor(ckpt_dir):
    """加载 CharTextCNN。兼容两种 checkpoint 格式:
       - 旧: model.pt 是纯 state_dict + 同目录 vocab.json, 输出 2 类 softmax
       - 新: model.pt 是 {state_dict, c2i, max_len, vocab_size} 单文件, 输出 1 logit
    """
    import torch, torch.nn as nn
    d = os.path.join(BASE, ckpt_dir)
    ckpt = torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # 新格式
        sd = ckpt["state_dict"]
        c2i = ckpt["c2i"] if not isinstance(ckpt["c2i"], str) else None
        if c2i is None:
            import json as _j
            c2i = _j.loads(ckpt["c2i"]) if isinstance(ckpt["c2i"], str) else ckpt["c2i"]
        MAX_LEN = ckpt["max_len"]
        emb_dim = sd["emb.weight"].shape[1]
        ks = [k for k in sd if k.startswith("convs.") and k.endswith(".weight")]
        nk = len(ks)
        out_ch = sd["convs.0.weight"].shape[0]
        out_dim = sd["fc.weight"].shape[0]   # 1 = 单logit, 2 = 双类
        conv_ks = [2,3,4,5,6][:nk]

        class CNN(nn.Module):
            def __init__(self, v, embed, nk_, out_ch_, out_dim_):
                super().__init__()
                self.emb = nn.Embedding(v+1, embed, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(embed, out_ch_, k) for k in conv_ks])
                self.drop = nn.Dropout(0.3)
                self.fc = nn.Linear(out_ch_*nk_, out_dim_)
            def forward(self, x):
                e = self.emb(x).transpose(1,2)
                h = [torch.relu(c(e)).max(dim=2)[0] for c in self.convs]
                return self.fc(self.drop(torch.cat(h,1)))
        m = CNN(len(c2i), emb_dim, nk, out_ch, out_dim)
        m.load_state_dict(sd)
        single_logit = (out_dim == 1)
    else:
        # 旧格式
        vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
        c2i, MAX_LEN = vocab["c2i"], vocab["MAX_LEN"]
        sd = ckpt
        emb_dim = sd["emb.weight"].shape[1]
        nk = len([k for k in sd if k.startswith("convs.") and k.endswith(".weight")])
        out_ch = sd["convs.0.weight"].shape[0]
        conv_ks = [2,3,4,5,6][:nk]
        class TextCNN(nn.Module):
            def __init__(self, v, embed, nk_):
                super().__init__()
                self.emb = nn.Embedding(v+1, embed, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(embed, out_ch, k) for k in conv_ks])
                self.fc = nn.Linear(out_ch*nk_, 2); self.drop = nn.Dropout(0.35)
            def forward(self, x):
                e = self.emb(x).transpose(1,2)
                p = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
                return self.fc(self.drop(torch.cat(p, dim=1)))
        m = TextCNN(len(c2i), emb_dim, nk)
        m.load_state_dict(sd)
        single_logit = False

    m.eval()
    def _pred(text):
        ids = [c2i.get(c, 0) for c in text[:MAX_LEN]]
        ids += [0]*(MAX_LEN-len(ids))
        with torch.no_grad():
            out = m(torch.tensor([ids]))
            if single_logit:
                return float(torch.sigmoid(out)[0].item())
            return float(torch.softmax(out, dim=1)[0, 1].item())
    return _pred

def pred_tier3(text):
    if "cnn_aug" not in _CACHE:
        try:
            _CACHE["cnn_aug"] = _make_cnn_predictor("detector_cnn_full")
        except ImportError as e:
            raise RuntimeError(
                "tier3 需要 PyTorch。安装: pip install torch  "
                "(或用默认的 tier1，准确率更高且无需 torch)"
            ) from e
    return _CACHE["cnn_aug"](text)

def pred_tier3o(text):
    if "cnn_orig" not in _CACHE:
        _CACHE["cnn_orig"] = _make_cnn_predictor("detector_tier3_cnn")  # 原版
    return _CACHE["cnn_orig"](text)

PREDICTORS = {"tier1": pred_tier1, "tier3": pred_tier3}

def pred_tier0(text):
    d = os.path.join(BASE, "detector_tier0")
    fd = _fast_nb(d)
    if fd is not None:
        return float(fd.predict_proba(text))
    vec = _load_joblib(d, "vec.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def pred_en_tier0(text):
    """英文轻量档: Hashing+NB, 手机/嵌入式"""
    d = os.path.join(BASE, "detector_en_tier0")
    fd = _fast_nb(d)
    if fd is not None:
        return float(fd.predict_proba(text))
    vec = _load_joblib(d, "vec.joblib")
    clf = _load_joblib(d, "clf.joblib")
    return float(clf.predict_proba(vec.transform([text]))[0, 1])

def pred_en(text):
    """英文检测器: 分组防泄漏切分 F1=0.9905, char_wb+word 双特征"""
    d = os.path.join(BASE, "detector_en_v5")
    fd = _fast(d)
    if fd is not None:
        return float(fd.predict_proba(text))
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
_ALL_PREDICTORS["en_tier0"] = pred_en_tier0

def bucket(p):
    return "高" if p >= 0.7 else ("中" if p >= 0.3 else "低")


def _reject_reason(t):
    """识别非自然语言输入, 返回原因字符串或 None

    JSON/HTML/代码行这些不是"写作风格"问题, 给它们打概率是误导。
    """
    import re as _re
    sample = t[:400]
    # JSON 对象/数组: 以 { 或 [ 开头, 且有 key: value 结构
    if _re.match(r"^\s*[\{\[]", sample) and _re.search(r'["\']?\w+["\']?\s*:', sample):
        return "JSON"
    # HTML/XML 标签
    if len(_re.findall(r"<[a-zA-Z/][^>]{0,40}>", sample)) >= 2:
        return "HTML"
    # 代码特征: 命中任意 2 项即可 (代码文本很短, 凑不满 3 项)
    # 必须用 sample 而不是 t: 这几条正则里 "\w+\(\w*[,)][^。；]*$" 带尾部
    # 定点回溯, 在 t 全文上跑会退化成 O(n^2)。实测 2 万字单行 955ms、
    # 20 万字 100 秒 —— 一个请求就能拖垮 HTTP 进程。
    # 判断"是不是代码"只看开头 400 字足够, 代码的缩进/关键字都在前面。
    code_signals = sum([
        bool(_re.search(r"^\s{2,}\S", sample, _re.M)),        # 缩进行
        bool(_re.search(r"\b(def|function|class|import|return|var|let|const|public|private)\s+\w", sample)),
        bool(_re.search(r"[{}()\[\]]\s*$", sample, _re.M)),   # 行末括号
        "==" in sample or "->" in sample or "::" in sample or ");" in sample,
        bool(_re.search(r"\w+\(\w*[,)][^。；]*$", sample)),    # 函数调用
    ])
    if code_signals >= 2:
        return "代码"
    # 纯数字/编号串
    if _re.fullmatch(r"[\d\s\.,;:\-\+\*/=()]+", t):
        return "纯数字/公式"
    # 超长无空格词串: 5000 个 a 不是写作风格, 是数据。
    # 只对拉丁字母串生效 —— 中文本来就不靠空格分词, 一整句话不打标点是
    # 正常的写作习惯(有人写东西就是一口气写下来), 不能判成非自然语言。
    if len(sample) > 30 and not _re.search(r"\s", sample[:1000]):
        _lat = len(_re.findall(r"[A-Za-z]", sample))
        _cjk = len(_re.findall(r"[\u4e00-\u9fff]", sample))
        if _cjk < len(sample) * 0.3 and _lat / len(sample) > 0.9:
            return "无分隔的长字符序列"
    # URL / base64 / 编码数据
    if _re.match(r"^\s*(https?|ftp)://\S+\s*$", sample):
        return "URL"
    # base64/base32 实测: 数字+符号占 0.10~0.15, 真实英文几乎为 0.000,
    # 所以 0.08 就能分开, 又不会误伤带数字的正常文本。
    if _re.fullmatch(r"[A-Za-z0-9+/=\s]{40,}", sample) and \
            len(_re.findall(r"[0-9+/=]", sample)) / len(sample) > 0.08:
        return "编码数据(base64等)"
    # 非中英文字符占主导: 模型没为这些语言训练过, 给了分也是瞎猜
    _letters = _re.findall(r"[A-Za-z\u4e00-\u9fff]", sample)
    _other_alpha = _re.findall(r"[^\W\d_A-Za-z\u4e00-\u9fff]", sample)
    if len(_other_alpha) > 10 and len(_other_alpha) > len(_letters):
        return "非中英文本(未训练的语言)"
    return None


def _repeat_ratio(t):
    """最长重复子串占全文的比例, 用于识别"同一句复制很多遍"的输入

    注意这个指标单独用会误伤: 一篇文章如果把 5 个段落复制两遍,
    ratio 也是 0.999, 但它不是垃圾 —— 那是常见的复制粘贴失误。
    所以 detect() 里还要看"这段内容本身有没有多样性"。
    """
    if len(t) < 40:
        return 0.0
    best = 0
    for span in (len(t)//8, len(t)//4, len(t)//2):
        if span < 20:
            continue
        head = t[:span]
        cnt = t.count(head)
        if cnt > 1:
            best = max(best, min(1.0, span * cnt / len(t)))
    return best


def _content_diversity(t):
    """不同句子的数量 / 句子总数

    用来区分两种"高重复":
      同一句复制 100 遍   -> diversity ≈ 0
      5 段文章复制 2 遍    -> diversity ≈ 0.5 (有 5 种不同内容)
    后者是复制粘贴失误, 不是垃圾输入, 应该照常检测。
    """
    import re as _re
    # 英文句号后跟空格才算断句, 否则 "3.14" 会被切开
    parts = _re.split(r"[。！？；\n]+|(?<=[.!?;])\s+", t)
    # 不要按长度过滤: "测试。" 只有 3 个字, 但重复 200 遍正是要检测的对象。
    # 之前用 len>=8 过滤, 结果这种文本 diversity 算出 1.0, 完全漏掉。
    sents = [x.strip() for x in parts if x.strip()]
    if not sents:
        return 1.0
    return len(set(sents)) / len(sents)


def detect(text, tier="auto"):
    # 输入校验: 空/纯空白/过短无意义输入直接返回明确结果, 不做无意义的概率猜测
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    # 超长输入保护必须放在最前面。
    # 放在后面时, 100 万字的文本会先在逐字符扫描(_meaningful)和
    # _reject_reason 的多条正则上耗掉几十秒, 长度检查根本轮不到执行。
    # 实测: 100 万字要 200 秒以上 —— HTTP 服务里一个请求就能打垮进程。
    MAX_CHARS = 500_000
    if len(text) > MAX_CHARS:
        return {"language": "unknown", "tier": tier, "ai_probability": 0.0,
                "bucket": "低",
                "confidence": f"文本过长({len(text)} 字，上限 {MAX_CHARS})，请分段检测",
                "text_len": len(text), "latency_ms": 0.0,
                "profile": "skipped", "error": "text_too_long"}

    stripped = text.strip()
    # 有效字符: 中英文/数字
    meaningful = sum(1 for c in stripped if c.isalnum() or '\u4e00' <= c <= '\u9fff')
    if meaningful == 0:
        return {"language": "unknown", "tier": tier, "ai_probability": 0.0,
                "bucket": "低", "confidence": "无有效内容(纯符号/空白)",
                "text_len": len(text), "latency_ms": 0.0,
                "profile": "skipped", "error": "no_meaningful_content"}

    # 非自然语言拒绝: 给 JSON/HTML/代码 打 AI 概率没有意义, 而且模型会
    # 输出 0.3~0.4 这种看起来煞有介事的假分数
    _reject = _reject_reason(stripped)
    if _reject:
        return {"language": "unknown", "tier": tier, "ai_probability": 0.0,
                "bucket": "低", "confidence": f"不是自然语言文本({_reject})",
                "text_len": len(text), "latency_ms": 0.0,
                "profile": "skipped", "error": "not_natural_language"}

    # 重复文本拒绝: "这是一段测试文本。" * 500 会被判成高 AI, 但那不是写作风格。
    # 必须同时看内容多样性: 一篇文章把 5 个段落复制两遍, ratio 也接近 1,
    # 但那是复制粘贴失误而不是垃圾输入, 应该照常检测。
    rep = _repeat_ratio(stripped)
    div = _content_diversity(stripped)
    if rep > 0.5 and div < 0.15 and len(stripped) > 200:
        return {"language": "unknown", "tier": tier, "ai_probability": 0.0,
                "bucket": "低", "confidence": f"重复内容占比 {rep:.0%}，无法判断写作风格",
                "text_len": len(text), "latency_ms": 0.0,
                "profile": "skipped", "error": "repetitive_content"}

    lang = "zh"
    try:
        from lang_router import detect_lang_detail
        lang = detect_lang_detail(text)["lang"]
    except Exception:
        pass
    prof = detect_profile()
    # 自动路由: 英文走英文模型; 低配机型走轻量档
    if tier == "auto":
        low_end = prof["recommended"] == "tier0"
        if lang == "en":
            chosen = "en_tier0" if low_end else "en"
        else:
            chosen = "tier0" if low_end else "tier1"
    else:
        chosen = tier
    if chosen not in _ALL_PREDICTORS:
        chosen = "tier1"
    t0 = time.time()
    p = _ALL_PREDICTORS[chosen](text)
    n = len(text)

    # 长度校准: 短文本信息不足, 概率应向 0.5 收缩, 避免给出虚假的极端值
    # 实测: 15字文本模型可能给 0.71, 但 15 字的判断本来就不可靠
    if n < 40:
        shrink = 0.45       # 最多收缩 45% 到 0.5
        p = p + (0.5 - p) * shrink
    elif n < 80:
        shrink = 0.25
        p = p + (0.5 - p) * shrink
    elif n < 150:
        shrink = 0.10
        p = p + (0.5 - p) * shrink
    # 英文模型(v8)的概率标定与中文不同: class_weight=balanced 会把整体分数压低,
    # 实测在 0.5 阈值下英文文体识别只剩 0/4, 而 0.30 阈值下误报 0%、识别 3/4。
    # 这里把英文分数重标定到与中文一致的口径, 让统一的 bucket() 阈值继续有效。
    if chosen.startswith("en"):
        CAL_LO, CAL_HI = 0.25, 0.80      # 原标度下相当于 0.5 / 0.99
        # 取 0.25 而非 0.30: 0.30 会把典型 AI 技术说明(原始 0.287)压到 0.488,
        # 卡在阈值下方一丁点。0.25 时该样本到 0.534, 真人误报仅 3/497(0.6%)。
        p = (p - CAL_LO) / (CAL_HI - CAL_LO) * 0.5 + 0.5
        p = min(0.999, max(0.001, p))

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
    # Windows 控制台默认 cp1252/cp936, 直接 print 中文会 UnicodeEncodeError。
    # 命令行入口强制 UTF-8, 避免"检测能跑但一输出就崩"。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = sys.argv[1:]
    if "--info" in args:
        print(json.dumps(detect_profile(), ensure_ascii=False, indent=2)); sys.exit(0)
    if "--benchmark" in args:
        # 中英各用对应语言的样本。以前统一用中文样本, en 档在中文上
        # 输出 0.051 —— 那个数字毫无意义, 只是让人以为英文模型坏了。
        ZH = ("首先，人工智能的发展是一个重要话题。其次，从多个角度看都有意义。"
              "总的来说需要综合考虑各种因素。")
        EN = ("To begin with, artificial intelligence is an important topic. "
              "Secondly, it matters from several perspectives. Overall, a "
              "balanced view is required when evaluating these systems.")
        print(f"{'档位':<10}{'AI概率':>8}{'首次':>10}{'热推理(中位)':>14}")
        print("-" * 44)
        for t in _ALL_PREDICTORS:
            sample = EN if t.startswith("en") else ZH
            try:
                t0 = time.perf_counter()
                first = _ALL_PREDICTORS[t](sample)
                cold = (time.perf_counter() - t0) * 1000
                # 预热后取 200 次中位数。5 次且用 time.time() 精度不够,
                # 会打印出 0.00ms 这种没意义的数字。
                times = []
                for _ in range(200):
                    t1 = time.perf_counter()
                    _ALL_PREDICTORS[t](sample)
                    times.append((time.perf_counter() - t1) * 1000)
                times.sort()
                med = times[len(times) // 2]
                print(f"{t:<10}{first:>8.3f}{cold:>9.0f}ms{med:>12.3f}ms")
            except Exception as e:
                print(f"{t:<10} 失败: {type(e).__name__} {str(e)[:50]}")
        print("\n注: '首次'含模型加载/import开销; '热推理'是 200 次调用的中位数")
        sys.exit(0)
    USAGE = ('用法: python model_zoo.py "文本" [--tier tier1|tier0|en|en_tier0] '
             '[--benchmark] [--info]')
    if not args:
        print(USAGE); sys.exit(1)

    # --tier 必须带值。以前直接 args[idx+1] 会 IndexError 崩掉。
    tier = "auto"
    if "--tier" in args:
        _i = args.index("--tier")
        if _i + 1 >= len(args):
            print(f"错误: --tier 后面缺少档位名\n{USAGE}", file=sys.stderr)
            sys.exit(2)
        tier = args[_i + 1]

    # 取第一个非选项、且不是 --tier 值 的参数作为待检测文本。
    # 以前直接 args[0], 于是 `--tier tier1` 会把 "--tier" 当成文本去检测,
    # 输出一个 JSON 好像真做了什么事。
    KNOWN_FLAGS = ("--benchmark", "--info", "--tier")
    text = None
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "--tier":
            skip_next = True
            continue
        if a in KNOWN_FLAGS:
            continue
        text = a
        break

    if text is None:
        print(f"错误: 没有给出待检测的文本\n{USAGE}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(detect(text, tier), ensure_ascii=False, indent=2))
