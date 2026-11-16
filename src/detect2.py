
"""
AIGC检测统一推理入口 v2 (TF-IDF版, CPU秒级)
用法: 
  python detect2.py "待检测文本"          # 自动判断语言
  python detect2.py --batch file.txt      # 批量
流程: 语言判断 -> 加载对应模型 -> 输出AI概率 + Turnitin式分桶
"""
import sys, os, json
import joblib
sys.path.insert(0, r"D:\ai_train")
from lang_router import detect_lang_detail

MODELS = {
    "zh": r"D:\ai_train\ckpt\detector_tier1_best",
    "en": r"D:\ai_train\ckpt\detector_en_clean",
}
_cache = {}

def load(lang):
    if lang not in _cache:
        p = MODELS[lang]
        vec = joblib.load(os.path.join(p, "tfidf.joblib"))
        clf = joblib.load(os.path.join(p, "clf.joblib"))
        _cache[lang] = (vec, clf)
    return _cache[lang]

def detect(text):
    info = detect_lang_detail(text)
    lang = info["lang"]
    vec, clf = load(lang)
    X = vec.transform([text])
    prob = float(clf.predict_proba(X)[0, 1])
    bucket = "高" if prob >= 0.7 else ("中" if prob >= 0.3 else "低")
    return {"language": lang, "ai_probability": round(prob, 4),
            "bucket": bucket, "detail": info,
            "note": "英文模型为demo级,需真实数据重训" if lang=="en" else ""}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python detect2.py \"待检测文本\""); sys.exit(1)
    texts = [sys.argv[1]]
    if sys.argv[1] == "--batch" and len(sys.argv) > 2:
        texts = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    for t in texts:
        print(json.dumps(detect(t), ensure_ascii=False))
