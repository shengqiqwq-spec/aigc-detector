"""
语言判断路由 - 字符级CJK占比, 毫秒级, 不需训练
用途: 输入文本 -> 判定中文/英文 -> 路由到对应特化检测模型
"""
import re

CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
LATIN = re.compile(r'[A-Za-z]')

def detect_lang(text, cjk_threshold=0.20):
    """
    返回 'zh' 或 'en'
    规则: CJK字符占比 >= 20% -> 中文, 否则英文
    对中英混合文本以主导语言为准
    """
    if not text or not text.strip():
        return 'en'
    n = len(text)
    cjk = len(CJK.findall(text))
    ratio = cjk / n
    return 'zh' if ratio >= cjk_threshold else 'en'

def detect_lang_detail(text, cjk_threshold=0.20):
    n = len(text)
    cjk = len(CJK.findall(text))
    lat = len(LATIN.findall(text))
    ratio = cjk / n if n else 0
    lang = 'zh' if ratio >= cjk_threshold else 'en'
    return {'lang': lang, 'cjk_ratio': round(ratio, 3),
            'cjk_chars': cjk, 'latin_chars': lat, 'total': n}

if __name__ == "__main__":
    tests = [
        "这是一个中文句子，用来测试语言判断功能是否正确。",
        "This is an English sentence to test the language detection router.",
        "这是一个 mixed 中英混合 text 句子 with english words",
        "你好 world",
        "机器学习 machine learning 是 AI 的 core 部分",
    ]
    for t in tests:
        print(detect_lang_detail(t), "|", t[:30])
