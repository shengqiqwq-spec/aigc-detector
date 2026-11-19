# -*- coding: utf-8 -*-
"""CLI 与公共 API 测试

model_zoo.py 是项目的门面（README 里第一条命令就是它），但覆盖率只有 49%。
这里把命令行入口和几条公共路径都跑一遍。
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SCRIPT = os.path.join(SRC, "model_zoo.py")


def _run(*args, timeout=300):
    """跑 CLI 并强制按 UTF-8 解码 stdout/stderr

    Windows 上 text=True 会用系统默认编码解码, 而 CLI 现在输出 UTF-8 —— 
    解码失败时 stdout 会变成 None, 后续断言就报 NoneType 不可迭代。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


class TestCLI:
    def test_detect_chinese(self):
        r = _run("人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能提高效率；"
                 "在教育领域，个性化学习系统能因材施教。")
        assert r.returncode == 0, r.stderr[-400:]
        data = json.loads(r.stdout)
        assert data["language"] == "zh"
        assert 0.0 <= data["ai_probability"] <= 1.0

    def test_detect_english(self):
        r = _run("Artificial intelligence represents a transformative paradigm shift "
                 "in modern society, and we must consider the implications carefully.")
        assert r.returncode == 0, r.stderr[-400:]
        data = json.loads(r.stdout)
        assert data["language"] == "en"

    def test_info_flag(self):
        r = _run("--info")
        assert r.returncode == 0, r.stderr[-400:]
        data = json.loads(r.stdout)
        for k in ("cores", "mem_gb", "recommended"):
            assert k in data

    def test_benchmark_flag(self):
        r = _run("--benchmark", timeout=600)
        assert r.returncode == 0, r.stderr[-400:]
        out = r.stdout
        assert "tier1" in out
        assert "en" in out
        # 不能出现 0.00ms 这种没意义的计时(time.time() 精度不够)
        assert "0.00ms" not in out.replace(" ", "")

    def test_benchmark_uses_matching_language_sample(self):
        """en 档必须用英文样本测

        以前统一用中文样本, en 档输出 0.051, 让人以为英文模型坏了。
        """
        r = _run("--benchmark", timeout=600)
        assert r.returncode == 0, r.stderr[-400:]
        for line in r.stdout.splitlines():
            if line.startswith("en ") or line.startswith("en_tier0"):
                p_ = float(line.split()[1])
                assert p_ > 0.3, f"en 档在英文样本上得分过低: {line}"
            if line.startswith("tier1"):
                p_ = float(line.split()[1])
                assert p_ > 0.3, f"tier1 在中文样本上得分过低: {line}"

    def test_explicit_tier(self):
        r = _run("这是一段用于测试指定档位的中文文本内容。", "--tier", "tier0")
        assert r.returncode == 0, r.stderr[-400:]
        assert json.loads(r.stdout)["tier"] == "tier0"

    def test_unknown_tier_falls_back(self):
        """给个不存在的档位不该崩, 应优雅回退"""
        r = _run("这是一段用于测试的中文文本内容。", "--tier", "nonexistent")
        assert r.returncode == 0, r.stderr[-400:]
        assert json.loads(r.stdout)["tier"] in ("tier1", "tier0", "en", "en_tier0")

    def test_empty_stdin_like_input(self):
        r = _run("")
        assert r.returncode == 0, r.stderr[-400:]
        data = json.loads(r.stdout)
        assert data.get("error") == "no_meaningful_content"


class TestConsoleEncoding:
    """Windows 控制台默认 cp1252/cp936, 输出中文会崩

    这个 bug 只在 Windows runner 上暴露: 本机终端恰好是 UTF-8, 一切都正常,
    但任何用 cmd/PowerShell 的用户跑 CLI 都会 UnicodeEncodeError。
    """

    def test_chinese_output_under_cp1252(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        env["PYTHONIOENCODING"] = "cp1252"
        r = subprocess.run(
            [sys.executable, SCRIPT, "人工智能技术正在深刻改变各行各业。"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300, env=env,
        )
        assert r.returncode == 0, f"cp1252 下崩溃: {r.stderr[-400:]}"
        assert "ai_probability" in r.stdout

    def test_info_under_cp1252(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        env["PYTHONIOENCODING"] = "cp1252"
        r = subprocess.run([sys.executable, SCRIPT, "--info"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180, env=env)
        assert r.returncode == 0, f"cp1252 下 --info 崩溃: {r.stderr[-400:]}"


class TestPublicAPI:
    """README 里示范的调用方式, 必须一直可用"""

    def test_detect_returns_required_keys(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        r = M.detect("这是一段足够长的中文测试文本，用于验证返回字段完整。")
        for k in ("language", "tier", "ai_probability", "bucket", "confidence"):
            assert k in r, f"detect() 缺少 {k}"

    def test_bucket_boundaries(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        assert M.bucket(0.95) == "高"
        assert M.bucket(0.7) == "高"
        assert M.bucket(0.69) == "中"
        assert M.bucket(0.3) == "中"
        assert M.bucket(0.29) == "低"

    def test_warm_up_returns_loaded_tiers(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        loaded = M.warm_up(("tier1", "en"))
        assert "tier1" in loaded
        assert "en" in loaded

    def test_warm_up_with_bad_tier_does_not_raise(self):
        """给了不存在的档位名, warm_up 应跳过而不是抛异常"""
        sys.path.insert(0, SRC)
        import model_zoo as M
        loaded = M.warm_up(("does_not_exist",))
        assert loaded == []

    def test_detect_none_input(self):
        """None 不该崩"""
        sys.path.insert(0, SRC)
        import model_zoo as M
        r = M.detect(None)
        assert "ai_probability" in r

    def test_detect_non_string_input(self):
        """数字输入不该崩"""
        sys.path.insert(0, SRC)
        import model_zoo as M
        r = M.detect(12345)
        assert "ai_probability" in r

    def test_length_shrinkage_direction(self):
        """短文本概率应被拉向 0.5, 且 confidence 标为低"""
        sys.path.insert(0, SRC)
        import model_zoo as M
        short = M.detect("人工智能正在改变世界。")
        assert "低" in short["confidence"] or "中低" in short["confidence"]

    def test_long_text_high_confidence(self):
        """长文本置信度标为高

        注意不能用重复句子拼长度 —— 那会被重复检测拦下(这是对的),
        要写一段真正长、内容不重复的文本。
        """
        sys.path.insert(0, SRC)
        import model_zoo as M
        long_text = (
            "人工智能技术正在深刻改变各行各业的面貌。在医疗领域，AI辅助诊断系统能够"
            "帮助医生更快识别影像中的异常，缩短病人的等待时间；在教育领域，个性化学习"
            "平台根据每个学生的进度调整练习难度，让因材施教不再只是一句口号；在交通"
            "领域，智能调度算法实时优化信号灯配时，缓解了高峰时段的拥堵状况。不过，"
            "技术带来便利的同时也引发了新的担忧。数据隐私如何保护，算法偏见怎样纠正，"
            "责任边界应该划在哪里，这些都需要在推广过程中逐步建立规范。有观点认为，"
            "与其纠结于技术本身，不如把注意力放在使用它的人和制度设计上。毕竟工具从来"
            "都是中性的，关键在于我们如何选择使用它。"
        )
        r = M.detect(long_text)
        assert "高" in r["confidence"]


class TestLangRouterBoundaries:
    """语言判断的边界 —— 判错语言会让整个模型选错"""

    def test_pure_chinese(self):
        sys.path.insert(0, SRC)
        import lang_router as L
        assert L.detect_lang("这是一个中文句子。") == "zh"

    def test_pure_english(self):
        sys.path.insert(0, SRC)
        import lang_router as L
        assert L.detect_lang("This is an English sentence.") == "en"

    def test_mixed_prefers_dominant(self):
        sys.path.insert(0, SRC)
        import lang_router as L
        assert L.detect_lang("机器学习 machine learning 是 AI 的核心部分") == "zh"
        assert L.detect_lang("The quick brown fox 跳过了 the lazy dog") == "en"

    def test_empty_and_whitespace_default_english(self):
        sys.path.insert(0, SRC)
        import lang_router as L
        assert L.detect_lang("") == "en"
        assert L.detect_lang("   \n\t ") == "en"

    def test_threshold_boundary(self):
        """20% 是阈值: 恰好 20% 判中文, 略低于则判英文"""
        sys.path.insert(0, SRC)
        import lang_router as L
        exactly_20 = "a" * 8 + "中中"        # 2/10
        assert L.detect_lang(exactly_20) == "zh"
        under_20 = "a" * 9 + "中"            # 1/10
        assert L.detect_lang(under_20) == "en"

    def test_detail_fields(self):
        sys.path.insert(0, SRC)
        import lang_router as L
        d = L.detect_lang_detail("中文 abc")
        for k in ("lang", "cjk_ratio", "cjk_chars", "latin_chars", "total"):
            assert k in d
        assert d["cjk_chars"] == 2
        assert d["latin_chars"] == 3

    def test_detail_on_empty_does_not_divide_by_zero(self):
        sys.path.insert(0, SRC)
        import lang_router as L
        d = L.detect_lang_detail("")
        assert d["cjk_ratio"] == 0
        assert d["total"] == 0

    def test_japanese_kanji_is_detected_as_chinese(self):
        """已知局限: 日文汉字落在 CJK 区, 会被判成中文

        这不是可以修的 bug (纯字符区间判断分不出中日), 但必须让行为可见:
        如果一个用户拿日文来检测, 它会走中文模型。
        """
        sys.path.insert(0, SRC)
        import lang_router as L
        d = L.detect_lang_detail("日本語のテキストです")
        assert d["lang"] == "zh", "行为变了, 请更新这条测试和文档"


class TestProfileWithoutPsutil:
    """没有 psutil 时不能瞎猜机型

    历史 bug: 无 psutil 时写死 cores=4/mem_gb=8.0, 把 32GB 机器误判成低配。
    requirements.txt 里 psutil 是可选的, 所以大多数用户都走这条路径。
    """

    def test_memory_probe_without_psutil(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        gb = M._mem_total_gb()
        assert 1.0 < gb < 4096.0, f"内存探测返回不合理: {gb}"

    def test_memory_probe_matches_psutil(self):
        """标准库探测应与 psutil 结果接近(±5%)"""
        sys.path.insert(0, SRC)
        import model_zoo as M
        psutil = pytest.importorskip("psutil")
        ours = M._mem_total_gb()
        theirs = psutil.virtual_memory().total / 2**30
        assert abs(ours - theirs) / theirs < 0.05, f"{ours} vs {theirs}"

    def test_cores_not_hardcoded(self):
        """核数应来自系统, 不是常数 4"""
        sys.path.insert(0, SRC)
        import model_zoo as M
        import os as _os
        real = _os.cpu_count()
        if real is None:
            pytest.skip("系统不报核数")
        M._PROFILE_CACHE = None
        p = M.detect_profile()
        # 有 psutil 时取物理核(可能小于逻辑核); 无 psutil 时等于逻辑核
        assert 1 <= p["cores"] <= real, f"核数 {p['cores']} 超出系统实际 {real}"


class TestNonProseRejection:
    """给非自然语言打分是误导, 但也不能误伤正常文本

    实测发现几个漏网: 5000 个 a 给了 0.317、base64 给了 0.322、URL 给了 0.402、
    阿拉伯文给了 0.421 —— 这些分数看起来煞有介事, 实际毫无意义。
    """

    REJECT = [
        ("超长无空格串", "a" * 5000),
        ("base64", "aGVsbG8gd29ybGQgdGhpcyBpcyBhIHRlc3Q=" * 20),
        ("URL", "https://example.com/very/long/path?with=params&and=more#fragment"),
        ("阿拉伯文", "هذا نص عربي للاختبار وهو نص طويل بما يكفي للتحليل"),
        ("日文", "日本語のテキストです。これはテスト用の文章です。"),
        ("XML", '<?xml version="1.0"?><root><item>test</item></root>'),
        ("纯数字", "1234567890" * 50),
    ]

    ACCEPT = [
        ("正常中文", "人工智能技术正在深刻改变各行各业的方方面面。"),
        ("正常英文", "Artificial intelligence is changing every industry in profound ways."),
        ("带数字中文", "2024年第三季度营收达到 15.3 亿元，同比增长 8.7%。"),
        ("带年份英文", "In 2024, revenue reached 15.3 billion, up 8.7% year over year."),
        ("中英混排", "机器学习 machine learning 是 AI 的核心部分。"),
        ("带URL中文", "详情访问 https://example.com 查看，内容挺全面。"),
        ("带括号公式", "根据公式 f(x) = x^2 + 1，可以推导出结果。"),
    ]

    def test_non_prose_is_rejected(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        for name, text in self.REJECT:
            d = M.detect(text)
            assert d.get("error"), f"{name} 没有被拒绝, 给了 {d['ai_probability']}"

    def test_chinese_without_punctuation_is_accepted(self):
        """中文不靠空格分词, 一整句不打标点是正常写法

        曾经被「无分隔长字符序列」规则误伤: 5000个a 该拒, 但一段 60 字的
        正常中文句子不该拒。
        """
        sys.path.insert(0, SRC)
        import model_zoo as M
        t = ("在这个信息爆炸的时代我们需要学会筛选有价值的内容而不是被大量的"
             "碎片化信息所淹没因此培养批判性思维和独立判断能力显得尤为重要。")
        d = M.detect(t)
        assert not d.get("error"), f"正常中文被误拒: {d['error']}"

    def test_normal_text_is_accepted(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        for name, text in self.ACCEPT:
            d = M.detect(text)
            assert not d.get("error"), f"{name} 被误拒: {d['error']}"


class TestRepetitionGuardPrecision:
    """重复检测要分清「垃圾重复」和「复制粘贴失误」

    实测发现: 5 段真实文章复制两遍, repeat_ratio 也是 0.999,
    但那是常见的手滑, 不是垃圾输入 —— 之前的判定会直接拒绝它,
    用户贴一篇正常报告却被说「内容重复度过高」。
    """

    PARAS = [
        "人工智能技术正在深刻改变医疗行业的面貌。影像辅助诊断系统已经能够识别肺部结节、"
        "乳腺病变等多种病灶，其准确率在部分任务上超过了初级放射科医生。",
        "教育领域的变化同样显著。个性化学习平台通过分析学生的答题记录，定位知识薄弱点，"
        "动态调整练习难度。相比传统的一刀切教学，这种方式让每个学生都能按照自己的节奏前进。",
        "交通领域的应用更加直观。智能信号灯控制系统根据实时车流调整配时，在高峰时段能显著"
        "缓解拥堵。自动驾驶技术虽然尚未完全成熟，但在园区、港口等封闭场景已经实现商业化运营。",
    ]

    def test_article_repeated_twice_is_not_rejected(self):
        """整篇复制两遍应该照常检测"""
        import model_zoo as M
        t = "\n".join(self.PARAS * 2)
        d = M.detect(t)
        assert d.get("error") != "repetitive_content", \
            "正常文章复制两遍被误判为重复内容"

    def test_same_sentence_many_times_is_rejected(self):
        import model_zoo as M
        assert M.detect("这是一段测试文本。" * 100).get("error") == "repetitive_content"

    def test_same_sentence_many_times_english(self):
        """英文句号后跟空格才算断句, 不能因为按中文标点切分而漏掉"""
        import model_zoo as M
        assert M.detect("This is a test sentence. " * 50).get("error") == "repetitive_content"

    def test_very_short_repeated_sentence(self):
        """「测试。」只有 3 个字 —— 分句时不能按长度过滤掉它

        之前用 len>=8 过滤, 结果 diversity 算出 1.0, 重复 200 遍的文本漏网。
        """
        import model_zoo as M
        assert M.detect("测试。" * 200).get("error") == "repetitive_content"

    def test_content_diversity_metric(self):
        import model_zoo as M
        # 同一句重复: 多样性接近 0
        assert M._content_diversity("这是一段测试文本。" * 50) < 0.1
        # 不同内容: 多样性接近 1
        assert M._content_diversity("\n".join(self.PARAS)) > 0.9


class TestFormalRegister:
    """公文改写不能毁语域

    UI 实跑时发现: 公文被改成 "高度重视→看重"、"认真组织→好好安排"，
    还插了 "（我们试过）" —— 读起来像小学生模仿公文。
    语域崩塌比留着 AI 味更糟。
    """

    OFFICIAL = ("各部门：为提升全体员工的专业技能与综合素质，进一步推动公司"
                "高质量发展，经研究决定开展年度员工培训计划。现将有关事项通知"
                "如下：请各部门高度重视，认真组织实施，确保培训取得实效。")

    def test_official_document_detected(self):
        sys.path.insert(0, SRC)
        import humanize2 as H
        assert H.is_formal(self.OFFICIAL) is True

    def test_casual_text_not_formal(self):
        sys.path.insert(0, SRC)
        import humanize2 as H
        assert H.is_formal("人工智能技术正在深刻改变各行各业。") is False

    def test_official_keeps_register(self):
        """公文改写后不能出现口语词"""
        sys.path.insert(0, SRC)
        import humanize2 as H
        out = H.humanize_v2(self.OFFICIAL)
        for bad in ("看重", "好好", "我们试过", "说实话", "挺关键", "效果一般"):
            assert bad not in out, f"公文里出现了口语词「{bad}」"

    def test_official_actually_changes(self):
        """必须真的改动, 不能原样返回(否则用户以为坏了)"""
        sys.path.insert(0, SRC)
        import humanize2 as H
        for _ in range(5):
            assert H.humanize_v2(self.OFFICIAL) != self.OFFICIAL

    def test_official_single_paren(self):
        """不能叠括号 —— 曾经出现（A）（B）连排"""
        sys.path.insert(0, SRC)
        import humanize2 as H
        for _ in range(5):
            out = H.humanize_v2(self.OFFICIAL)
            assert out.count("（") <= 1, f"括注叠加: {out}"

    def test_casual_still_gets_colloquial(self):
        """非公文必须仍然口语化, 否则这个功能没意义"""
        sys.path.insert(0, SRC)
        import humanize2 as H
        casual = "人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能够提高效率。"
        out = H.humanize_v2(casual)
        assert out != casual


class TestModelFiles:
    """仓库发布的模型文件必须完整且不含 pickle"""

    def test_all_tiers_have_npz(self):
        sys.path.insert(0, SRC)
        import model_zoo as M
        import glob
        for d in glob.glob(os.path.join(M.BASE, "detector_*")):
            name = os.path.basename(d)
            if name == "detector_cnn_full":
                continue
            has = (os.path.exists(os.path.join(d, "fast.npz"))
                   or os.path.exists(os.path.join(d, "fast_nb.npz")))
            assert has, f"{name} 缺少 npz 快路径"

    def test_npz_has_no_pickle_payload(self):
        """npz 必须能用 allow_pickle=False 打开, 且不含 object 数组"""
        import numpy as np
        import glob
        files = glob.glob(os.path.join(ROOT, "ckpt", "*", "*.npz"))
        assert files, "没找到任何 npz"
        for f in files:
            z = np.load(f, allow_pickle=False)
            for k in z.keys():
                assert z[k].dtype != object, f"{f}:{k} 含 object 数组"
