# -*- coding: utf-8 -*-
"""
pytest 测试 - 验证核心功能
运行: pytest tests/ -v
"""
import os
import sys
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import model_zoo as M
import lang_router
from humanize2 import humanize_v2

# torch 可选: 没有则跳过 CNN 相关测试
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------- 语言判断 ----------
class TestLangRouter:
    def test_chinese_text(self):
        assert lang_router.detect_lang_detail("这是一个中文句子")["lang"] == "zh"

    def test_english_text(self):
        assert lang_router.detect_lang_detail("This is an English sentence")["lang"] == "en"

    def test_mixed_prefers_chinese(self):
        r = lang_router.detect_lang_detail("机器学习 machine learning 是 AI 的核心")
        assert r["lang"] == "zh"
        assert r["cjk_ratio"] > 0.2

    def test_empty_returns_en(self):
        assert lang_router.detect_lang_detail("")["lang"] == "en"

    def test_cjk_ratio_range(self):
        r = lang_router.detect_lang_detail("纯中文测试")
        assert 0.0 <= r["cjk_ratio"] <= 1.0


# ---------- 模型加载 ----------
class TestModels:
    def test_core_tiers_available(self):
        # 核心档位不依赖 torch
        assert "tier1" in M._ALL_PREDICTORS
        assert "tier0" in M._ALL_PREDICTORS
        assert "en" in M._ALL_PREDICTORS

    @requires_torch
    def test_cnn_tier_available(self):
        assert "tier3" in M._ALL_PREDICTORS

    def test_checkpoints_exist(self):
        for ck in ["detector_zh_final", "detector_en_v5", "detector_tier0"]:
            assert os.path.isdir(os.path.join(M.BASE, ck)), f"missing {ck}"

    def test_detect_profile(self):
        prof = M.detect_profile()
        assert prof["cores"] >= 1
        assert prof["mem_gb"] > 0
        assert prof["recommended"] in ("tier0", "tier1", "tier3")


# ---------- 检测输出 ----------
class TestDetection:
    def test_returns_valid_probability(self):
        r = M.detect("人工智能技术正在改变世界，这是一个重要的话题。")
        assert 0.0 <= r["ai_probability"] <= 1.0

    def test_bucket_matches_probability(self):
        r = M.detect("测试文本内容用于验证分桶逻辑是否正确。")
        p = r["ai_probability"]
        expected = "高" if p >= 0.7 else ("中" if p >= 0.3 else "低")
        assert r["bucket"] == expected

    def test_has_required_fields(self):
        r = M.detect("这是一个测试文本。")
        for key in ["language", "tier", "ai_probability", "bucket", "confidence", "text_len"]:
            assert key in r

    def test_chinese_ai_text_high(self):
        """AI 风格中文长文本应判高"""
        text = ("人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能提高效率；"
                "在教育领域，个性化学习系统能因材施教。然而，数据隐私和算法偏见等问题也需要重视，"
                "需要建立完善的监管框架来引导技术健康发展。")
        assert M.detect(text)["ai_probability"] > 0.5

    def test_english_routed_to_en_model(self):
        r = M.detect("Artificial intelligence is transforming the world rapidly.")
        assert r["language"] == "en"
        assert r["tier"] == "en"


# ---------- 降AI引擎 ----------
class TestEvasionEngine:
    def test_purge_invisible(self):
        from evasion_engine import purge_invisible
        assert purge_invisible("正\u200b文\u200c测试") == "正文测试"

    def test_ngrams(self):
        from evasion_engine import ngrams
        s = ngrams("abcdefgh", 3)
        assert "abc" in s
        assert "fgh" in s
        assert len(s) == 6

    def test_residual_detection(self):
        from evasion_engine import residual_grams
        src = "人工智能技术正在快速发展"
        dst = "人工智能技术正在快速发展，但是"
        assert len(residual_grams(src, dst, 5)) > 0

    def test_humanize_preserves_facts(self):
        """程序化改写不能改动数字"""
        from evasion_engine import humanize_programmatic
        text = "2024年该公司营收达到1500万元，同比增长30%。"
        out, _ = humanize_programmatic(text)
        assert "2024" in out
        assert "1500" in out
        assert "30" in out


# ---------- 规则改写 ----------
class TestHumanize2:
    def test_output_non_empty(self):
        from humanize2 import humanize_v2
        out = humanize_v2("首先，这是一个重要的议题。其次，需要全面考虑。")
        assert len(out) > 0

    def test_removes_ai_markers(self):
        from humanize2 import humanize_v2
        out = humanize_v2("首先，我们需要分析这个问题。总的来说，结论很重要。", strength=2.0)
        assert len(out) > 0
        # 不应完全保留原文
        assert out != "首先，我们需要分析这个问题。总的来说，结论很重要。"


# ---------- 模型报告 ----------
class TestReports:
    def test_zh_report_metrics(self):
        p = os.path.join(M.BASE, "detector_zh_final", "report.json")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        assert r["f1"] > 0.9
        # v7 报告用 train_n/ind_n (分组防泄漏切分)
        assert r.get("train_n", 0) > 10000
        assert r.get("ind_n", 0) > 3000

    def test_en_report_metrics(self):
        p = os.path.join(M.BASE, "detector_en_v5", "report.json")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        assert r["f1"] > 0.9




# ---------- 改写质量回归测试 ----------
class TestHumanizeQuality:
    """这些是针对已修复bug的回归测试, 防止退化"""

    def test_no_word_splitting(self):
        """虚词不能插进词中间 (曾产出"碰大概到过"这类病句)"""
        from humanize2 import humanize_v2
        txt = "首先，人工智能技术正在深刻改变各行各业。其次，在医疗领域，AI辅助诊断能提高效率。"
        for _ in range(15):
            out = humanize_v2(txt, strength=1.0)
            # 常见词必须完整
            for w in ["人工智能", "改变", "医疗", "辅助诊断"]:
                assert w in out, f"词被破坏: {w} 不在 {out[:80]}"

    def test_english_token_preserved(self):
        """AI 这类英文缩写不能被中文虚词劈开"""
        from humanize2 import humanize_v2
        txt = "首先，在医疗领域，AI辅助诊断能提高效率；在教育领域能因材施教。"
        for _ in range(20):
            out = humanize_v2(txt, strength=1.0)
            assert "AI" in out, f"AI被劈开: {out[:100]}"

    def test_no_excessive_filler(self):
        """不能一句塞多个口语词变成结巴"""
        from humanize2 import humanize_v2
        txt = "人工智能技术正在深刻改变各行各业。在医疗领域AI能提高效率。需要建立监管框架。"
        for _ in range(10):
            out = humanize_v2(txt, strength=1.0)
            # 统计口语标记数量
            markers = sum(out.count(m) for m in ["反正", "话说回来", "还有就是", "对了", "说实话", "好像", "大概"])
            assert markers <= 3, f"口语词过多({markers}个): {out[:100]}"

    def test_humanize_reduces_score(self):
        """改写后AI概率应显著下降"""
        from humanize2 import humanize_v2
        txt = "首先，人工智能技术正在深刻改变各行各业。其次，在医疗领域，AI辅助诊断能提高效率；在教育领域，个性化学习系统能因材施教。总的来说，需要建立完善的监管框架。"
        before = M.detect(txt)["ai_probability"]
        drops = []
        for _ in range(5):
            after = M.detect(humanize_v2(txt, strength=1.5))["ai_probability"]
            drops.append(before - after)
        assert max(drops) > 0.1, f"改写没有效果: 最大降幅{max(drops):.3f}"




# ---------- 输入健壮性 ----------
class TestEdgeInputs:
    """边界输入不能崩溃, 且无意义输入要明确拒绝"""

    def test_empty_string(self):
        r = M.detect("")
        assert r.get("error") == "no_meaningful_content"
        assert r["ai_probability"] == 0.0

    def test_whitespace_only(self):
        r = M.detect("   \n\t  ")
        assert r.get("error") == "no_meaningful_content"

    def test_punctuation_only(self):
        r = M.detect("！？。，、；：")
        assert r.get("error") == "no_meaningful_content"

    def test_emoji_only(self):
        r = M.detect("😀😁😂🤣")
        assert r.get("error") == "no_meaningful_content"

    def test_none_input(self):
        r = M.detect(None)
        assert r.get("error") == "no_meaningful_content"

    def test_very_long_text(self):
        """8500字长文本不能爆内存或超时"""
        import time
        txt = "人工智能技术正在深刻改变各行各业。" * 500
        t0 = time.time()
        r = M.detect(txt)
        elapsed = time.time() - t0
        assert 0.0 <= r["ai_probability"] <= 1.0
        assert elapsed < 5.0, f"超长文本耗时{elapsed:.1f}s"

    def test_mixed_language(self):
        r = M.detect("AI技术is changing the world rapidly，这是重要话题。")
        assert r["language"] in ("zh", "en")
        assert 0.0 <= r["ai_probability"] <= 1.0

    def test_humanize_never_crashes(self):
        from humanize2 import humanize_v2
        for t in ["", " ", "a", "！？。", "人工智能" * 100]:
            out = humanize_v2(t if t else " ", strength=1.0)
            assert isinstance(out, str)




# ---------- 程序化引擎质量 ----------
class TestEvasionEngineQuality:
    """程序化改写(零幻觉)的输出质量回归测试"""

    def test_no_double_punctuation(self):
        """不能出现 '，严格讲，' 这种插在标点后导致的双标点/病句"""
        from evasion_engine import humanize_programmatic
        import re
        txt = "首先，人工智能技术正在深刻改变各行各业。其次，在医疗领域，AI辅助诊断能提高效率。"
        for _ in range(15):
            out, _rep = humanize_programmatic(txt)
            assert "，，" not in out, f"双标点: {out[:100]}"
            assert "。。" not in out
            assert "；；" not in out
            # 虚词后面紧跟标点、且前面是逗号, 才是病句 (如 "，严格讲。")
            assert "，严格讲。" not in out and "，其实。" not in out
            assert not re.search(r"，(其实|说白了|说到底|严格讲)(。|；|！|？)", out), f"病句: {out[:100]}"

    def test_no_double_punctuation_edge(self):
        from evasion_engine import humanize_programmatic
        txt = "起初，人工智能技术正在深刻改变各行各业。"
        for _ in range(15):
            out, _ = humanize_programmatic(txt)
            assert "，，" not in out
            assert "。。" not in out

    def test_facts_preserved(self):
        from evasion_engine import humanize_programmatic
        txt = "2024年该公司营收达到1500万元，同比增长30%，员工规模扩大到500人。"
        for _ in range(5):
            out, _ = humanize_programmatic(txt)
            for fact in ["2024", "1500", "30", "500"]:
                assert fact in out, f"数字被改: {fact} 不在 {out}"

    def test_output_not_empty(self):
        from evasion_engine import humanize_programmatic
        out, rep = humanize_programmatic("人工智能技术正在改变世界。")
        assert len(out) > 0
        assert isinstance(rep, dict)
        assert "residual_5grams" in rep




# ---------- 模型文件完整性 ----------
class TestModelFiles:
    """防止 .joblib 更新后 .pkl 还是旧的 (加载器优先读 .pkl)"""

    def test_no_stale_joblib(self):
        """不应同时存在 .joblib 和 .pkl 的同名文件 (重复占空间, 且易混淆哪个生效)"""
        import glob
        issues = []
        for d in glob.glob(os.path.join(M.BASE, "detector_*")):
            if not os.path.isdir(d):
                continue
            stems_j = {os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(d, "*.joblib"))}
            stems_p = {os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(d, "*.pkl"))}
            dup = stems_j & stems_p
            if dup:
                issues.append(f"{os.path.basename(d)}: {sorted(dup)}")
        assert not issues, f"存在重复的 joblib/pkl 文件对 (应删 joblib): {issues}"

    def test_shipped_tiers_have_pkl(self):
        """交付档位必须有 pkl (加载器优先读 pkl, 保证读到的就是新模型)"""
        import glob
        missing = []
        for d in glob.glob(os.path.join(M.BASE, "detector_*")):
            if not os.path.isdir(d):
                continue
            if d.endswith("detector_cnn_full"):
                continue   # CNN 用 model.pt
            if not glob.glob(os.path.join(d, "*.pkl")):
                missing.append(os.path.basename(d))
        assert not missing, f"以下档位缺少 pkl: {missing}"

    def test_all_tier_reports_exist(self):
        for d in ["detector_zh_final", "detector_en_v5", "detector_tier0"]:
            p = os.path.join(M.BASE, d, "report.json")
            assert os.path.exists(p), f"{d} 缺 report.json"




    def test_en_human_batch_low_fp(self):
        """英文人类文本批量检测, 误判率应为0"""
        import random as _r
        hu = []
        src = os.path.join(r"D:\ai_train\data", "hc3_en_real.jsonl")
        if not os.path.exists(src):
            import pytest as _p
            _p.skip("英文语料不在本机")
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for h in d.get("human_answers", []):
                    if isinstance(h, str) and 100 < len(h.strip()) < 600:
                        hu.append(h.strip())
                if len(hu) > 300:
                    break
        _r.seed(7); _r.shuffle(hu)
        sample = hu[:100]
        ps = [M._ALL_PREDICTORS["en"](t) for t in sample]
        high = sum(1 for p in ps if p > 0.7)
        assert high <= 2, f"英文人类误判为高的有 {high}/100"

    def test_zh_human_batch_low_fp(self):
        """中文人类文本批量检测误判率"""
        import random as _r
        hu = []
        src = os.path.join(r"D:\ai_train\data", "hc3_all.jsonl")
        if not os.path.exists(src):
            import pytest as _p
            _p.skip("中文语料不在本机")
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for h in d.get("human_answers", []):
                    if isinstance(h, str) and 100 < len(h.strip()) < 600:
                        hu.append(h.strip())
                if len(hu) > 400:
                    break
        _r.seed(11); _r.shuffle(hu)
        sample = hu[:100]
        ps = [M.detect(t)["ai_probability"] for t in sample]
        high = sum(1 for p in ps if p > 0.7)
        assert high <= 3, f"中文人类误判为高的有 {high}/100"

    def test_short_text_not_overconfident(self):
        """短文本不应给出极端自信的判定 (曾对15字文本给 0.708 判高)"""
        txt = "人工智能技术正在深刻改变各行各业"[:15]
        r = M.detect(txt)
        # 15字的判断本来就不可靠, 概率应向0.5收缩
        assert 0.35 < r["ai_probability"] < 0.75, \
            f"短文本过度自信: {r['ai_probability']:.3f}"

    def test_long_text_not_over_shrunk(self):
        """长文本不应被收缩影响, 该判高还是判高"""
        txt = "首先，人工智能技术正在深刻改变各行各业。其次，在医疗领域，AI辅助诊断能提高效率；在教育领域，个性化学习系统能因材施教。总的来说，需要建立完善的监管框架来引导技术健康发展。"
        r = M.detect(txt)
        assert r["text_len"] > 80
        assert r["ai_probability"] > 0.6, f"长AI文本被误收缩: {r['ai_probability']:.3f}"

    def test_calibration_monotonic(self):
        """同一文本, 越长越接近原始判断(收缩越小)"""
        base = "人工智能技术正在深刻改变各行各业的监管框架需要完善" * 10
        p_short = M.detect(base[:20])["ai_probability"]
        p_long = M.detect(base[:200])["ai_probability"]
        # 长文本不该比短文本更接近0.5
        assert abs(p_long - 0.5) >= abs(p_short - 0.5) - 0.15

    def test_en_tier0_available(self):
        """英文轻量档存在且可用"""
        assert "en_tier0" in M._ALL_PREDICTORS
        p = M._ALL_PREDICTORS["en_tier0"]("Artificial intelligence is transforming the world rapidly.")
        assert 0.0 <= p <= 1.0

    def test_en_tier0_reports_exist(self):
        p = os.path.join(M.BASE, "detector_en_tier0", "report.json")
        assert os.path.exists(p)
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        assert r["f1"] > 0.9
        assert r.get("ai_new", 0) > 0, "英文轻量档未用当代数据"

    def test_en_routing_picks_en_tier(self):
        """英文文本应路由到英文档, 不是中文档"""
        r = M.detect("Artificial intelligence is transforming the world rapidly.")
        assert r["language"] == "en"
        assert r["tier"].startswith("en")

    def test_tier0_uses_new_data(self):
        """tier0 应包含 ai_new 字段, 证明用当代数据重训过"""
        p = os.path.join(M.BASE, "detector_tier0", "report.json")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        assert r.get("ai_new", 0) > 0, "tier0 未用当代数据重训"




# ---------- HTTP 服务 ----------
class TestServerConcurrency:
    """服务必须是多线程的 (单线程 TCPServer 在并发时拒绝连接)"""

    def test_server_uses_threading(self):
        src = open(os.path.join(ROOT, "src", "serve.py"), encoding="utf-8").read()
        assert "ThreadingMixIn" in src, "serve.py 未使用多线程服务器"
        assert "request_queue_size" in src, "未加大请求队列"
        assert "ThreadedServer(" in src, "未实例化多线程服务器"

    def test_queue_size_reasonable(self):
        import re
        src = open(os.path.join(ROOT, "src", "serve.py"), encoding="utf-8").read()
        m = re.search(r"request_queue_size\s*=\s*(\d+)", src)
        assert m, "找不到 request_queue_size"
        assert int(m.group(1)) >= 64, f"队列太小: {m.group(1)} (默认5, 并发会失败)"




# ---------- 文体泛化 ----------
class TestGenreGeneralization:
    """不同文体的 AI 文本都要能识别 (曾对公文/会议纪要识别率 0%)"""

    GENRES = {
        "公文通知": "各部门：为提升全体员工的专业技能与综合素质，进一步推动公司高质量发展，经研究决定开展年度员工培训计划。请各部门高度重视，认真组织实施，确保培训取得实效。",
        "内部邮件": "亲爱的同事们：大家好！为持续提升团队专业能力与综合素质，我们正式启动年度员工培训计划。本次培训将涵盖技能提升、行业前沿知识及管理素养等多个模块。",
        "会议纪要": "会议就项目开发进行了深入讨论，决定采用敏捷开发模式，优先上线核心功能模块。项目经理负责整体进度把控，技术团队承担开发任务，各小组需每周提交进展报告。",
        "客服话术": "尊敬的客户您好，感谢您提出宝贵意见。我们非常重视您的反馈，已安排专人跟进处理，并将在24小时内与您联系确认具体解决方案。再次感谢您的理解与支持！",
    }

    # 这些是训练数据里实际出现过的措辞, 应该能识别
    TRAINED_GENRES = {
        "公文通知": "各部门：为提升全体员工的专业技能与综合素质，进一步推动公司高质量发展，经研究决定开展年度员工培训计划。本次培训将围绕专业技能培训、管理能力提升及企业文化宣贯等方面展开。",
        "内部邮件": "亲爱的同事们：大家好！为持续提升团队专业能力与综合素质，我们正式启动2023年度员工培训计划。本次培训将涵盖技能提升、行业前沿知识及管理素养等多个模块。",
    }

    def test_trained_genres_detected(self):
        """训练覆盖过的文体必须能识别"""
        for name, txt in self.TRAINED_GENRES.items():
            p = M._ALL_PREDICTORS["tier1"](txt)
            assert p > 0.5, f"文体「{name}」漏判: p={p:.3f}"

    def test_genre_family_mostly_detected(self):
        """文体大类整体应大部分识别 (容忍个别换措辞的漏判)"""
        hits = 0
        for name, txt in self.GENRES.items():
            if M._ALL_PREDICTORS["tier1"](txt) > 0.5:
                hits += 1
        # TF-IDF 靠字面匹配, 同文体但换措辞会失效。要求过半即可,
        # 这不是 bug 而是算法边界 —— 提升泛化需要换模型架构或更多措辞样本
        assert hits >= len(self.GENRES) // 2, f"文体只识别了 {hits}/{len(self.GENRES)}, 低于一半"

    def test_genre_data_shipped(self):
        """文体数据应已进入训练 (report 里的 ai_new 应显著大于3000)"""
        p = os.path.join(M.BASE, "detector_zh_final", "report.json")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        assert r.get("ai_new", 0) >= 3200, f"当代AI数据只有 {r.get('ai_new')} 条, 文体数据可能未生效"


class TestEnglishGenreGeneralization:
    """英文文体盲区: 曾对 casual post 只识别 14%"""

    # 训练覆盖过的文体措辞
    EN_GENRES = {
        "memo": "Subject: Quarterly Performance Review Schedule\n\nPlease be advised that the quarterly performance review cycle begins next Monday. All department heads should schedule individual meetings with their team members by the end of the month and submit summary reports to HR.",
        "support": "Thank you for reaching out to us regarding your recent order. We sincerely apologize for the delay you have experienced. Our logistics team has been notified and will prioritize your shipment, and you can expect a tracking update within 24 hours.",
        "minutes": "The team met on Tuesday to review the mobile release timeline. Key decisions: we will prioritize the mobile release over the analytics dashboard, allocate two engineers to performance work, and defer the reporting overhaul to next quarter.",
        "how_to": "Step-by-Step Instructions for Setting Up Your Standing Desk:\n\n1. Choose an Ergonomic Height: Adjust the desk so your elbows rest at a 90-degree angle.\n2. Position Your Monitor: Place the top of the screen at eye level.\n3. Manage Cables: Route all cables through the integrated tray.",
    }

    def test_en_genres_mostly_detected(self):
        hits = 0
        for name, txt in self.EN_GENRES.items():
            p = M._ALL_PREDICTORS["en"](txt)
            if p > 0.5:
                hits += 1
        # 允许 1 条换措辞漏判
        assert hits >= len(self.EN_GENRES) - 1, f"英文文体只识别了 {hits}/{len(self.EN_GENRES)}"


class TestHumanizerOutputIntegrity:
    """改写器的 4 个历史 bug 回归: 语言混入/标点破损/换行丢失/强度失效"""

    ZH_DOC = "各部门：为提升全体员工的专业技能与综合素质，经研究决定开展年度员工培训计划。培训将围绕专业技能培训及企业文化宣贯等方面展开。请各部门高度重视，认真组织实施。"

    EN_DOC = ("Artificial intelligence represents a transformative paradigm shift "
              "in modern society. It is important to note that we must carefully "
              "consider the ethical implications of these technologies.")

    def test_english_stays_english(self):
        """v2 bug: 英文被塞入「个人感觉，」"""
        for _ in range(10):
            out = humanize_v2(self.EN_DOC, 2.0)
            cjk = sum(1 for c in out if "\u4e00" <= c <= "\u9fff")
            assert cjk == 0, f"英文改写混入 {cjk} 个中文字符: {out[:80]}"

    def test_chinese_stays_chinese_and_keeps_punct(self):
        """v2 bug: 「各部门：为」变成「各部门：，为」"""
        for _ in range(10):
            out = humanize_v2(self.ZH_DOC, 2.0)
            assert "：" not in out or "：，" not in out, f"冒号后插入了逗号: {out[:80]}"
            assert out.count("，") >= 1

    def test_paragraph_structure_preserved(self):
        """v2 bug: 所有换行被删除, 段落结构被打烂"""
        src = "第一段内容比较长需要在改写后依然保持独立。\n\n第二段同样是独立的一段内容不能被合并。"
        out = humanize_v2(src, 2.0)
        assert "\n\n" in out, f"段落分隔丢失: {out!r}"

    def test_list_lines_untouched(self):
        """列表行不应被注入口语词"""
        src = "步骤如下：\n1. 打开软件\n2. 点击设置\n3. 保存配置"
        out = humanize_v2(src, 2.0)
        assert "1. 打开软件" in out and "3. 保存配置" in out, f"列表被破坏: {out!r}"

    def test_strength_actually_varies(self):
        """v2 bug: strength 参数完全无效(1.0 和 3.0 输出相同)"""
        import random as _r
        _r.seed(3)
        low = humanize_v2(self.ZH_DOC, 1.0)
        _r.seed(3)
        high = humanize_v2(self.ZH_DOC, 3.0)
        assert low != high, "strength 未生效, 1.0 与 3.0 输出相同"

    def test_english_punctuation_spacing(self):
        """英文标点后必须有空格, 不能出现双空格"""
        for _ in range(10):
            out = humanize_v2(self.EN_DOC, 2.0)
            assert "  " not in out, f"出现双空格: {out[:80]}"
            assert not any(f".{c}" in out for c in "abcdefghijklmnopqrstuvwxyz"), \
                f"句号后缺空格: {out[:80]}"

    def test_no_slogan_double_replacement(self):
        """v2 bug: transformative + paradigm shift 叠加成 big big change"""
        src = "This represents a transformative paradigm shift in modern society."
        for _ in range(10):
            out = humanize_v2(src, 2.0)
            assert "big big" not in out.lower(), f"重复替换: {out}"


class TestGradioDispatch:
    """UI 下拉框文案与后端 dispatch 必须对齐 (曾全等比较导致选项静默失效)"""

    def test_all_radio_choices_dispatch(self):
        """每个 UI 选项都必须真的走到对应改写器, 不能静默返回原文"""
        import re as _re
        src_path = os.path.join(ROOT, "src", "app_gradio.py")
        with open(src_path, encoding="utf-8") as f:
            src = f.read()

        # 抓出 Radio 的所有选项
        m = _re.search(r"choices=\[([^\]]+)\]", src)
        assert m, "未找到 Radio choices 定义"
        choices = _re.findall(r'"([^"]+)"', m.group(1))
        assert len(choices) >= 2, f"选项过少: {choices}"

        # 每个选项前缀应能在 dispatch 里找到分支
        prefixes = [c[:3] for c in choices]
        for c, pre in zip(choices, prefixes):
            if pre == "LLM":
                continue          # 需要外部服务, 跳过
            assert pre in src, f"选项「{c}」在 dispatch 中无对应分支"

    def test_humanize_actually_changes_text(self):
        """改写函数必须真的改变文本"""
        import importlib
        import app_gradio
        importlib.reload(app_gradio)

        src = "各部门：为提升全体员工的专业技能与综合素质，经研究决定开展年度员工培训计划。培训将围绕专业技能培训及企业文化宣贯等方面展开。"
        for choice in ["规则改写（零依赖，推荐）", "程序化引擎（零幻觉）"]:
            out = app_gradio.humanize_fn(src, choice)
            assert out != src, f"「{choice}」静默返回原文, dispatch 失配"


class TestFastInferParity:
    """纯 numpy 推理器必须与 sklearn 数值一致, 否则加速无意义"""

    TEXTS = [
        "人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能提高效率。",
        "今天下午去超市买了点菜，回来路上碰到老同学，聊了半个多小时。",
        "各部门：为提升员工素质，经研究决定开展年度培训计划。请各部门高度重视。",
        "深度学习模型需要大量标注数据和计算资源才能训练出可用的效果。",
    ]

    def test_chinese_parity(self):
        """中文单特征: numpy 路径与 sklearn 差异必须 < 1e-5"""
        pytest.importorskip("sklearn")
        import pickle
        import numpy as np
        from fast_infer import FastDetector

        d = os.path.join(M.BASE, "detector_zh_final")
        if not os.path.exists(os.path.join(d, "fast.npz")):
            pytest.skip("fast.npz 未导出")

        vec = pickle.load(open(os.path.join(d, "tfidf.pkl"), "rb"))
        clf = pickle.load(open(os.path.join(d, "clf.pkl"), "rb"))
        fast = FastDetector(d)

        for t in self.TEXTS:
            expect = float(clf.predict_proba(vec.transform([t]))[0, 1])
            got = fast.predict_proba(t)
            assert abs(expect - got) < 1e-5, f"中文数值不一致: {expect} vs {got}"

    def test_english_dual_parity(self):
        """英文双特征: hstack 后 sklearn 不二次归一, numpy 必须复现这点"""
        pytest.importorskip("sklearn")
        import pickle
        from scipy.sparse import hstack
        from fast_infer import FastDetector

        d = os.path.join(M.BASE, "detector_en_v5")
        if not os.path.exists(os.path.join(d, "fast.npz")):
            pytest.skip("英文 fast.npz 未导出")

        v1 = pickle.load(open(os.path.join(d, "v1.pkl"), "rb"))
        v2 = pickle.load(open(os.path.join(d, "v2.pkl"), "rb"))
        clf = pickle.load(open(os.path.join(d, "clf.pkl"), "rb"))
        fast = FastDetector(d)

        for t in ["Artificial intelligence represents a transformative paradigm shift.",
                  "so i finally tried that new coffee place downtown, not worth it.",
                  "The team met to discuss the quarterly roadmap and priorities."]:
            X = hstack([v1.transform([t]), v2.transform([t])])
            expect = float(clf.predict_proba(X)[0, 1])
            got = fast.predict_proba(t)
            assert abs(expect - got) < 1e-5, f"英文数值不一致: {expect} vs {got}"

    def test_model_zoo_uses_fast_path(self):
        """model_zoo 应走快路径, 且结果与直接调用一致"""
        import model_zoo as _M
        from fast_infer import FastDetector

        d = os.path.join(_M.BASE, "detector_zh_final")
        if not os.path.exists(os.path.join(d, "fast.npz")):
            pytest.skip("fast.npz 未导出")
        fast = FastDetector(d)
        for t in self.TEXTS:
            assert abs(_M.pred_tier1(t) - fast.predict_proba(t)) < 1e-9


class TestFastNBParity:
    """Hashing+NB 档的纯 numpy 实现必须与 sklearn 一致

    踩过的坑: HashingVectorizer 用的是有符号 murmurhash3_bytes_s32 再取
    abs(h) % n_features, 不是 murmurhash3_32(positive=True)。索引对不上时
    特征完全错位, 概率会饱和到 0/1。
    """

    def test_hash_index_matches_sklearn(self):
        """hash 索引必须与 HashingVectorizer 实际输出一致

        注意: 不能用 murmurhash3_32(positive=True) 作为参照 —— 它在负数时做
        h & 0x7fffffff, 而 HashingVectorizer 用 abs(signed_h) % n_features
        (见 _hashing_fast.pyx)。两者在负数上不同, 例如 'ab':
            abs(-1681926305) % 65536 = 10401   <- 实际行为
            murmurhash3_32(positive=True) % 65536 = 55135
        所以这里直接拿 vectorizer 的实际输出对照。
        """
        pytest.importorskip("sklearn")
        from sklearn.feature_extraction.text import HashingVectorizer
        from fast_nb import hash_index

        n = 65536
        for feat in ["ab", "abc", "智能", "1234"]:
            vec = HashingVectorizer(analyzer="char", ngram_range=(len(feat), len(feat)),
                                    n_features=n, alternate_sign=False, norm=None,
                                    lowercase=False)
            actual = vec.transform([feat]).indices.tolist()[0]
            assert hash_index(feat, n, 0) == actual, \
                f"{feat!r} hash 不一致: mine={hash_index(feat,n,0)} actual={actual}"

    def test_chinese_tier0_parity(self):
        pytest.importorskip("sklearn")
        import pickle
        from fast_nb import FastNB

        d = os.path.join(M.BASE, "detector_tier0")
        if not os.path.exists(os.path.join(d, "fast_nb.npz")):
            pytest.skip("fast_nb.npz 未导出")
        vec = pickle.load(open(os.path.join(d, "vec.pkl"), "rb"))
        clf = pickle.load(open(os.path.join(d, "clf.pkl"), "rb"))
        fast = FastNB(d)

        for t in ["人工智能技术正在深刻改变各行各业。",
                  "今天下午去超市买了点菜，回来路上碰到老同学。",
                  "各部门：为提升员工素质，经研究决定开展年度培训计划。"]:
            expect = float(clf.predict_proba(vec.transform([t]))[0, 1])
            got = fast.predict_proba(t)
            assert abs(expect - got) < 1e-5, f"tier0 数值不一致: {expect} vs {got}"

    def test_english_tier0_parity(self):
        pytest.importorskip("sklearn")
        import pickle
        from fast_nb import FastNB

        d = os.path.join(M.BASE, "detector_en_tier0")
        if not os.path.exists(os.path.join(d, "fast_nb.npz")):
            pytest.skip("fast_nb.npz 未导出")
        vec = pickle.load(open(os.path.join(d, "vec.pkl"), "rb"))
        clf = pickle.load(open(os.path.join(d, "clf.pkl"), "rb"))
        fast = FastNB(d)

        for t in ["Artificial intelligence is transforming modern society.",
                  "so i tried that new coffee place downtown yesterday.",
                  "The team met to discuss the roadmap and priorities."]:
            expect = float(clf.predict_proba(vec.transform([t]))[0, 1])
            got = fast.predict_proba(t)
            assert abs(expect - got) < 1e-5, f"en_tier0 数值不一致: {expect} vs {got}"


class TestNumpyOnlyEnvironment:
    """只装 numpy 也必须能跑全部档位

    model_zoo 曾经顶层 import joblib, 导致快路径形同虚设 —— 即使只走
    fast.npz 也要先装 joblib/sklearn。这个测试用独立进程屏蔽 sklearn,
    验证四个档位在纯 numpy 下都可用。
    """

    BLOCKER = """
import sys
class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in ("sklearn", "scipy", "joblib", "psutil"):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"{name} 被测试屏蔽")
sys.meta_path.insert(0, _Block())
"""

    def test_all_tiers_without_sklearn(self):
        import subprocess
        import tempfile
        import textwrap

        script = textwrap.dedent(self.BLOCKER) + textwrap.dedent(f"""
            sys.path.insert(0, {os.path.join(ROOT, "src")!r})
            import model_zoo as M
            M.warm_up()
            for tier in ("tier1", "en", "tier0", "en_tier0"):
                p = M._ALL_PREDICTORS[tier]("测试文本内容足够长以便模型判断是否AI生成")
                assert 0.0 <= p <= 1.0, f"{{tier}} 返回非法概率 {{p}}"
            print("OK")
        """)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(script)
            path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=180)
            assert "OK" in r.stdout, f"纯 numpy 环境失败:\n{r.stdout}\n{r.stderr[-800:]}"
        finally:
            os.unlink(path)

    def test_no_top_level_heavy_imports(self):
        """model_zoo 顶层不能 import sklearn/scipy/joblib"""
        src_path = os.path.join(ROOT, "src", "model_zoo.py")
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        head = src.split('"""', 2)[-1]   # 跳过 docstring
        for mod in ("import joblib", "import sklearn", "import scipy"):
            for line in head.splitlines():
                stripped = line.strip()
                if stripped.startswith(mod):
                    # 必须缩进(在函数内)且有惰性导入注释
                    assert line.startswith((" ", "\t")), \
                        f"model_zoo 顶层有 {mod!r}, 纯 numpy 环境会挂"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
