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
        """如果同目录同时有 .joblib 和 .pkl, 它们应该都存在, 且不能只有 joblib"""
        import glob
        ckpt_root = M.BASE
        issues = []
        for d in glob.glob(os.path.join(ckpt_root, "detector_*")):
            if not os.path.isdir(d):
                continue
            joblibs = glob.glob(os.path.join(d, "*.joblib"))
            pkls = glob.glob(os.path.join(d, "*.pkl"))
            if joblibs and not pkls:
                # 只有 joblib, 加载器会走 fallback, 可以接受但应提示
                issues.append(f"{os.path.basename(d)}: only joblib")
        # 不允许"只有joblib"的情况出现在交付档位里
        assert not issues, f"以下档位缺少 pkl (加载器会 fallback): {issues}"

    def test_all_tier_reports_exist(self):
        for d in ["detector_zh_final", "detector_en_v5", "detector_tier0"]:
            p = os.path.join(M.BASE, d, "report.json")
            assert os.path.exists(p), f"{d} 缺 report.json"


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
