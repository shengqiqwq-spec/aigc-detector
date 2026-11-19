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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
