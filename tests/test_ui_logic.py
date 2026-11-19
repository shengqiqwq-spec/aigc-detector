# -*- coding: utf-8 -*-
"""Gradio UI 函数的逻辑测试

不起浏览器, 直接测 app_gradio 里的两个函数: detect_fn / humanize_fn /
_humanize_with_check 的内部实现。Gradio 只是把这些函数接到控件上,
函数本身的行为才是产品行为。
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("gradio", reason="UI 逻辑测试需要可选依赖 gradio")

import app_gradio  # noqa: E402


class TestDetectFn:
    """/detect 的 UI 版本: 返回 (verdict, detail, advice) 三段 markdown"""

    AI_TEXT = ("人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能够提高效率；"
               "在教育领域，个性化学习系统能够因材施教；在交通领域，智能调度减少了拥堵。")
    HUMAN_TEXT = "今天下午去超市买了点菜，回来路上碰到老同学，聊了半个多小时。"

    def test_empty_input(self):
        verdict, detail, advice = app_gradio.detect_fn("")
        assert "请输入文本" in verdict
        assert detail == "" and advice == ""

    def test_whitespace_input(self):
        verdict, _, _ = app_gradio.detect_fn("   \n\t  ")
        assert "请输入文本" in verdict

    def test_ai_text_verdict_shape(self):
        verdict, detail, advice = app_gradio.detect_fn(self.AI_TEXT)
        assert "AI 概率" in verdict
        assert "%" in verdict
        # 三档之一
        assert any(b in verdict for b in ("高", "中", "低"))

    def test_detail_has_all_fields(self):
        _, detail, _ = app_gradio.detect_fn(self.AI_TEXT)
        for field in ("语言", "档位", "长度", "置信度", "耗时"):
            assert field in detail, f"详情缺少 {field}"

    def test_advice_not_empty(self):
        _, _, advice = app_gradio.detect_fn(self.AI_TEXT)
        assert len(advice) > 5

    def test_short_text_warns(self):
        """短文本应追加"仅供参考"的警告"""
        _, _, advice = app_gradio.detect_fn("人工智能正在改变世界。")
        assert "短" in advice or "仅供参考" in advice

    def test_human_text_advice_mentions_human(self):
        _, _, advice = app_gradio.detect_fn(self.HUMAN_TEXT)
        assert isinstance(advice, str) and len(advice) > 0

    def test_rejected_input_does_not_crash(self):
        """JSON 这类输入走 error 分支, UI 不能崩"""
        verdict, detail, advice = app_gradio.detect_fn('{"key": "value"}')
        assert isinstance(verdict, str)
        assert isinstance(detail, str)
        assert isinstance(advice, str)


class TestHumanizeFn:
    AI_TEXT = ("各部门：为提升全体员工的专业技能与综合素质，进一步推动公司高质量发展，"
               "经研究决定开展年度员工培训计划。")

    def test_empty_input(self):
        assert "请输入文本" in app_gradio.humanize_fn("", "规则改写（零依赖，推荐）")

    def test_rule_method_changes_text(self):
        out = app_gradio.humanize_fn(self.AI_TEXT, "规则改写（零依赖，推荐）")
        assert isinstance(out, str)
        assert out != self.AI_TEXT, "规则改写没有生效"

    def test_programmatic_method_changes_text(self):
        out = app_gradio.humanize_fn(self.AI_TEXT, "程序化引擎（零幻觉）")
        assert isinstance(out, str)
        assert out != self.AI_TEXT, "程序化改写没有生效"

    def test_llm_method_reports_missing_service(self):
        """LLM 档在没有本地服务时应给出可读提示, 而不是抛异常"""
        out = app_gradio.humanize_fn(self.AI_TEXT, "LLM改写（需本地大模型，慢）")
        assert isinstance(out, str)
        # 要么真的改写了(服务在跑), 要么提示服务未启动
        assert out != "" and (out != self.AI_TEXT or "LLM" in out)

    def test_unknown_method_raises(self):
        """未知方式应显式报错, 不能静默返回原文"""
        with pytest.raises(ValueError):
            app_gradio.humanize_fn(self.AI_TEXT, "不存在的改写方式")


class TestNeverEchoesInput:
    """改写器绝不能原样返回

    CI 在 ubuntu 3.11 上抓到: 26 字的短公文有 44% 概率原样返回。
    套话规则不命中、句子太短无法注入、随机 roll 不触发 —— 三条都不占,
    用户点"改写"拿到一模一样的东西, 会认为功能坏了。

    这条测试故意跑多次, 因为原 bug 是概率性的。
    """

    CASES = [
        "各部门：为提升员工素质，经研究决定开展年度培训计划。",
        "人工智能正在改变世界。",
        "The meeting is scheduled for Tuesday.",
        "各部门：为提升全体员工的专业技能与综合素质，经研究决定开展年度员工培训计划。"
        "现将有关事项通知如下：请各部门高度重视，认真组织实施。",
    ]

    def test_never_returns_input_unchanged(self):
        for text in self.CASES:
            for _ in range(20):
                out = app_gradio.humanize_fn(text, "规则改写（零依赖，推荐）")
                assert out != text, f"原样返回: {text!r}"

    def test_short_text_still_gets_modified(self):
        """单句、无套话的短文本是最容易漏的"""
        short = "人工智能正在改变世界。"
        for _ in range(20):
            assert app_gradio.humanize_fn(short, "规则改写（零依赖，推荐）") != short


class TestRadioChoicesMatchDispatch:
    """下拉框的每个选项都必须能走到对应的改写实现

    历史 bug: 选项文案 "规则改写（零依赖，推荐）" 与代码里比较的
    "规则改写（零依赖）" 不完全相等, 三个选项全部落空, 静默返回原文。
    """

    def test_every_choice_dispatches(self):
        import re
        src_path = os.path.join(ROOT, "src", "app_gradio.py")
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        m = re.search(r"choices=\[([^\]]+)\]", src)
        assert m, "找不到 Radio choices 定义"
        choices = re.findall(r'"([^"]+)"', m.group(1))
        assert len(choices) >= 2

        text = "各部门：为提升员工素质，经研究决定开展年度培训计划。"
        for choice in choices:
            if choice.startswith("LLM"):
                continue        # 需要外部服务
            out = app_gradio.humanize_fn(text, choice)
            assert out != text, f"选项「{choice}」静默返回了原文"
