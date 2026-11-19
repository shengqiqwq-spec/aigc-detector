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
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True,
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
        # 至少要报告 tier1 和 en 两档
        assert "tier1" in out
        assert "en" in out

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
            capture_output=True, text=True, timeout=300, env=env,
        )
        assert r.returncode == 0, f"cp1252 下崩溃: {r.stderr[-400:]}"
        assert "ai_probability" in r.stdout

    def test_info_under_cp1252(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = SRC
        env["PYTHONIOENCODING"] = "cp1252"
        r = subprocess.run([sys.executable, SCRIPT, "--info"],
                           capture_output=True, text=True, timeout=180, env=env)
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
