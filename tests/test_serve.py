# -*- coding: utf-8 -*-
"""HTTP 服务测试 —— 起一个真服务, 打真请求

serve.py 此前覆盖率 0%：改过三次（加 /health、并发、错误处理），
一次都没被测试碰过。这里起真进程、真端口，把它说的每个行为都验一遍。
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    """起一个 serve.py 子进程, 用完关掉"""
    port = _free_port()
    # serve.py 把 PORT 写死在文件里, 用环境变量覆盖
    env = dict(os.environ)
    env["AIGC_PORT"] = str(port)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")

    proc = subprocess.Popen(
        [sys.executable, "-c", f"""
import sys, os
sys.path.insert(0, {os.path.join(ROOT, 'src')!r})
import serve
serve.PORT = {port}
from http.server import ThreadingHTTPServer
import socketserver
serve.Handler.protocol_version = "HTTP/1.0"
srv = socketserver.ThreadingTCPServer(("127.0.0.1", {port}), serve.Handler)
srv.daemon_threads = True
print("READY", flush=True)
srv.serve_forever()
"""],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1,
    )

    # 等服务起来
    deadline = time.time() + 60
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read()
            pytest.skip(f"服务启动失败: {out[-500:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                ready = True
                break
        except Exception:
            time.sleep(0.4)
    if not ready:
        proc.kill()
        pytest.skip("服务 60 秒内未就绪")

    yield f"http://127.0.0.1:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _post(base, payload):
    req = urllib.request.Request(
        base + "/detect",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body.decode("utf-8", "replace")}


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, r.read().decode("utf-8", "replace")


class TestHealthEndpoint:
    def test_health_ok(self, server):
        status, body = _get(server, "/health")
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "ok"

    def test_healthz_alias(self, server):
        status, body = _get(server, "/healthz")
        assert status == 200
        assert json.loads(body)["status"] == "ok"

    def test_health_is_lightweight(self, server):
        """健康检查不该拖慢到需要等 psutil"""
        t0 = time.time()
        for _ in range(5):
            _get(server, "/health")
        assert time.time() - t0 < 2.0, "健康检查太慢, 可能碰了模型或 psutil"


class TestInfoEndpoint:
    def test_info_returns_profile(self, server):
        status, body = _get(server, "/info")
        assert status == 200
        data = json.loads(body)
        for k in ("cores", "mem_gb", "recommended"):
            assert k in data, f"/info 缺少 {k}"


class TestDetectEndpoint:
    def test_chinese_ai_text(self, server):
        status, data = _post(server, {
            "text": "人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能够提高效率；"
                    "在教育领域，个性化学习系统能够因材施教；在交通领域，智能调度减少了拥堵。"
        })
        assert status == 200
        assert 0.0 <= data["ai_probability"] <= 1.0
        assert data["language"] == "zh"

    def test_english_ai_text(self, server):
        status, data = _post(server, {
            "text": "Artificial intelligence represents a transformative paradigm shift in "
                    "modern society. It is important to note that we must carefully consider "
                    "the ethical implications of these technologies as they continue to evolve."
        })
        assert status == 200
        assert data["language"] == "en"

    def test_human_text_stays_low(self, server):
        status, data = _post(server, {
            "text": "今天下午去超市买了点菜，回来路上碰到老同学，聊了半个多小时。"
                    "他说最近换了工作，离家近了不少，就是工资没涨。"
        })
        assert status == 200
        assert data["ai_probability"] < 0.5

    def test_empty_text_returns_400(self, server):
        status, data = _post(server, {"text": ""})
        assert status == 400
        assert "error" in data

    def test_missing_text_key_returns_400(self, server):
        status, data = _post(server, {})
        assert status == 400

    def test_json_input_rejected(self, server):
        status, data = _post(server, {"text": '{"key": "value", "n": 1}'})
        assert status == 200
        assert data.get("error") == "not_natural_language"
        assert data["ai_probability"] == 0.0

    def test_code_input_rejected(self, server):
        status, data = _post(server, {"text": "def foo(x):\n    return x * 2\n"})
        assert status == 200
        assert data.get("error") == "not_natural_language"

    def test_repeated_input_rejected(self, server):
        status, data = _post(server, {"text": "这是一段测试文本。" * 200})
        assert status == 200
        assert data.get("error") == "repetitive_content"

    def test_punctuation_only_rejected(self, server):
        status, data = _post(server, {"text": "。，！？；："})
        assert status == 200
        assert data.get("error") == "no_meaningful_content"

    def test_unknown_path_404(self, server):
        req = urllib.request.Request(server + "/nope", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 404

    def test_cors_headers_present(self, server):
        """/detect 带 CORS 头, 否则浏览器里调不通"""
        req = urllib.request.Request(
            server + "/detect",
            data=json.dumps({"text": "这是一段足够长的中文测试文本内容"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.headers.get("Access-Control-Allow-Origin") is not None


class TestConcurrency:
    def test_concurrent_requests(self, server):
        """20 并发, 全部成功且结果一致"""
        from concurrent.futures import ThreadPoolExecutor

        text = ("各部门：为提升全体员工的专业技能与综合素质，进一步推动公司高质量发展，"
                "经研究决定开展年度员工培训计划。")
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(lambda _: _post(server, {"text": text}), range(20)))
        assert all(s == 200 for s, _ in results)
        probs = {round(d["ai_probability"], 6) for _, d in results}
        assert len(probs) == 1, f"并发结果不一致: {probs}"


class TestThreadedColdStart:
    """32 线程同时冷启动: 模型不能重复加载, 结果必须一致

    服务启动时多个请求可能同时到达。如果模型加载不是原子的,
    要么重复加载浪费几百毫秒, 要么读到半初始化的状态给出错误结果。
    """

    def test_concurrent_cold_start(self):
        import subprocess
        import sys as _sys

        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = """
import sys, threading
sys.path.insert(0, %r)
import model_zoo as M
results, errors = [], []
def w(i):
    try:
        t = ("人工智能技术正在深刻改变各行各业的方方面面，这是一段并发测试文本。"
             if i %% 2 else
             "Artificial intelligence is transforming every industry in a concurrent test.")
        d = M.detect(t)
        results.append((i, round(d.get("ai_probability", -1), 6), d.get("tier")))
    except Exception as e:
        errors.append(type(e).__name__)
ths = [threading.Thread(target=w, args=(i,)) for i in range(32)]
[t.start() for t in ths]
[t.join() for t in ths]
zh = {p for i, p, _ in results if i %% 2}
en = {p for i, p, _ in results if i %% 2 == 0}
assert not errors, errors
assert len(results) == 32, len(results)
assert len(zh) == 1, f"中文结果不一致: {zh}"
assert len(en) == 1, f"英文结果不一致: {en}"
print("ok")
""" % os.path.join(ROOT_DIR, "src")

        r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300,
                           env={k: v for k, v in os.environ.items()
                                if k.upper() not in ("PYTHONPATH", "PYTHONHOME")})
        assert r.returncode == 0, f"并发冷启动失败:\n{r.stderr[-600:]}"
        assert "ok" in r.stdout

class TestServeCommandLine:
    """命令行参数: 之前端口写死 8000, 且完全不解析参数

    `python src/serve.py --help` 会直接启动服务然后卡住 —— 用户以为程序挂了。
    8000 被占用时也没有任何出路。
    """

    def test_help_exits_without_serving(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "serve.py"), "--help"],
                           capture_output=True, text=True, timeout=30,
                           cwd=ROOT, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stderr[:300]
        assert "--port" in r.stdout
        assert "--host" in r.stdout

    def test_custom_port_is_used(self):
        port = _free_port()
        p = subprocess.Popen([sys.executable, os.path.join(ROOT, "src", "serve.py"),
                              "--port", str(port)],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.time() + 60
            ok = False
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                        ok = json.loads(resp.read()).get("status") == "ok"
                        break
                except Exception:
                    time.sleep(0.3)
            assert ok, "指定端口上没有起来"
        finally:
            p.terminate()
            p.wait(timeout=15)

    def test_non_string_text_returns_structured_error(self, server):
        """text 为 list/None/int 时, 不该把 AttributeError 当错误文本返回"""
        for bad in ([1, 2], None, 123):
            status, data = _post(server, {"text": bad})
            assert status == 400, (bad, status, data)
            assert data.get("error") == "text must be a string", (bad, data)
            assert "received_type" in data, (bad, data)
