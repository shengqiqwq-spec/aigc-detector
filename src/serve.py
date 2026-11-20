# -*- coding: utf-8 -*-
"""
本地检测 API 服务 - 纯标准库, 零依赖

用法: python src/serve.py
然后浏览器打开 http://127.0.0.1:8000/
（页面是仓库根目录的 demo.html, 由本服务在 / 路径提供）
"""
import sys, os, json, http.server, socketserver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_zoo as M

# 启动即预热: 消除首次请求的冷启动延迟 (~2.7s -> 2ms)
try:
    M.warm_up()
except Exception:
    pass

class Handler(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        # 健康检查: 容器探针/负载均衡/监控都靠它, 必须极轻量
        if self.path in ("/health", "/healthz"):
            body = json.dumps({"status": "ok", "service": "aigc-detector"},
                              ensure_ascii=False).encode()
            self.send_response(200); self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif self.path == "/info":
            body = json.dumps(M.detect_profile(), ensure_ascii=False).encode()
            self.send_response(200); self._cors()
            self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            self.send_response(200); self._cors()
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers()
            demo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo.html")
            if os.path.exists(demo):
                self.wfile.write(open(demo, encoding="utf-8").read().encode())
            else:
                self.wfile.write(b"demo.html not found")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/detect":
            self.send_response(404); self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            text = data.get("text", "")
            if not text.strip():
                raise ValueError("empty text")
            r = M.detect(text)
            body = json.dumps(r, ensure_ascii=False).encode()
            self.send_response(200); self._cors()
            self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}, ensure_ascii=False).encode()
            self.send_response(400); self._cors()
            self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    # Windows 控制台默认 cp936/cp1252, 直接 print 中文会乱码或 UnicodeEncodeError。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    # 端口原来写死 8000, 而且完全不解析命令行参数 —— `python serve.py --help`
    # 会直接把服务启起来然后卡住, 用户以为程序挂了。8000 被占用时也没有出路。
    import argparse
    ap = argparse.ArgumentParser(
        description="AIGC 文本检测 HTTP 服务",
        epilog="示例: python src/serve.py --port 8000\n"
               "      curl -X POST http://127.0.0.1:8000/detect "
               "-H 'Content-Type: application/json' -d '{\"text\":\"要检测的文本\"}'")
    ap.add_argument("--port", type=int, default=8000, help="监听端口 (默认 8000)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址 (默认 127.0.0.1; 局域网访问用 0.0.0.0)")
    args = ap.parse_args()
    PORT = args.port
    HOST = args.host

    # 多线程服务器: 单线程 TCPServer 在并发>队列时直接拒绝连接(实测50并发失败14个)
    class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True
        request_queue_size = 128      # 默认5太小, 并发一上来就拒绝

    socketserver.TCPServer.allow_reuse_address = True
    # 启动前预热, 让首个请求就是热的
    import time as _t
    _t0 = _t.time()
    _loaded = M.warm_up()
    print(f"模型预热完成: {', '.join(_loaded)} ({( _t.time()-_t0)*1000:.0f}ms)")
    try:
        httpd = ThreadedServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"端口 {PORT} 绑定失败: {e}")
        print(f"换一个端口试试: python src/serve.py --port {PORT + 1}")
        sys.exit(1)
    shown = "127.0.0.1" if HOST in ("0.0.0.0", "") else HOST
    print(f"检测服务启动: http://{shown}:{PORT}")
    print(f"Demo页面:    http://{shown}:{PORT}/")
    print(f"接口:        POST http://{shown}:{PORT}/detect  {{\"text\": \"...\"}}")
    print("按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
