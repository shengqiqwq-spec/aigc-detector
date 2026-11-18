# -*- coding: utf-8 -*-
"""
Gradio 图形界面 - AIGC 检测
用法: pip install gradio && python app.py
然后浏览器打开 http://127.0.0.1:7860
"""
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import model_zoo as M

import gradio as gr

SAMPLE_ZH = """人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能提高效率；在教育领域，个性化学习系统能因材施教。然而，数据隐私和算法偏见等问题也需要重视，需要建立完善的监管框架来引导技术健康发展。"""

SAMPLE_EN = """Artificial intelligence is transforming various industries. In healthcare, AI-assisted diagnosis improves efficiency. In education, personalized learning systems help students. However, privacy concerns and algorithmic bias must be addressed through proper regulation."""


def detect(text):
    if not text or not text.strip():
        return "⚠️ 请输入文本", "", 0.0
    r = M.detect(text)
    p = r["ai_probability"]
    bucket = r["bucket"]
    icon = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(bucket, "⚪")
    summary = f"{icon} 判定：**{bucket}**  |  AI 概率：**{p:.1%}**  |  语言：{r['language'].upper()}  |  档位：{r['tier']}"
    detail = (
        f"- **文本长度**：{r['text_len']} 字\n"
        f"- **置信度**：{r['confidence']}\n"
        f"- **响应耗时**：{r['latency_ms']} ms\n"
        f"- **建议**：{'AI 生成可能性高' if p >= 0.7 else ('存疑，可能人机混合' if p >= 0.3 else '人类写作可能性高')}"
    )
    return summary, detail, p


with gr.Blocks(title="AIGC 检测器") as demo:
    gr.Markdown("""
    # 🔍 AIGC 文本检测器

    本地运行 · 中英双语自动识别 · 4 档模型自动选档

    **提示**：检测 150 字以上文本准确率最高（99%+），几十字短句仅供参考。
    """)

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(label="待检测文本", placeholder="粘贴要检测的文本（中文或英文）...", lines=8)
            with gr.Row():
                btn = gr.Button("🔍 开始检测", variant="primary", scale=2)
                clear = gr.Button("🗑 清空", scale=1)
            gr.Examples(examples=[[SAMPLE_ZH], [SAMPLE_EN]], inputs=text_input, label="示例")
        with gr.Column(scale=2):
            result = gr.Markdown(label="检测结果")
            detail = gr.Markdown(label="详细信息")
            prob = gr.Slider(0, 1, value=0, label="AI 概率", interactive=False)

    gr.Markdown("---\n### ⚠️ 免责声明\n本工具仅供研究与教育用途，检测结果不等同于 Turnitin 等商业系统，不能作为学术指控的唯一依据。")

    btn.click(detect, inputs=text_input, outputs=[result, detail, prob])
    clear.click(lambda: ("", "", 0.0), outputs=[text_input, result, prob])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
