# -*- coding: utf-8 -*-
"""
Gradio 网页界面 - 比 demo.html 更强大
用法: python src/app_gradio.py
然后浏览器打开 http://127.0.0.1:7860
"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gradio as gr
import model_zoo as M


def detect_fn(text):
    if not text or not text.strip():
        return "请输入文本", "", ""
    r = M.detect(text)
    prob = r["ai_probability"]
    bucket = r["bucket"]
    emoji = {"高": "🔴", "中": "🟡", "低": "🟢"}[bucket]
    verdict = f"{emoji} **{bucket}** — AI 概率 {prob:.1%}"
    detail = (f"- 语言：{r['language']}\n"
              f"- 使用档位：{r['tier']}\n"
              f"- 文本长度：{r['text_len']} 字\n"
              f"- 置信度：{r['confidence']}\n"
              f"- 耗时：{r['latency_ms']:.0f} ms")
    advice = {
        "高": "大概率是 AI 生成。建议人工复核，不要仅凭此结果下结论。",
        "中": "不确定——可能是人机混合或改写文本。建议结合上下文判断。",
        "低": "大概率是人类写作。",
    }[bucket]
    if r["text_len"] < 50:
        advice += "\n\n⚠️ 文本较短，准确率有限，仅供参考。"
    return verdict, detail, advice


def humanize_fn(text, method):
    if not text or not text.strip():
        return "请输入文本"
    if method == "规则改写（零依赖）":
        from humanize2 import humanize_v2
        return humanize_v2(text, strength=2.0)
    elif method == "程序化引擎（零幻觉）":
        from evasion_engine import humanize_programmatic
        out, _ = humanize_programmatic(text)
        return out
    elif method == "LLM改写（需14B服务）":
        try:
            from humanize_pro import rewrite
            return rewrite(text)
        except Exception as e:
            return f"LLM 服务未启动或出错：{e}\n\n请先运行 llama-server + Qwen3-14B。"
    return text


with gr.Blocks(title="AIGC 检测器") as demo:
    gr.Markdown("""
    # 🔍 AIGC 文本检测器 & 降AI工具

    完全本地运行 · 中英双语自动识别 · 多机型适配

    **提示**：检测 150 字以上文本准确率最高（99%+），短句仅供参考。
    """)

    with gr.Tab("📊 文本检测"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="待检测文本", placeholder="粘贴要检测的文本（中文或英文）...", lines=8)
                detect_btn = gr.Button("🔍 开始检测", variant="primary")
            with gr.Column():
                verdict_out = gr.Markdown(label="检测结果")
                detail_out = gr.Markdown(label="详细信息")
                advice_out = gr.Markdown(label="建议")
        detect_btn.click(detect_fn, inputs=[text_input], outputs=[verdict_out, detail_out, advice_out])
        gr.Examples(examples=[
            ["人工智能技术正在深刻改变各行各业。在医疗领域，AI辅助诊断能提高效率；在教育领域，个性化学习系统能因材施教。然而，数据隐私和算法偏见等问题也需要重视，需要建立完善的监管框架来引导技术健康发展。"],
            ["说实话这游戏我玩了三年，氪金不少，但最近越来越没意思了。策划天天搞那些花里胡哨的活动，实际内容一点不带更新的，我看是脑子进水了。"],
        ], inputs=[text_input], label="示例")

    with gr.Tab("🛠 降AI改写"):
        gr.Markdown("选择改写方式：规则改写(零依赖) / 程序化引擎(零幻觉) / LLM改写(最强,需14B)")
        with gr.Row():
            with gr.Column():
                hu_input = gr.Textbox(label="待改写文本", lines=8)
                method = gr.Radio(choices=["规则改写（零依赖）", "程序化引擎（零幻觉）", "LLM改写（需14B服务）"],
                                  value="规则改写（零依赖）", label="改写方式")
                hu_btn = gr.Button("✍️ 开始改写", variant="primary")
            with gr.Column():
                hu_output = gr.Textbox(label="改写结果", lines=8)
        hu_btn.click(humanize_fn, inputs=[hu_input, method], outputs=[hu_output])

    with gr.Tab("ℹ️ 关于"):
        gr.Markdown("""
        ### 模型能力
        - zh_final: F1 0.9933 (中文默认)
        - en_v5: F1 0.9955 (英文)
        - cnn_full: 0.954 (极速)
        - tier0: 0.897 (手机)

        ### 限制
        - 短文本(<50字)约96%
        - 识别模型指纹而非AI感
        - 不等同 Turnitin

        [GitHub](https://github.com/shengqiqwq-spec/aigc-detector)
        """)

if __name__ == "__main__":
    print("启动 Gradio: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
