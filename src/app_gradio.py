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

# 后台线程预热模型, 不阻塞界面启动
import threading
def _warm():
    try:
        M.warm_up()
    except Exception:
        pass
threading.Thread(target=_warm, daemon=True).start()


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
    # 用前缀匹配而不是全等: 下拉框文案带后缀(如"（零依赖，推荐）"),
    # 全等比较曾导致三个选项全部失配、静默返回原文。
    m = (method or "").strip()
    if m.startswith("规则改写"):
        from humanize2 import humanize_v2
        return humanize_v2(text, strength=2.0)
    if m.startswith("程序化"):
        from evasion_engine import humanize_programmatic
        out, _ = humanize_programmatic(text)
        return out
    if m.startswith("LLM"):
        try:
            from humanize_pro import rewrite
            return rewrite(text)
        except Exception as e:
            return f"LLM 服务未启动或出错：{e}\n\n请先运行 llama-server + Qwen3-14B。"
    raise ValueError(f"未知改写方式: {method!r}")


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
        gr.Markdown("选择改写方式：**规则改写**(零依赖,秒出,推荐) / **程序化引擎**(零幻觉,保数字) / **LLM改写**(需本地大模型,本机裸跑约50秒/段)")
        with gr.Row():
            with gr.Column():
                hu_input = gr.Textbox(label="待改写文本", lines=8)
                method = gr.Radio(choices=["规则改写（零依赖，推荐）", "程序化引擎（零幻觉）", "LLM改写（需本地大模型，慢）"],
                                  value="规则改写（零依赖，推荐）", label="改写方式")
                hu_btn = gr.Button("✍️ 开始改写", variant="primary")
            with gr.Column():
                hu_output = gr.Textbox(label="改写结果", lines=8)
        hu_btn.click(humanize_fn, inputs=[hu_input, method], outputs=[hu_output])

    with gr.Tab("ℹ️ 关于"):
        gr.Markdown("""
        ### 模型能力（分组防泄漏评估）

        | 档位 | 语言 | F1 | 体积 | 说明 |
        |---|---|---|---|---|
        | tier1 | 中文 | 0.9802 | 13MB | 默认 |
        | en | 英文 | 0.9905 | 12MB | 英文默认 |
        | tier3 | 中文 | 0.9709 | 1.9MB | 需 torch, 0.5ms |
        | tier0 | 中文 | 0.9144 | 4.2MB | 手机/嵌入式 |
        | en_tier0 | 英文 | 0.9408 | 4.2MB | 手机/嵌入式 |

        ### 怎么用才准

        - **150 字以上**最可靠（长文本 F1 ≈ 0.99）
        - **40 字以下**信息不足，结果仅供参考，会向"中"档收缩
        - 同一次判断请**至少测两段**，单段结果波动正常
        - 判"高"不等于"作弊"，判"低"也不等于"没用 AI"，**只作参考不作指控**

        ### 已知限制

        - **识别模型指纹，不是抽象"AI 感"**：对真实模型输出极准，但人类模仿 AI 风格可能漏判
        - **知情者可以绕过**：刻意避开 AI 语言习惯（不用"首先/其次/综上所述"、多写短句和具体细节）会判低。任何基于统计特征的检测器都有这个边界
        - **换新模型需重训**：模型持续进化，数据要重新生成
        - **不等同 Turnitin**：独立指标，无法对齐任何商业检测器的精确百分比

        ### 降AI 说明

        - **规则改写**：零依赖、秒出，降幅约 15–40%，语句通顺但不保证大幅降分
        - **程序化引擎**：靠 n-gram 审计打散特征，**绝不改数字和事实**，适合需要保真度的场合
        - **LLM 改写**：需要本地大模型。14B 单轮可把 0.89 降到 0.18，但本机约 0.7 tok/s（一段要 50 秒）；1.5B 快但力度不足。没有 GPU 建议用前两种

        [GitHub](https://github.com/shengqiqwq-spec/aigc-detector)
        """)

if __name__ == "__main__":
    print("启动 Gradio: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
