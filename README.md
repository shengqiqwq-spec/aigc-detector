# AI Text Detector & Humanizer

A lightweight, fully-local toolkit for **AIGC text detection** and **AI-content humanization**, with a model matrix that scales from phones to desktops.

中文 | [English](#english)

## 亮点

- **多档位模型矩阵**：2MB 手机版 → 12MB 通用版，按硬件自动选择
- **中文检测 F1 = 0.9934**（独立测试集 27562 条，长文本 100%）
- **纯本地运行**：无需联网、无 API 费用、不上传任何数据
- **零幻觉降AI**：程序化 n-gram 审计方案，不改事实/数字
- **多硬件加速**：支持 CPU / Intel Arc GPU (Vulkan) / Intel NPU (OpenVINO)

## 快速开始

```bash
pip install -r requirements.txt

# 检测文本（自动判断语言 + 自动选档）
python src/model_zoo.py "待检测的文本"

# 查看机型与推荐档位
python src/model_zoo.py --info
```


## 在线 Demo（本地）



或者双击 `启动Demo.bat`（Windows）。

## 模型矩阵

| 档位 | 模型 | 体积 | 整体F1 | 短文本 | 长文本 | 适用 |
|---|---|---|---|---|---|---|
| **zh_final** ⭐ | TF-IDF+LR | 14MB | **0.9934** | 0.9921 | 1.0000 | 全机型（默认） |
| cnn_full | CharTextCNN | 5.1MB | 0.9540 | — | — | 极速21ms |
| tier0 | Hashing+NB | 2.1MB | 0.897 | 0.890 | 0.963 | 手机/嵌入式 |
| en | TF-IDF(en) | 8MB | 0.982 | — | — | 英文 |

*独立测试集 27562 条，基于 HC3 中文句子级增强数据

## 降 AI 率

| 方案 | 效果 | 依赖 |
|---|---|---|
| `humanize_pro.py` | **58.7%**（14B 自动迭代） | Qwen3-14B 本地 |
| `humanize2.py` | 44%（规则替换） | 无 |
| `evasion_engine.py` | 零幻觉 + n-gram 审计 | 无 |

## 文档

- [使用指南](docs/使用指南.md) — 什么场景准、怎么读结果
- [模型卡](docs/MODEL_CARD.md) — 训练数据、指标、限制
- [运行报告](docs/运行报告.md) — 一晚训练纪实

## 已知限制

1. **短文本弱于长文本**：<50 字信息不足，检测必然有歧义
2. **英文存在领域漂移**：HC3 英文为 2022 年 GPT-3.5 风格
3. **不等同 Turnitin**：独立指标，非商业检测器复刻
4. **猫鼠博弈**：生成模型进化后检测器需重训

## 伦理与免责

本工具仅供**研究与教育**用途。不得用于学术不端行为。检测结果不能作为认定学术不端的唯一证据。

## 许可

Apache 2.0。模型基于 [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (CC BY-SA 4.0) 训练。

---

<a name="english"></a>
## English

A fully-local AIGC text detection + humanization toolkit.

- **Chinese detection F1 = 0.9934** (independent test, 27,562 samples; 100% on long text)
- **Multi-tier models**: 2MB (mobile) → 14MB (universal), auto-selected by hardware
- **100% local**: no network, no API cost, no data upload
- **Multi-accelerator**: CPU / Intel Arc GPU (Vulkan) / Intel NPU (OpenVINO)

```bash
python src/model_zoo.py "your text here"      # detect
python src/model_zoo.py --info                 # show hardware profile
```

For research and educational use only. Not a substitute for Turnitin.
