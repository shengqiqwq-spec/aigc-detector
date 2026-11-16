---
language:
  - zh
  - en
license: apache-2.0
tags:
  - ai-generated-text-detection
  - text-classification
  - chinese
  - lightweight
  - tfidf
datasets:
  - Hello-SimpleAI/HC3
  - Hello-SimpleAI/HC3-Chinese
metrics:
  - f1
  - roc_auc
---

# AIGC Detector - Model Card

## 模型概览

| 档位 | 模型 | 体积 | 整体F1 | 短文本 | 长文本 | 依赖 |
|---|---|---|---|---|---|---|
| **zh_v5** ⭐ | TF-IDF+LR | 12MB | **0.9927** | 0.9912 | 1.0000 | sklearn |
| en_v4 | TF-IDF+LR | 1.2MB | 0.9919 | — | — | sklearn |
| tier0 | Hashing+NB | 2.1MB | 0.897 | 0.890 | 0.963 | sklearn |
| cnn_full | CharTextCNN | 5.1MB | 0.954 | — | — | torch |

## 训练数据

- **中文**: HC3-Chinese 句子级增强 183,743 条 + Qwen2.5 生成当代中文 AI 文本 560 条（提升新式 AI 识别）
- **英文**: HC3-English 46,296 人类 + Qwen2.5 生成当代英文 AI 500 条（修复领域漂移）

## 关键设计

### 1. 当代数据混合
HC3 是 2022 年 GPT-3.5 数据，对当代模型输出识别率低。混入本地 Qwen 生成的当代 AI 文本后：
- 中文当代 AI 检测率：96.7%（29/30）
- 英文真实文本：AI 0.73/AI 0.71 vs 人类 0.14/0.10

### 2. 全量数据不采样
17 万条全量训练 vs 6 万采样，F1 提升 2.2 个点。

### 3. 字符级 TF-IDF
`char(1,5)` + 250k 特征 + C=8，无需分词，中文友好。

## 评估方法

独立测试集 27,562 条（随机种子 999，与训练种子 42 不同），分长度段评估。

## 局限

1. **短文本弱于长文本**：<50 字信息不足
2. **人类模仿 AI 可能漏判**
3. **不等同 Turnitin**
4. **需随生成模型进化重训**

## 许可

Apache 2.0。基于 HC3 (CC BY-SA 4.0) 训练。
