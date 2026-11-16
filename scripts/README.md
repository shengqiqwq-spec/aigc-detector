# 训练脚本说明

这些脚本用于复现模型的训练过程。**注意：脚本中有硬编码的 `D:\ai_train` 路径**，使用前需要改成你自己的路径。

## 文件说明

| 脚本 | 作用 |
|---|---|
| `train_zh_v6.py` | 中文检测器训练（最终版 zh_final） |
| `train_en_v4.py` | 英文检测器训练（en_v5） |
| `train_cnn_full.py` | CNN 极速模型训练 |
| `augment_data.py` | 数据增强（句子级切分） |
| `gen_zh_modern.py` | 用 Qwen 生成当代中文 AI 数据 |
| `gen_zh_div.py` | 多样化中文 AI 生成 |
| `gen_en_npu.py` | 英文 AI 数据生成（NPU/GPU） |
| `clean_en_data.py` | 英文数据清洗 |

## 复现步骤

### 1. 准备基础数据
```bash
# 下载 HC3 中文数据集
# https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese
# 放到 data/hc3_all.jsonl
```

### 2. 数据增强
```bash
python scripts/augment_data.py   # 生成 hc3_aug.jsonl
```

### 3. 生成当代 AI 数据（需要本地 LLM）
```bash
# 需要 OpenVINO 或 llama.cpp + Qwen 模型
python scripts/gen_zh_modern.py
```

### 4. 训练
```bash
python scripts/train_zh_v6.py    # 中文
python scripts/train_en_v4.py    # 英文
```

## 修改路径

在脚本开头找到：
```python
OUT = r"D:\ai_train\ckpt\detector_zh_final"
DATA = r"D:\ai_train\data\hc3_aug.jsonl"
```
改成你自己的路径即可。

## 训练配置（中文最终版）

- 数据：HC3 183743 + 当代 AI 1860
- 特征：字符 TF-IDF char(1,5), 300k 维
- 模型：LogisticRegression C=12
- 结果：F1=0.9933（独立测试集）

## 环境

```
scikit-learn>=1.3
scipy>=1.10
numpy>=1.24
joblib>=1.3
```

生成当代 AI 数据另需 openvino-genai 或 llama.cpp。
