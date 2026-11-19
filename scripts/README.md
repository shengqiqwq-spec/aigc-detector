# 训练脚本 / Training Scripts

**所有脚本均使用相对路径**（自动定位仓库根目录下的 `data/` 和 `ckpt/`），
clone 后即可运行，无需改代码。可用环境变量覆盖：

```bash
export AIGC_DATA_DIR=/path/to/data
export AIGC_CKPT_DIR=/path/to/ckpt
```

## 文件说明

| 脚本 | 作用 | 产出 |
|---|---|---|
| `train_zh_v7.py` | **中文检测器**（当前线上版本） | `ckpt/detector_zh_final/` |
| `train_zh_v8.py` | 中文 v8 实验：补 6 个文体盲区（**未上线**，见下） | `ckpt/detector_zh_v8_w*/` |
| `train_en_v8.py` | **英文检测器 v8**（当前线上版本，修了 9% 假阳性） | `ckpt/detector_en_v5/` |
| `gen_ai_text.py` | 用本地 LLM 生成当代 AI 文本（中/英） | `data/ai_{zh,en}_ov.jsonl` |
| `gen_zh_short.py` | 专门生成短文本 AI 样本 | `data/ai_zh_short2.jsonl` |
| `augment_data.py` | HC3 句子级增强 | — |
| `clean_en_data.py` | 英文数据清洗 | — |
| `gen_zh_blindspots.py` | 生成 6 个盲区文体（产品介绍/影评/攻略等） | `data/ai_zh_blindspots.jsonl` |
| `clean_blindspots.py` | 严格清洗盲区数据（去元话语/标题/markdown） | `data/ai_zh_blindspots_clean.jsonl` |
| `train_tier0.py` | **中文轻量档**（4MB，手机/嵌入式） | `ckpt/detector_tier0/` |
| `train_en_tier0.py` | **英文轻量档** | `ckpt/detector_en_tier0/` |
| `train_cnn_full.py` | CharTextCNN 极速档（需 torch） | `ckpt/detector_cnn_full/` |

## 复现完整流程

### 1. 准备基础数据

下载 HC3 中文数据集并放到 `data/hc3_all.jsonl`：

```bash
mkdir -p data
curl -L -o data/hc3_all.jsonl \
  https://hf-mirror.com/datasets/Hello-SimpleAI/HC3-Chinese/resolve/main/all.jsonl
```

英文对应 `HC3`（非 `HC3-Chinese`），存为 `data/hc3_en_real.jsonl`。

格式为每行一个 JSON：

```json
{"question": "...", "human_answers": ["..."], "chatgpt_answers": ["..."], "source": "..."}
```

### 2. 生成当代 AI 文本（关键步骤）

公共语料是 2022 年 GPT-3.5 时代的。**不补当代数据，模型对新模型完全失效**
（实测：只用 HC3 训练时，对现代模型输出的召回接近 0）。

```bash
# 需要 OpenVINO + 一个本地 LLM（示例用 Qwen2.5-1.5B int4）
pip install openvino-genai
python scripts/gen_ai_text.py zh 400    # 中文 400 条
python scripts/gen_ai_text.py en 300    # 英文 300 条
python scripts/gen_zh_short.py 500      # 短文本 500 条
```

### 3. 训练

```bash
python scripts/train_zh_v7.py     # 中文主力档
python scripts/train_en_v7.py     # 英文
python scripts/train_tier0.py     # 中文轻量档
python scripts/train_en_tier0.py  # 英文轻量档
```

脚本会打印每档 C 值的测试集/独立集 F1，并把最优模型存到 `ckpt/`。

## 训练配置（当前线上版本）

| 项目 | 中文 | 英文 |
|---|---|---|
| 基础语料 | HC3-Chinese 12,853 QA | HC3-English 21,432 QA |
| 当代 AI 补充 | 3,217 条 | 1,695 条 |
| 特征 | char(1,5) TF-IDF, 300k 维 | char_wb(2,5) 200k + word(1,2) 100k |
| 模型 | LogisticRegression C=12 | LogisticRegression C=10 |
| 切分 | **按原问题分组**（防泄漏） | 同左 |
| 独立集 F1 | **0.9802** | **0.9905** |
| 轻量档 tier0 F1 | 0.9144 (Hashing+NB, 4.2MB) | — |

⚠️ **模型文件格式**：脚本输出 `.joblib`，但线上加载器**优先读 `.pkl`**。
训练完请转换，否则改动不生效：

```python
import joblib, pickle
obj = joblib.load("ckpt/xxx/clf.joblib")
pickle.dump(obj, open("ckpt/xxx/clf.pkl","wb"), protocol=pickle.HIGHEST_PROTOCOL)
```

## 两个必须遵守的坑

### 1. 必须按原问题分组切分

HC3 每个问题都同时有人类答案和 AI 答案。用普通 `train_test_split`
会把同一问题的两边分到训练集和测试集，**模型学到的是话题而不是文风**。
症状：随机切分 F1 = 0.99，分组切分后塌到 0.50（等于瞎猜）。

### 2. 必须保留旧时代 AI 样本

只喂当代模型生成的数据，模型会变成"某个小模型的指纹检测器"。
实测：只用 Qwen-1.5B 数据训练后，对 HC3 的 GPT-3.5 文本召回只有 **2.8%**。
把 HC3 自带的 `chatgpt_answers` 一并作为 AI 正样本后，回到 **99.6%**。

正确做法：**AI 正类 = 语料自带的旧时代 AI + 若干不同当代模型的输出**。

## 环境

```
scikit-learn>=1.3
scipy>=1.10
numpy>=1.24
joblib>=1.3
```

生成数据另需 `openvino-genai`（或 llama.cpp + GGUF 模型）。

## 为什么 `train_zh_v8.py` 没有上线

它试图修 6 个文体盲区（产品介绍 / 学习方法 / 求职信 / 学术摘要 / 影评 / 旅游攻略），
实测这 6 类 AI 文本的检测概率只有 0.10–0.17。

生成 173 条专项样本、清洗、按 3/10/30/100 倍加权重训，结果是：

| 盲区权重 | 盲区识别 | 独立 F1 |
|---|---|---|
| 3 | 0/6 | 0.9742 |
| 10 | 1/6 | 0.9725 |
| 30 | 1/6 | 0.9743 |
| 100 | 2/6 | 0.9695 |

最好也只到 2/6，而 F1 从 v7 的 0.9814 掉到 0.9743。**净负收益，所以不部署。**

根因不是数据量：这类文本每句都由具体名词构成（型号、地名、片名、方法名），
TF-IDF 判的是词频统计，不认识的词等于不存在。换模型架构才能解决。
脚本保留下来，是为了别人换架构时可以复用这批数据和评估方法。
