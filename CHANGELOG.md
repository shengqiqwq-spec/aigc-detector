# 更新日志 / Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-11-17

首个公开版本。

### ✨ 新增

- **4 档检测模型**，覆盖手机到服务器：
  - `zh_final` (14MB) — 中文主力，F1 0.9933
  - `en_v5` (1.5MB) — 英文，F1 0.9955
  - `cnn_full` (5.1MB) — 极速 CharTextCNN，21ms
  - `tier0` (2.1MB) — 手机版，F1 0.8853
- **中英双语自动路由**：按 CJK 字符占比判断语言，毫秒级切换模型
- **降AI 三档方案**：
  - `humanize_pro.py` — 14B LLM 自动迭代，降幅 58.7%
  - `humanize2.py` — 规则改写，零依赖，降幅 44%
  - `evasion_engine.py` — 程序化 n-gram 审计，零幻觉保事实
- **网页 Demo** (`demo.html`) + 本地 HTTP API (`src/serve.py`)
- **完整测试**：22 个 pytest 用例，GitHub Actions CI
- **双语文档**：README.md (中文) + README_EN.md (English)

### 🔬 技术细节

- 训练数据：HC3 中文 18.4 万条 + Qwen 生成当代 AI 文本 1,860 条
- 模型：字符级 TF-IDF (char 1-5, 300k 维) + LogisticRegression
- 独立测试集评估：27,562 条，F1 0.9933（长文本 99.95%）
- 实测：真实 AI 文本 100% 判高，人类文本 0% 误判

### 🎯 已知限制

- 短文本（<50 字）准确率约 96%，信息量不足是根本原因
- 识别的是「模型指纹」而非抽象「AI 感」，人类模仿 AI 可能漏判
- 不等同 Turnitin 等商业检测器

---

## 规划中

- [ ] 支持更多 AI 模型风格（GPT-4 / Claude / 文心 / 豆包）
- [ ] 日语 / 韩语检测器
- [ ] Gradio 图形界面
- [ ] 模型量化压缩（进一步减小体积）
- [ ] 批量检测 CLI
