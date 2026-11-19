# 贡献指南 / Contributing

[中文](#中文) | [English](#english)

---

## 中文

感谢你对本项目的兴趣！以下是参与贡献的说明。

### 🐛 报告问题

提交 Issue 时请包含：

1. **环境信息**：操作系统、Python 版本、是否安装 torch
2. **复现步骤**：具体命令
3. **预期结果 vs 实际结果**
4. **错误日志**（完整 traceback）

```bash
# 收集环境信息
python --version
python src/model_zoo.py --info
```

### 💡 建议功能

开 Issue 说明：
- 使用场景（你想解决什么问题）
- 你期望的行为
- 如果有类似的实现，参考链接

### 🔧 提交代码

```bash
# 1. Fork 并克隆
git clone https://github.com/<your-name>/aigc-detector.git
cd aigc-detector

# 2. 安装开发依赖
pip install -r requirements-dev.txt

# 3. 创建分支
git checkout -b feat/your-feature

# 4. 开发并测试
pytest tests/ -v

# 5. 提交（遵循约定式提交）
git commit -m "feat: add support for XXX"
git push origin feat/your-feature
```

### 📝 代码规范

- **Python 3.11+**，遵循 PEP 8（CI 覆盖 3.11 / 3.13 / 3.14）
- 新增功能**必须**带测试
- 保持零可选依赖的原则（核心功能只用 sklearn）
- 中文注释优先（项目主要面向中文用户）

### 🎯 特别欢迎的贡献

| 方向 | 说明 |
|---|---|
| **新模型风格数据** | 用 GPT-4 / 文心 / 豆包生成的 AI 文本，帮助提升泛化 |
| **新语言支持** | 日语、韩语等检测器 |
| **降AI 优化** | 更好的改写策略 |
| **文档翻译** | 更多语言版本的 README |
| **性能优化** | 推理速度、内存占用 |

### ⚠️ 数据提交规范

如果你要贡献训练数据：

- **必须**说明数据来源（哪个模型、什么 prompt 生成的）
- **不要**提交他人隐私数据或受版权保护的文本
- 附上生成脚本，便于复现

### 📜 许可

提交的代码默认采用 Apache 2.0 许可。

---

## English

Thanks for your interest in contributing!

### 🐛 Reporting Issues

Please include:

1. **Environment**: OS, Python version, whether torch is installed
2. **Reproduction steps**: exact commands
3. **Expected vs actual behavior**
4. **Full error traceback**

```bash
python --version
python src/model_zoo.py --info
```

### 🔧 Pull Requests

```bash
git clone https://github.com/<your-name>/aigc-detector.git
cd aigc-detector
pip install -r requirements-dev.txt
git checkout -b feat/your-feature

# develop & test
pytest tests/ -v

git commit -m "feat: add support for XXX"
git push origin feat/your-feature
```

### 📝 Code Style

- **Python 3.11+**, PEP 8 (CI covers 3.11 / 3.13 / 3.14)
- New features **must** have tests
- Keep core dependency-free (only sklearn for detect)
- Chinese comments preferred (primary audience)

### 🎯 Especially Welcome

| Area | Description |
|---|---|
| **New model-style data** | AI texts from GPT-4 / Claude / Gemini to improve generalization |
| **New language support** | Japanese, Korean detectors |
| **Humanizer improvements** | Better rewriting strategies |
| **Doc translations** | More README languages |
| **Performance** | Inference speed, memory footprint |

### ⚠️ Data Contribution Rules

- **Must** state data source (which model, which prompt)
- **Do not** submit private data or copyrighted text
- Include the generation script for reproducibility

### 📜 License

Contributions are licensed under Apache 2.0.
