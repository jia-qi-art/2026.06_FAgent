# 诊断生成模型对比实验

## 目标与模型

比较同一份 Relation-EVGAT、知识库和 LoRA/Qwen 状态证据下，不同生成模型输出结构化诊断与自然语言报告的能力。

- `rule-agent`：确定性规则基线，主要作为证据忠实度上限与流水线检查。
- `qwen-flash`：低成本弱基线。
- `qwen3.5-flash`：较新的轻量弱基线。
- `deepseek-v4-flash`：DeepSeek 同家族轻量消融。
- `deepseek-v4-pro`：目标模型。

冒烟实验默认使用两个数据集、每个数据集两个事件和两类问题，共 8 个案例、32 次付费模型调用。规则模型不产生 API 费用。

## 统一输出

每个模型都必须返回一个 JSON 对象，包含：

- `conclusion`
- `root_causes`
- `degraded_edges`
- `recommendations`
- `citations`
- `report_text`

其中 `report_text` 是自然语言运维报告，其余字段用于结构化评估和前端展示。

## 自动指标

- JSON 有效率与 Schema 完整率。
- EVGAT Top-1 根因提及率与 Top-3 候选覆盖率。
- 证据外传感器幻觉数量。
- 知识库引用精确率。
- 建议条数、报告长度、端到端延迟和 Token 用量。

这些指标评价格式、证据忠实度和可操作性，不等同于真实根因准确率。当前 WaDI 根因标签文件明确标记为 `pending_ground_truth_labels`；在人工标签补齐前，不报告 Hit@K、MRR 或“模型诊断准确率”。

## 运行

只生成案例并运行免费规则基线：

```powershell
python -m backend.benchmark_diagnosis_models
```

配置真实百炼 Key 后运行 API 模型：

```powershell
python -m backend.benchmark_diagnosis_models --execute
```

结果写入 `outputs/model_benchmark/`：`cases.json`、`results.json`、`summary.json` 和 `summary.csv`。

正式实验建议扩展到每个数据集 10 个事件：

```powershell
python -m backend.benchmark_diagnosis_models --events-per-dataset 10 --execute
```

