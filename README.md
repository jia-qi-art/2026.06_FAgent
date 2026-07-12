# 工业时序异常诊断 Agent 平台（Relation-EVGAT Demo）

这是一个独立版 MVP：项目内包含 Relation-EVGAT 必要算法脚本、WaDI/SMD 样例数据和默认 outputs，不需要运行时引用旧项目目录。

## 启动虚拟环境（可选）

```powershell
.venv\Scripts\activate
```

## 配置模型与向量检索

复制 `.env.example` 中的配置到项目根目录 `.env`，并填写真实的阿里云百炼 API Key：

```dotenv
LLM_PROVIDER=dashscope
LLM_MODEL=deepseek-v4-pro
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_REASONING_EFFORT=high

RAG_USE_CHROMA=true
RAG_VECTOR_PROVIDER=chroma
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
```

`EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL` 留空时分别复用 `LLM_API_KEY` 和 `LLM_BASE_URL`。不要提交包含真实密钥的 `.env`。

表格分析 Agent 使用独立 Kimi 配置，不影响诊断链路模型：

```dotenv
TABLE_AGENT_KIMI_API_KEY=sk-xxxxxxxx
TABLE_AGENT_KIMI_BASE_URL=https://api.moonshot.cn/v1
TABLE_AGENT_KIMI_MODEL=kimi-k2.6
```

## 启动后端

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173/dashboard`。

## 主要功能

- `/dashboard`：实时监控、异常分数、报警时间线、Agent 面板。
- `/relations`：传感器关系图、Top 退化边、边向量对比。
- `/root-cause`：Top-K 根因候选、证据卡片、历史曲线。
- `/diagnosis`：汇总 Relation-EVGAT、LoRA/Qwen 和知识库证据，由 DeepSeek V4 Pro 生成诊断。
- `/table-analysis`：上传 Excel / CSV，由 Kimi 生成安全 SQL 和多图表计划，在 DuckDB 中执行并展示结果。
- `/knowledge`：本地 Chroma 文档索引与检索，embedding 不可用时自动降级关键词检索。
- `/report`：结构化证据、自然语言报告和工具调用日志。
- `/finetune`：Qwen LoRA 微调、评估和 adapter 管理。
- `/history`：诊断任务与文档智检历史。

文档提取、PDF/图片 OCR、工业信息抽取和跨模态分析继续使用 Ernie，不受 DeepSeek 诊断链路影响。

## 后端接口

- `GET /api/health`
- `GET /api/datasets`
- `POST /api/jobs/train`
- `GET /api/jobs/{job_id}`
- `GET /api/overview?dataset=WaDI_A2_ds10`
- `GET /api/timeseries?dataset=WaDI_A2_ds10`
- `GET /api/relation-graph?dataset=WaDI_A2_ds10&event_id=1`
- `GET /api/root-cause?dataset=WaDI_A2_ds10&event_id=1`
- `POST /api/agent/ask`
- `GET /api/report?dataset=WaDI_A2_ds10&event_id=1`
- `POST /api/table/upload`
- `GET /api/table/{file_id}/schema`
- `POST /api/table/query`
- `POST /api/diagnosis/tasks`
- `GET /api/diagnosis/tasks/{task_id}`
- `GET /api/diagnosis/history`
- `GET /api/knowledge/documents`
- `POST /api/knowledge/upload`
- `POST /api/knowledge/search`
- `POST /api/document/extract-info`
- `POST /api/agent/cross-modal`
- `GET /api/finetune/status`

## 表格分析 Agent

数据流：

```text
上传 Excel/CSV -> pandas 读取 -> DuckDB 写入 table_data -> schema 预览
自然语言问题 -> Kimi 生成主 SELECT SQL + visualizations 图表 SQL -> 后端安全校验 -> DuckDB 执行 -> 查询结果、解释、曲线图、饼图和可选柱状图
```

安全约束：

- 只允许查询 `table_data` 的 `SELECT` 语句。
- 禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`CREATE`、`COPY`、`PRAGMA` 等写入或外部读取能力。
- `REPLACE()` 等普通字符串函数不会被误判为危险操作。
- Kimi 图表计划必须至少包含 `line` 和 `pie`，后端逐条执行并校验图表数据。

## 训练说明

前端“启动轻量训练”会调用本项目内的 `relation_evgat/run_top_ready_relation_gat.py`，默认使用 `epochs=1`、`max_train_windows=1000` 做快速闭环验证。首次演示无需等待训练，系统会直接读取复制进来的 `outputs/top_ready_relation_gat/WaDI_A2_ds10/full_joint`。

## 模型对比实验

实验比较规则基线、Qwen Flash、Qwen 3.5 Flash、DeepSeek V4 Flash 和 DeepSeek V4 Pro。默认冒烟规模为 2 个数据集 × 2 个事件 × 2 类问题，共 8 个案例。

仅生成案例并运行免费规则基线：

```powershell
python backend\benchmark_diagnosis_models.py
```

使用百炼 API 执行 32 次冒烟调用：

```powershell
python backend\benchmark_diagnosis_models.py --execute
```

生成两张对比图：

```powershell
python backend\make_model_benchmark_figures.py
```

实验结果位于 `outputs/model_benchmark/`：

- `summary.csv`：完整模型指标表。
- `artifacts/fig01_quality_comparison.png|pdf`：结构化质量与证据忠实度分组柱状图。
- `artifacts/fig02_quality_latency_pareto.png|pdf`：质量—延迟 Pareto 图。
- `artifacts/table01_model_metrics.xlsx`：格式化模型指标表。

当前项目尚无人工确认的事件级根因标签，因此这些指标评估 JSON/Schema、证据忠实度、引用、幻觉、延迟和 Token 用量，不应表述为真实诊断准确率。完整实验定义见 `MODEL_BENCHMARK.md`。
