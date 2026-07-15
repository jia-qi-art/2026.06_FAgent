import React, { useState } from "react";

const API = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";
const CHART_COLORS = ["#2e566d", "#c85b45", "#4f7b45", "#7a5da8", "#c58b2b", "#317f8f", "#b54f7a", "#60798f"];

async function postJson(path: string, body: any) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function formatError(err: any) {
  const raw = String(err?.message || err || "请求失败");
  try {
    const parsed = JSON.parse(raw);
    return parsed.detail || raw;
  } catch {
    return raw;
  }
}

function generatorLabel(value?: string) {
  return value === "kimi" ? "Kimi 生成" : "非 Kimi 来源";
}

export default function TableAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState<any>(null);
  const [question, setQuestion] = useState("哪个设备报警次数最多？");
  const [answer, setAnswer] = useState<any>(null);
  const [error, setError] = useState("");
  const [busyAction, setBusyAction] = useState<"" | "upload" | "query">("");

  const upload = async () => {
    if (!file) return;

    setBusyAction("upload");
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(API + "/api/table/upload", {
        method: "POST",
        body: form,
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMeta(data);
      setAnswer(null);
    } catch (err: any) {
      setError(formatError(err));
    } finally {
      setBusyAction("");
    }
  };

  const ask = async () => {
    if (!meta?.file_id) return;

    setBusyAction("query");
    setError("");
    try {
      const data = await postJson("/api/table/query", {
        file_id: meta.file_id,
        question,
      });
      setAnswer(data);
    } catch (err: any) {
      setError(formatError(err));
    } finally {
      setBusyAction("");
    }
  };

  const onFileChange = (selected: File | null) => {
    setFile(selected);
    setMeta(null);
    setAnswer(null);
    setError("");
  };

  const isUploading = busyAction === "upload";
  const isQuerying = busyAction === "query";
  const examples = ["自动决定展示内容，并给出趋势曲线和占比饼图", "每天的报警数量趋势是什么？", "各报警类型占比是多少？"];

  return (
    <div className="table-layout">
      <section className="panel table-upload-panel">
        <div className="panel-title">
          <h2>表格智能分析 Agent</h2>
          <p>Excel / CSV 上传、字段识别、自然语言查询、SQL 结果解释</p>
        </div>

        <label className="field-label">数据文件</label>
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => onFileChange(e.target.files?.[0] || null)}
        />

        <div className="diagnosis-actions">
          <button onClick={upload} disabled={!file || Boolean(busyAction)}>
            {isUploading ? "解析中..." : "上传并解析"}
          </button>
          <span>{file ? file.name : "未选择文件"}</span>
        </div>

        {meta && (
          <div className="table-meta-grid">
            <div><span>文件</span><strong>{meta.filename}</strong></div>
            <div><span>Sheet</span><strong>{meta.active_sheet}</strong></div>
            <div><span>行数</span><strong>{meta.rows}</strong></div>
            <div><span>列数</span><strong>{meta.columns}</strong></div>
          </div>
        )}

        {error && <div className="state error table-error">{error}</div>}
      </section>

      {meta && (
        <section className="panel">
          <div className="panel-title">
            <h2>字段结构</h2>
            <p>{meta.schema.length} 个业务字段，预览前 5 个样例值</p>
          </div>

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>字段名</th>
                  <th>类型</th>
                  <th>缺失率</th>
                  <th>唯一值</th>
                  <th>样例值</th>
                </tr>
              </thead>
              <tbody>
                {meta.schema.map((col: any) => (
                  <tr key={col.name}>
                    <td>{col.name}</td>
                    <td>{col.type}</td>
                    <td>{Math.round(col.missing_ratio * 10000) / 100}%</td>
                    <td>{col.unique_count}</td>
                    <td>{(col.sample_values || []).join(" / ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {meta && (
        <section className="panel">
          <div className="panel-title">
            <h2>自然语言查询</h2>
            <p>Kimi 生成安全 SELECT SQL，后端校验后在 DuckDB 中执行</p>
          </div>

          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} />

          <div className="table-question-examples">
            {examples.map((item) => (
              <button key={item} onClick={() => setQuestion(item)} disabled={Boolean(busyAction)}>
                {item}
              </button>
            ))}
          </div>

          <div className="diagnosis-actions">
            <button onClick={ask} disabled={Boolean(busyAction) || !question.trim()}>
              {isQuerying ? "分析中..." : "开始分析"}
            </button>
            <span>{isQuerying ? "正在请求 Kimi 并执行 SQL..." : "就绪"}</span>
          </div>
        </section>
      )}

      {answer && (
        <section className="panel">
          <div className="panel-title">
            <h2>生成的 SQL</h2>
            <p>{answer.llm_reason}</p>
          </div>
          <div className={`table-source ${answer.generated_by === "kimi" ? "kimi" : "rules"}`}>
            <strong>{generatorLabel(answer.generated_by)}</strong>
            {answer.fallback_reason && <span>原因：{answer.fallback_reason}</span>}
          </div>
          <pre className="sql-box">{answer.sql}</pre>
        </section>
      )}

      {answer && (
        <section className="panel">
          <div className="panel-title">
            <h2>查询结果</h2>
            <p>
              来源：{answer.source_location.filename} / {answer.source_location.sheet} / 返回 {answer.result.length} 行
            </p>
          </div>

          <ResultTable rows={answer.result} />
        </section>
      )}

      {answer && (
        <section className="panel">
          <div className="panel-title">
            <h2>智能解释与可视化</h2>
            <p>解释、图表 SQL 和图表字段均来自 Kimi 本次生成的多图表计划</p>
          </div>

          <div className="report-card">
            <p>{answer.explanation}</p>
          </div>

          <ChartGallery answer={answer} />
        </section>
      )}
    </div>
  );
}

function ResultTable({ rows }: { rows: any[] }) {
  if (!rows?.length) return <div className="state compact">没有查询结果</div>;

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(0, 100).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartView({ spec }: { spec: any }) {
  if (!spec || spec.chart_type === "table") return null;
  if (!spec.data?.length || !spec.x_field || !spec.y_field) return null;

  if (spec.chart_type === "line") return <LineChart spec={spec} />;
  if (spec.chart_type === "pie") return <PieChart spec={spec} />;
  return <BarChart spec={spec} />;
}

function ChartGallery({ answer }: { answer: any }) {
  const specs = Array.isArray(answer?.visualizations) && answer.visualizations.length
    ? answer.visualizations
    : answer?.chart_spec
      ? [answer.chart_spec]
      : [];
  const renderable = specs.filter(isRenderableSpec);

  if (!renderable.length) return <div className="state compact">Kimi 没有返回可渲染图表数据</div>;

  return (
    <div className="visualization-stack">
      {renderable.map((spec: any, index: number) => (
        <div className="viz-block" key={`${spec.chart_type}-${index}-${spec.title || spec.sql || ""}`}>
          <ChartView spec={spec} />
          {spec.reason && <p className="viz-note">Kimi 选择原因：{spec.reason}</p>}
        </div>
      ))}
    </div>
  );
}

function isRenderableSpec(spec: any) {
  return Boolean(
    spec
    && spec.chart_type !== "table"
    && spec.data?.length
    && spec.x_field
    && spec.y_field,
  );
}

function formatAxisLabel(value: any) {
  const text = String(value || "");
  if (text === "x_field") return "类别";
  if (text === "y_field") return "数值";
  return text;
}

function BarChart({ spec }: { spec: any }) {
  const values = spec.data.map((item: any) => Number(item[spec.y_field]) || 0);
  const max = Math.max(...values, 1);

  return (
    <div className="simple-chart">
      <h3>{spec.title}</h3>
      {spec.data.slice(0, 12).map((item: any, index: number) => {
        const value = Number(item[spec.y_field]) || 0;
        const width = Math.max(4, (value / max) * 100);

        return (
          <div className="chart-row" key={`${index}-${String(item[spec.x_field])}`}>
            <span title={String(item[spec.x_field])}>{String(item[spec.x_field])}</span>
            <div className="chart-bar-bg">
              <div className="chart-bar-fill" style={{ width: `${width}%` }} />
            </div>
            <strong>{value}</strong>
          </div>
        );
      })}
    </div>
  );
}

function LineChart({ spec }: { spec: any }) {
  const rows = spec.data
    .map((item: any) => ({
      label: String(item[spec.x_field] ?? ""),
      value: Number(item[spec.y_field]) || 0,
    }))
    .filter((item: any) => item.label);

  if (!rows.length) return null;

  const width = 720;
  const height = 260;
  const padX = 42;
  const padY = 28;
  const max = Math.max(...rows.map((item) => item.value), 1);
  const min = Math.min(...rows.map((item) => item.value), 0);
  const span = max - min || 1;

  const points = rows.map((item, index) => {
    const x = padX + (index / Math.max(1, rows.length - 1)) * (width - padX * 2);
    const y = height - padY - ((item.value - min) / span) * (height - padY * 2);
    return { ...item, x, y };
  });
  const polyline = points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");

  return (
    <div className="simple-chart">
      <h3>{spec.title}</h3>
      <svg className="table-line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={spec.title}>
        <line x1={padX} x2={width - padX} y1={height - padY} y2={height - padY} />
        <line x1={padX} x2={padX} y1={padY} y2={height - padY} />
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = height - padY - ratio * (height - padY * 2);
          return <line className="grid-line" key={ratio} x1={padX} x2={width - padX} y1={y} y2={y} />;
        })}
        <polyline points={polyline} />
        {points.map((point) => (
          <g key={`${point.label}-${point.x}`}>
            <circle cx={point.x} cy={point.y} r="4" />
            <title>{`${point.label}: ${point.value}`}</title>
          </g>
        ))}
      </svg>
      <div className="line-labels">
        <span>{rows[0].label}</span>
        <strong>{formatAxisLabel(spec.y_field)}</strong>
        <span>{rows[rows.length - 1].label}</span>
      </div>
    </div>
  );
}

function PieChart({ spec }: { spec: any }) {
  const rows = spec.data
    .map((item: any, index: number) => ({
      label: String(item[spec.x_field] ?? ""),
      value: Math.max(0, Number(item[spec.y_field]) || 0),
      color: CHART_COLORS[index % CHART_COLORS.length],
    }))
    .filter((item: any) => item.label && item.value > 0);

  if (!rows.length) return null;

  const total = rows.reduce((sum: number, item: any) => sum + item.value, 0) || 1;
  let startAngle = -90;

  return (
    <div className="simple-chart">
      <h3>{spec.title}</h3>
      <div className="pie-layout">
        <svg className="pie-chart" viewBox="0 0 260 260" role="img" aria-label={spec.title}>
          {rows.map((item: any) => {
            const angle = (item.value / total) * 360;
            const path = describeArc(130, 130, 96, startAngle, startAngle + angle);
            startAngle += angle;
            return (
              <path key={item.label} d={path} fill={item.color}>
                <title>{`${item.label}: ${item.value}`}</title>
              </path>
            );
          })}
          <circle cx="130" cy="130" r="48" />
          <text x="130" y="125" textAnchor="middle">合计</text>
          <text x="130" y="146" textAnchor="middle">{total}</text>
        </svg>
        <div className="pie-legend">
          {rows.map((item: any) => (
            <div key={item.label}>
              <i style={{ background: item.color }} />
              <span>{item.label}</span>
              <strong>{Math.round((item.value / total) * 1000) / 10}%</strong>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function describeArc(cx: number, cy: number, radius: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, radius, endAngle);
  const end = polarToCartesian(cx, cy, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return [
    "M", cx, cy,
    "L", start.x, start.y,
    "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y,
    "Z",
  ].join(" ");
}

function polarToCartesian(cx: number, cy: number, radius: number, angleInDegrees: number) {
  const angleInRadians = (angleInDegrees * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(angleInRadians),
    y: cy + radius * Math.sin(angleInRadians),
  };
}
