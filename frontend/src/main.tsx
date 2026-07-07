import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";


const API = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";
const STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "训练完成",
  failed: "失败",
  created: "已创建",
  detecting: "检测中",
  completed: "已完成",
  ready: "就绪",
  ok: "成功",
};

const STAGE_LABELS: Record<string, string> = {
  created: "创建任务",
  detecting: "异常检测",
  evidence_collecting: "证据汇总",
  retrieving_knowledge: "知识检索",
  reasoning: "诊断推理",
  reporting: "生成报告",
  completed: "完成",
};

const TOOL_LABELS: Record<string, string> = {
  get_event_summary: "异常事件摘要",
  rank_root_causes: "根因候选排序",
  inspect_edge_degradation: "关系退化检查",
  inspect_sensor_window: "变量窗口检查",
  retrieve_maintenance_knowledge: "知识库检索",
  generate_report: "诊断报告生成",
  detect_event: "异常检测",
  rank_root_cause: "根因排序",
  erniebot_reasoning: "大模型推理",
};

function statusLabel(value?: string) {
  return STATUS_LABELS[value || ""] || value || "未知";
}

function stageLabel(value?: string) {
  return STAGE_LABELS[value || ""] || value || "未知阶段";
}

function toolLabel(value?: string) {
  return TOOL_LABELS[value || ""] || value || "工具调用";
}

async function getJson(path: string) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postJson(path: string, body: any) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function useData(path: string, deps: any[]) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    setError("");
    getJson(path).then((d) => live && setData(d)).catch((e) => live && setError(String(e)));
    return () => { live = false; };
  }, deps);
  return { data, error, setData };
}

function App() {
  const [route, setRoute] = useState(location.pathname === "/" ? "/dashboard" : location.pathname);
  const [dataset, set数据集] = useState("WaDI_A2_ds10");
  const [eventId, set事件Id] = useState(1);
  const [job, setJob] = useState<any>(null);

  useEffect(() => {
    if (!job || ["succeeded", "failed"].includes(job.status)) return;
    const t = setInterval(() => getJson(`/api/jobs/${job.job_id}`).then(setJob), 1600);
    return () => clearInterval(t);
  }, [job]);

  useEffect(() => {
    const onPop = () => setRoute(location.pathname === "/" ? "/dashboard" : location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const nav = (path: string) => { history.pushState({}, "", path); setRoute(path); };
  const train = async () => setJob(await postJson("/api/jobs/train", { dataset, epochs: 1, max_train_windows: 1000, eval_stride: 8, edge_mode: "full", use_relation_degradation: true }));

  return <div className="app">
    <TopBar dataset={dataset} job={job} />
    <Sidebar route={route} nav={nav} />
    <main className="workspace">
      <Toolbar dataset={dataset} set数据集={set数据集} train={train} job={job} />
      {route === "/dashboard" && <Dashboard dataset={dataset} eventId={eventId} set事件Id={set事件Id} />}
      {route === "/relations" && <Relations dataset={dataset} eventId={eventId} />}
      {route === "/root-cause" && <RootCause dataset={dataset} eventId={eventId} />}
      {route === "/diagnosis" && <Diagnosis dataset={dataset} eventId={eventId} />}
      {route === "/knowledge" && <Knowledge />}
      {route === "/report" && <报告 dataset={dataset} eventId={eventId} />}
      {route === "/history" && <DiagnosisHistory />}
    </main>
  </div>;
}

function TopBar({ dataset, job }: any) {
  return <header className="topbar">
    <div className="brand-mark">RE</div>
    <div><div className="brand">R-EVGAT Agent</div><div className="subtitle">工业时序异常诊断工作台</div></div>
    <div className="top-spacer" />
    <span className="pill">{dataset}</span>
    <span className="pill alert">可解释边向量</span>
    <span className="agent-online">规则 Agent 就绪</span>
    {job && <span className={`job ${job.status}`}>训练 {statusLabel(job.status)}</span>}
  </header>;
}

function Sidebar({ route, nav }: any) {
  const items = [["/dashboard", "实时监控"], ["/relations", "关系退化"], ["/root-cause", "根因分析"], ["/diagnosis", "诊断 Agent"], ["/knowledge", "知识库"], ["/report", "诊断报告"], ["/finetune", "模型微调"], ["/history", "查询记录"]];
  return <aside className="sidebar">
    {items.map(([p, label]) => <button key={p} className={route === p ? "active" : ""} onClick={() => nav(p)}>{label}</button>)}
    <div className="model-card"><span>系统架构</span><strong>Agent + RAG</strong><small>ChromaDB 配置</small><small>关键词兜底检索</small></div>
  </aside>;
}

function Toolbar({ dataset, set数据集, train, job }: any) {
  const ds = useData("/api/datasets", []);
  return <section className="toolbar">
    <div><label>数据集</label><select value={dataset} onChange={(e) => set数据集(e.target.value)}>{(ds.data?.datasets || [{ id: dataset }]).map((d: any) => <option key={d.id} value={d.id}>{d.id}</option>)}</select></div>
    <button onClick={train} disabled={job?.status === "queued" || job?.status === "running"}>启动轻量训练</button>
    <div className="upload-placeholder">CSV 上传占位：字段映射 → 正常参考段 → 训练任务</div>
  </section>;
}

function Dashboard({ dataset, eventId, set事件Id }: any) {
  const ov = useData(`/api/overview?dataset=${dataset}`, [dataset]);
  const ts = useData(`/api/timeseries?dataset=${dataset}`, [dataset]);
  if (ov.error || ts.error) return <State text={ov.error || ts.error} />;
  if (!ov.data || !ts.data) return <State text="正在加载 Relation-EVGAT 输出..." />;
  return <div className="page-grid with-agent">
    <section className="metrics"><Metric title="数据集" value={dataset} note={`${ts.data.sensors.length} sensors`} /><Metric title="事件" value={`#${eventId}`} note="当前选中报警" /><Metric title="异常分数" value={ov.data.current_score.toFixed(2)} note={`阈值 ${ov.data.threshold.toFixed(2)}`} /><Metric title="状态" value={ov.data.alert ? "报警" : "正常"} note={`${ov.data.num_events} events`} accent={ov.data.alert} /></section>
    <section className="panel large"><PanelTitle title="多变量传感器曲线" note="窗口来自本项目内置样例数据" /><MultiLineChart data={ts.data.points} sensors={ts.data.sensors.slice(0, 5)} /></section>
    <section className="panel"><PanelTitle title="异常分数时间线" note="Relation-EVGAT 节点-边联合分数" /><LineChart data={ov.data.series.map((x: any) => ({ x: x.time, y: x.score, label: x.label }))} threshold={ov.data.threshold} /><div className="event-row">{ov.data.events.slice(0, 8).map((e: any) => <button key={e.event_id} className={e.event_id === eventId ? "active" : ""} onClick={() => set事件Id(e.event_id)}>事件 #{e.event_id}</button>)}</div></section>
    <AgentPanel dataset={dataset} eventId={eventId} />
  </div>;
}

function Relations({ dataset, eventId }: any) {
  const g = useData(`/api/relation-graph?dataset=${dataset}&event_id=${eventId}`, [dataset, eventId]);
  if (g.error) return <State text={g.error} />;
  if (!g.data) return <State text="正在加载关系图..." />;
  return <div className="page-grid with-agent">
    <section className="panel relation-panel"><PanelTitle title="传感器关系图" note="边表示关系退化证据，不直接等同严格因果链" /><RelationSvg graph={g.data} /></section>
    <section className="panel top-degraded"><PanelTitle title="Top 退化边" note="边退化分数" />{g.data.top_edges.map((e: any) => <BarLine key={`${e.source}-${e.target}`} label={`${e.source} -> ${e.target}`} value={e.degradation} />)}</section>
    <section className="panel edge-vector"><PanelTitle title="边向量对比" note="正常窗口 vs 异常窗口边向量" /><div className="vector-grid">{g.data.edge_vector_compare.map((x: any) => <div key={x.component}><strong>{x.component}</strong><BarLine label="正常" value={x.normal} calm /><BarLine label="异常" value={x.anomaly} /></div>)}</div></section>
    <AgentPanel dataset={dataset} eventId={eventId} />
  </div>;
}

function RootCause({ dataset, eventId }: any) {
  const rc = useData(`/api/root-cause?dataset=${dataset}&event_id=${eventId}`, [dataset, eventId]);
  const ts = useData(`/api/timeseries?dataset=${dataset}`, [dataset]);
  if (rc.error || ts.error) return <State text={rc.error || ts.error} />;
  if (!rc.data || !ts.data) return <State text="正在加载根因排序..." />;
  return <div className="page-grid with-agent">
    <section className="panel root-list"><PanelTitle title="Top-K 根因候选" note="结合节点误差与边退化证据排序" />{rc.data.candidates.slice(0, 7).map((c: any) => <BarLine key={c.name} label={`${c.rank}. ${c.name}`} value={c.normalized} />)}</section>
    <section className="panel evidence"><PanelTitle title="证据卡片" note="节点 + 边 + 异常窗口" />{rc.data.evidence.map((e: any) => <div className="evidence-card" key={e.label}><span>{e.label}</span><strong>{e.value}</strong></div>)}</section>
    <section className="panel history-panel"><PanelTitle title="候选变量历史曲线" note="正常窗口与异常窗口对比" /><MultiLineChart data={ts.data.points} sensors={ts.data.sensors.slice(0, 4)} /></section>
    <AgentPanel dataset={dataset} eventId={eventId} />
  </div>;
}

function Diagnosis({ dataset, eventId }: any) {
  const [question, set诊断问题] = useState("为什么报警？请给出根因、关系退化证据和排查步骤。");
  const [task, setTask] = useState<any>(null);
  const [thinking, setThinking] = useState<string[]>([]);
  const [reportStream, set报告Stream] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!task?.task_id || ["completed", "failed"].includes(task.status)) return;
    const t = setInterval(() => getJson(`/api/diagnosis/tasks/${task.task_id}`).then(setTask), 1200);
    return () => clearInterval(t);
  }, [task]);

  useEffect(() => {
    if (!task?.task_id) return;
    setThinking(task.thinking_chunks || []);
    set报告Stream(task.report_chunks || []);
    const thinkingSource = new EventSource(`${API}/api/diagnosis/tasks/${task.task_id}/thinking/stream`);
    const reportSource = new EventSource(`${API}/api/diagnosis/tasks/${task.task_id}/report/stream`);
    const addThinking = (e: MessageEvent) => { const data = JSON.parse(e.data); if (data.text) setThinking((prev) => [...prev, data.text]); };
    const add报告 = (e: MessageEvent) => { const data = JSON.parse(e.data); if (data.text) set报告Stream((prev) => [...prev, data.text]); };
    ["thinking", "tool", "error"].forEach((name) => thinkingSource.addEventListener(name, addThinking));
    ["report", "error"].forEach((name) => reportSource.addEventListener(name, add报告));
    return () => { thinkingSource.close(); reportSource.close(); };
  }, [task?.task_id]);

  const start = async () => {
    setBusy(true); setThinking([]); set报告Stream([]);
    try { setTask(await postJson("/api/diagnosis/tasks", { dataset, event_id: eventId, question, use_llm: true })); }
    finally { setBusy(false); }
  };

  const stages = ["created", "detecting", "evidence_collecting", "retrieving_knowledge", "reasoning", "reporting", "completed"];
  const activeIndex = Math.max(0, stages.indexOf(task?.stage || "created"));
  const calls = task?.tool_calls || task?.result?.tool_calls || [];

  return <div className="diagnosis-layout">
    <section className="panel diagnosis-command"><PanelTitle title="诊断任务" note="先调用 Relation-EVGAT 工具链，再由 DashScope 大模型生成诊断" />
      <label className="field-label">诊断问题</label><textarea value={question} onChange={(e) => set诊断问题(e.target.value)} />
      <div className="diagnosis-actions"><button onClick={start} disabled={busy || task?.status === "running"}>开始诊断</button><span>{task ? `任务 ${task.task_id.slice(0, 8)} · ${statusLabel(task.status)}` : "暂无任务"}</span></div>
      <div className="stage-track">{stages.map((stage, idx) => <div key={stage} className={idx <= activeIndex ? "done" : ""}><i />{stageLabel(stage)}</div>)}</div>
    </section>
    <section className="panel thinking-panel"><PanelTitle title="Agent 思考流" note="阶段事件与工具执行更新" /><LogList rows={thinking} empty="等待任务流输出..." /></section>
    <section className="panel tool-panel"><PanelTitle title="工具调用日志" note="工业诊断工具" />{calls.length ? calls.map((c: any) => <div className="tool-row" key={c.name}><strong>{toolLabel(c.name)}</strong><span>{statusLabel(c.status)}</span><small>{c.duration_ms || 0} ms</small></div>) : <StateCompact text="任务启动后会显示工具调用。" />}</section>
      <section className="panel final-panel"><PanelTitle title="报告生成流" note="包含根因与排查建议的最终回答" /><LogList rows={reportStream} empty="报告流尚未生成。" /></section>
      <OCRDocPanel dataset={dataset} eventId={eventId} />
  </div>;
}

function Knowledge() {
  const docs = useData("/api/knowledge/documents", []);
  const [filename, set文件名] = useState("wadi_operator_note.md");
  const [content, set文档内容] = useState("WaDI 运维记录：当泵、液位等变量附近出现关系退化时，需要在同一异常窗口内核对传感器校准、泵指令状态、阀门反馈以及上下游水位趋势。");
  const [query, setQuery] = useState("WaDI relation degradation pump level 排查");
  const [hits, setHits] = useState<any[]>([]);
  const [message, setMessage] = useState("");
  const upload = async () => { const res = await postJson("/api/knowledge/upload", { filename, content }); setMessage(`${res.status}: ${res.filename}`); docs.setData(await getJson("/api/knowledge/documents")); };
  const search = async () => { const res = await postJson("/api/knowledge/search", { query, top_k: 5 }); setHits(res.hits || []); };
  return <div className="knowledge-layout">
    <section className="panel kb-status"><PanelTitle title="向量库配置" note="支持 ChromaDB，默认关键词检索兜底" />
      <div className="config-grid">{Object.entries(docs.data?.status || {}).slice(0, 8).map(([k, v]) => <div key={k}><span>{k}</span><strong>{typeof v === "object" ? JSON.stringify(v) : String(v)}</strong></div>)}</div>
    </section>
    <section className="panel kb-upload"><PanelTitle title="上传知识" note="演示版支持 JSON 文本上传，后端也可接收原始文本" /><label className="field-label">文件名</label><input value={filename} onChange={(e) => set文件名(e.target.value)} /><label className="field-label">文档内容</label><textarea value={content} onChange={(e) => set文档内容(e.target.value)} /><div className="diagnosis-actions"><button onClick={upload}>索引文档</button><span>{message}</span></div></section>
    <section className="panel kb-docs"><PanelTitle title="已索引文档" note="本地持久化知识切片" />{(docs.data?.documents || []).map((d: any) => <div className="doc-row" key={d.doc_id}><strong>{d.filename}</strong><span>{d.chunk_count} 个切片</span><small>{d.doc_id.slice(0, 10)}</small></div>)}</section>
    <section className="panel kb-search"><PanelTitle title="检索预览" note="诊断 Agent 会调用该检索结果" /><div className="search-row"><input value={query} onChange={(e) => setQuery(e.target.value)} /><button onClick={search}>检索</button></div>{hits.map((h) => <article className="hit" key={h.chunk_id}><strong>{h.title}</strong><small>分数 {h.score.toFixed(4)}</small><p>{h.text}</p></article>)}</section>
  </div>;
}

function 报告({ dataset, eventId }: any) {
  const rep = useData(`/api/report?dataset=${dataset}&event_id=${eventId}`, [dataset, eventId]);
  const [answer, setAnswer] = useState("");
  const [calls, setCalls] = useState<any[]>([]);
  const ask = async (q: string) => { const r = await postJson("/api/agent/ask", { dataset, event_id: eventId, question: q }); setAnswer(r.answer); setCalls(r.tool_calls); };
  if (rep.error) return <State text={rep.error} />;
  if (!rep.data) return <State text="正在加载报告..." />;
  const handleExportPdf = () => { window.print(); };
  return <div className="report-layout"><section className="panel chat-main"><PanelTitle title="诊断问答" note="快捷问题" /><div className="quick">{["为什么报警？", "最可疑变量", "生成报告", "排查步骤"].map((q) => <button key={q} onClick={() => ask(q)}>{q}</button>)}</div><div className="bubble user">这个事件为什么报警？</div><div className="bubble agent"><strong>Agent 诊断</strong><p>{answer || rep.data.sections.map((s: any) => s.body).join(" ")}</p></div><div className="tool-log">{(calls.length ? calls : ["get_event_summary", "rank_root_causes", "inspect_edge_degradation", "generate_report"].map((name) => ({ name, status: "ready" }))).map((c: any) => <span key={c.name}>{toolLabel(c.name)}：{statusLabel(c.status)}</span>)}</div></section><section className="panel report-main" id="printable-report"><PanelTitle title="自动生成诊断报告" note={`事件 ID: ${rep.data.event_id}   时间窗口：${rep.data.time_window}`} />{rep.data.sections.map((s: any) => <article className="report-card" key={s.title}><h3>{s.title}</h3><p>{s.body}</p></article>)}<div className="report-actions no-print"><button onClick={handleExportPdf}>导出 PDF</button><button>工具调用日志</button></div></section></div>;
}

function DiagnosisHistory() {
  const [tab, setTab] = useState<"diagnosis" | "document">("diagnosis");
  const [tasks, setTasks] = useState<any[]>([]);
  const [docs, setDocs] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const fetchTasks = () => {
    getJson("/api/diagnosis/history").then((d) => setTasks(d.tasks || [])).catch(() => {});
  };
  const fetchDocs = () => {
    getJson("/api/document/history").then((d) => setDocs(d.documents || [])).catch(() => {});
  };

  useEffect(() => { fetchTasks(); fetchDocs(); }, []);

  const loadTask = async (taskId: string) => {
    setLoading(true);
    try { const t = await getJson(`/api/diagnosis/tasks/${taskId}`); setSelected(t); }
    catch { setSelected(null); }
    finally { setLoading(false); }
  };

  const loadDoc = async (docId: string) => {
    setLoading(true);
    try { const r = await getJson(`/api/document/history/${docId}`); setSelected(r); }
    catch { setSelected(null); }
    finally { setLoading(false); }
  };

  const deleteDoc = async (docId: string) => {
    await fetch(`${API}/api/document/history/${docId}`, { method: "DELETE" });
    if (selected?.doc_id === docId) setSelected(null);
    fetchDocs();
  };

  const formatDate = (ts: number) => {
    const d = new Date(ts * 1000);
    return `${d.getFullYear()}-${(d.getMonth() + 1).toString().padStart(2, "0")}-${d.getDate().toString().padStart(2, "0")} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  };

  const stLabels: Record<string, string> = { completed: "已完成", failed: "失败", running: "运行中" };

  const isDocTab = tab === "document";
  const list = isDocTab ? docs : tasks;
  const listTitle = isDocTab ? "文档智检记录" : "诊断查询记录";
  const listNote = `共 ${list.length} 条`;
  const emptyText = isDocTab ? "暂无记录，请先在「诊断 Agent」页面上传文档。" : "暂无诊断记录，请先在「诊断 Agent」中执行任务。";

  const info = !isDocTab ? null : (selected?.industrial_info || {});
  const params = (info && info.process_params) || {};

  return <div className="history-layout" id="printable-history">
    <section className="panel history-list-panel no-print-tabs">
      <div className="history-tabs">
        <button className={!isDocTab ? "active" : ""} onClick={() => { setTab("diagnosis"); setSelected(null); }}>诊断记录</button>
        <button className={isDocTab ? "active" : ""} onClick={() => { setTab("document"); setSelected(null); }}>文档记录</button>
      </div>
      <PanelTitle title={listTitle} note={listNote} />
      <div className="history-query-list">
        {list.length === 0 && <StateCompact text={emptyText} />}
        {isDocTab
          ? docs.map((d: any) => (
            <div key={d.doc_id} className={`history-query-row ${selected?.doc_id === d.doc_id ? "active" : ""}`} onClick={() => loadDoc(d.doc_id)}>
              <div className="hq-top">
                <span className="hq-question">{d.filename}</span>
                <button className="history-del" onClick={(e) => { e.stopPropagation(); deleteDoc(d.doc_id); }} title="删除">×</button>
              </div>
              <div className="hq-meta">{formatDate(d.created_at)} · {d.text_length} 字符{d.has_info ? " · 已抽取" : ""}</div>
              {d.text_preview && <div className="hq-preview">{d.text_preview}...</div>}
            </div>
          ))
          : tasks.map((t: any) => (
            <div key={t.task_id} className={`history-query-row ${selected?.task_id === t.task_id ? "active" : ""}`} onClick={() => loadTask(t.task_id)}>
              <div className="hq-top">
                <span className="hq-question">{t.question}</span>
                <span className={`hq-status ${t.status}`}>{stLabels[t.status] || t.status}</span>
              </div>
              <div className="hq-meta">{formatDate(t.created_at)} · {t.dataset} · 事件 #{t.event_id || "-"}{t.use_llm ? " · LLM" : ""}</div>
            </div>
          ))}
      </div>
    </section>

    <section className="panel history-detail-panel" id="printable-detail">
      <PanelTitle title={isDocTab ? "文档详情" : "问答详情"} note={selected ? (isDocTab ? selected.filename : `任务 ${selected.task_id?.slice(0, 10)}...`) : "点击左侧记录查看"} />
      {loading && <State text="加载中..." />}
      {!loading && selected && !isDocTab && (
        <div className="history-detail">
          <div className="detail-block"><div className="detail-label">诊断问题</div><div className="detail-value question">{selected.question}</div></div>
          <div className="detail-block"><div className="detail-label">数据集 / 事件</div><div className="detail-value">{selected.dataset} · 事件 #{selected.event_id || "-"} · {selected.use_llm ? "LLM 推理" : "规则引擎"}</div></div>
          <div className="detail-block"><div className="detail-label">诊断回答</div><div className="detail-value answer">{selected.result?.answer || selected.report_chunks?.join("") || "无回答内容"}</div></div>
          {selected.tool_calls?.length > 0 && (
            <div className="detail-block"><div className="detail-label">工具调用</div>
              <div className="detail-calls">{selected.tool_calls.map((c: any) => <span key={c.name} className={c.status}>{toolLabel(c.name)} · {statusLabel(c.status)}</span>)}</div>
            </div>
          )}
        </div>
      )}
      {!loading && selected && isDocTab && (
        <div className="history-detail">
          <div className="detail-block"><div className="detail-label">文件名</div><div className="detail-value question">{selected.filename}</div></div>
          <div className="detail-block"><div className="detail-label">上传时间</div><div className="detail-value">{formatDate(selected.created_at)}</div></div>
          {info && (Object.keys(params).length > 0 || info.defect_type || info.material) && (
            <div className="detail-block"><div className="detail-label">关键工业信息</div>
              <div className="info-grid" style={{marginTop: 4}}>
                {info.material && <div className="info-item"><span>材料/部件</span><strong>{info.material}</strong></div>}
                {info.defect_type && <div className="info-item"><span>缺陷类型</span><strong className="defect">{info.defect_type}</strong></div>}
                {info.defect_location && <div className="info-item"><span>缺陷位置</span><strong>{info.defect_location}</strong></div>}
                {info.defect_severity && <div className="info-item"><span>严重程度</span><strong>{info.defect_severity}</strong></div>}
                {info.batch_number && <div className="info-item"><span>批次号</span><strong>{info.batch_number}</strong></div>}
                {info.inspection_result && <div className="info-item"><span>检测结论</span><strong className={info.inspection_result?.includes("不合格") ? "defect" : ""}>{info.inspection_result}</strong></div>}
                {info.summary && <div className="info-item full"><span>摘要</span><strong>{info.summary}</strong></div>}
              </div>
              {Object.keys(params).length > 0 && <div className="param-grid" style={{marginTop: 8}}>{Object.entries(params).map(([k, v]: [string, any]) => <div className="info-item" key={k}><span>{k}</span><strong>{String(v)}</strong></div>)}</div>}
            </div>
          )}
          <div className="detail-block"><div className="detail-label">文档全文</div><div className="detail-value answer">{selected.doc_text?.substring(0, 2000) || "无内容"}{selected.doc_text?.length > 2000 ? "..." : ""}</div></div>
          {selected.cross_modal_analysis && (
            <div className="detail-block"><div className="detail-label">跨模态分析结果</div><div className="detail-value answer">{selected.cross_modal_analysis}</div></div>
          )}
          <div className="no-print"><button className="detail-delete-btn" onClick={() => deleteDoc(selected.doc_id)}>删除此记录</button></div>
        </div>
      )}
      {!loading && !selected && <StateCompact text={isDocTab ? "选择一条记录查看文档详情。" : "选择一条记录查看当时的诊断问题和回答。"} />}
      <div className="no-print" style={{marginTop: 16}}><button onClick={() => window.print()} className="print-btn">打印</button></div>
    </section>
  </div>;
}

function OCRDocPanel({ dataset, eventId }: any) {
  const [docText, setDocText] = useState("");
  const [docId, setDocId] = useState("");
  const [paragraphsCount, setParagraphsCount] = useState(0);
  const [industrialInfo, setIndustrialInfo] = useState<any>(null);
  const [crossAnalysis, setCrossAnalysis] = useState("");
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [statusMsg, setStatusMsg] = useState("上传工业文档（Word / PDF，质检报告、工艺卡等），自动提取文字并抽取工艺参数、缺陷描述");
  const [history, setHistory] = useState<any[]>([]);

  const fetchHistory = async () => {
    try {
      const res = await getJson("/api/document/history");
      setHistory(res.documents || []);
    } catch { /* 忽略 */ }
  };

  useEffect(() => { fetchHistory(); }, []);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith(".docx") && !file.name.endsWith(".doc") && !file.name.endsWith(".pdf")) {
      setStatusMsg("仅支持 .docx、.doc 或 .pdf 文件");
      return;
    }
    setUploading(true);
    setFileName(file.name);
    setStatusMsg("文档文字提取中...");
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = (reader.result as string);
      try {
        const res = await postJson("/api/document/extract-info", { file_base64: base64, filename: file.name });
        setDocId(res.doc_id || "");
        setDocText(res.doc_text || "");
        setParagraphsCount(res.paragraphs_count || 0);
        setIndustrialInfo(res.industrial_info || {});
        setCrossAnalysis(res.cross_modal_analysis || "");
        setStatusMsg(`提取完成：${res.doc_text?.length || 0} 字符，已抽取关键信息`);
        fetchHistory();
      } catch (err) {
        setStatusMsg("提取失败：" + String(err));
        setDocText("");
        setDocId("");
        setIndustrialInfo(null);
      } finally {
        setUploading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const loadFromHistory = async (docId: string, fname: string) => {
    setUploading(true);
    setStatusMsg("加载历史文档...");
    try {
      const res = await getJson(`/api/document/history/${docId}`);
      setDocId(res.doc_id || "");
      setDocText(res.doc_text || "");
      setParagraphsCount(res.doc_text?.length || 0);
      setIndustrialInfo(res.industrial_info || {});
      setCrossAnalysis(res.cross_modal_analysis || "");
      setFileName(fname);
      setStatusMsg(`已加载：${fname}`);
    } catch (err) {
      setStatusMsg("加载失败：" + String(err));
    } finally {
      setUploading(false);
    }
  };

  const deleteFromHistory = async (docId: string) => {
    try {
      await fetch(`${API}/api/document/history/${docId}`, { method: "DELETE" });
      fetchHistory();
    } catch { /* 忽略 */ }
  };

  const runCrossModal = async () => {
    if (!docText) return;
    setStatusMsg("跨模态关联分析中...");
    try {
      const res = await postJson("/api/agent/cross-modal", {
        dataset,
        event_id: eventId,
        doc_text: docText,
        doc_info: industrialInfo || {},
      });
      const analysis = res.analysis || "";
      setCrossAnalysis(analysis);
      setStatusMsg("跨模态关联分析完成");
      if (docId && analysis) {
        try {
          await fetch(`${API}/api/document/history/${docId}/analysis`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ analysis }),
          });
          fetchHistory();
        } catch { /* 静默保存 */ }
      }
    } catch (err) {
      setStatusMsg("跨模态分析失败：" + String(err));
    }
  };

  const info = industrialInfo || {};
  const params = info.process_params || {};

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  };

  return <section className="panel ocr-panel">
    <PanelTitle title="工厂文档智检" note="上传质检报告/工艺卡（Word / PDF） → 文字提取 → ErnieBot 抽取工艺参数、缺陷描述 → 跨模态关联传感器异常" />
    <div className="ocr-upload-row">
      <label className="upload-btn">{uploading ? "处理中..." : "选择文档"}
        <input type="file" accept=".docx,.doc,.pdf" onChange={handleFile} disabled={uploading} style={{display:"none"}} />
      </label>
      <span className="ocr-status">{statusMsg}</span>
      {fileName && <span className="file-name-tag">{fileName}</span>}
    </div>

    {history.length > 0 && <div className="ocr-history">
      <div className="ocr-section-title">历史上传</div>
      <div className="history-list">
        {history.slice(0, 8).map((h: any) => (
          <div className="history-row" key={h.doc_id}>
            <span className="history-name" onClick={() => loadFromHistory(h.doc_id, h.filename)} title={h.text_preview}>
              {h.filename}
            </span>
            <span className="history-meta">{formatTime(h.created_at)} · {h.text_length} 字符{h.has_info ? " · 已抽取" : ""}</span>
            <button className="history-del" onClick={() => deleteFromHistory(h.doc_id)} title="删除">×</button>
          </div>
        ))}
      </div>
    </div>}

    {docText && <div className="ocr-result-section">
      <div className="ocr-section-title">提取的文字内容</div>
      <pre className="ocr-text">{docText.substring(0, 800)}{docText.length > 800 ? "..." : ""}</pre>
    </div>}

    {industrialInfo && (Object.keys(params).length > 0 || info.defect_type || info.material) && <div className="ocr-result-section">
      <div className="ocr-section-title">ErnieBot 抽取的关键工业信息</div>
      <div className="info-grid">
        {info.material && <div className="info-item"><span>材料/部件</span><strong>{info.material}</strong></div>}
        {info.defect_type && <div className="info-item"><span>缺陷类型</span><strong className="defect">{info.defect_type}</strong></div>}
        {info.defect_location && <div className="info-item"><span>缺陷位置</span><strong>{info.defect_location}</strong></div>}
        {info.defect_severity && <div className="info-item"><span>严重程度</span><strong>{info.defect_severity}</strong></div>}
        {info.batch_number && <div className="info-item"><span>批次号</span><strong>{info.batch_number}</strong></div>}
        {info.inspection_result && <div className="info-item"><span>检测结论</span><strong className={info.inspection_result.includes("不合格") ? "defect" : ""}>{info.inspection_result}</strong></div>}
        {info.summary && <div className="info-item full"><span>摘要</span><strong>{info.summary}</strong></div>}
      </div>
      {Object.keys(params).length > 0 && <div className="param-grid">
        <div className="ocr-section-title" style={{marginTop:8}}>工艺参数</div>
        {Object.entries(params).map(([k, v]) => <div className="info-item" key={k}><span>{k}</span><strong>{String(v)}</strong></div>)}
      </div>}
    </div>}

    {docText && <div className="ocr-actions">
      <button onClick={runCrossModal}>跨模态关联分析</button>
    </div>}

    {crossAnalysis && <div className="ocr-result-section">
      <div className="ocr-section-title">跨模态关联分析结果</div>
      <pre className="ocr-text cross-modal">{crossAnalysis}</pre>
    </div>}
  </section>;
}

function AgentPanel({ dataset, eventId }: any) {
  const [answer, setAnswer] = useState("我可以为当前事件调用异常摘要、根因排序、关系退化、知识库检索和报告生成工具。");
  const ask = async (q: string) => { const r = await postJson("/api/agent/ask", { dataset, event_id: eventId, question: q }); setAnswer(r.answer); };
  return <aside className="agent-panel"><h2>诊断 Agent</h2><p>异常摘要 / 根因排序 / 关系退化 / RAG / 报告</p><div className="bubble user">用户：为什么报警？</div><div className="bubble agent">{answer}</div><div className="agent-actions"><button onClick={() => ask("生成报告")}>报告</button><button onClick={() => ask("排查步骤")}>排查步骤</button></div></aside>;
}

function Metric({ title, value, note, accent }: any) { return <div className={`metric ${accent ? "accent" : ""}`}><span>{title}</span><strong>{value}</strong><small>{note}</small></div>; }
function PanelTitle({ title, note }: any) { return <div className="panel-title"><h2>{title}</h2><p>{note}</p></div>; }
function State({ text }: any) { return <div className="state">{text}</div>; }
function StateCompact({ text }: any) { return <div className="state compact">{text}</div>; }
function LogList({ rows, empty }: any) { return <div className="log-list">{rows.length ? rows.map((row: string, i: number) => <p key={`${i}-${row}`}>{row}</p>) : <StateCompact text={empty} />}</div>; }
function BarLine({ label, value, calm }: any) { return <div className="bar-line"><span>{label}</span><div className="bar-bg"><div className={calm ? "bar-fill calm" : "bar-fill"} style={{ width: `${Math.max(3, Math.min(100, value * 100))}%` }} /></div><strong>{Number(value).toFixed(2)}</strong></div>; }

function MultiLineChart({ data, sensors }: any) {
  const colors = ["#1f6f8b", "#bd4f38", "#4f7b45", "#8b5fbf", "#c78a1f"];
  return <div><svg className="chart" viewBox="0 0 820 260" preserveAspectRatio="none"><Grid />{sensors.map((sensor: string, idx: number) => <polyline key={sensor} fill="none" stroke={colors[idx % colors.length]} strokeWidth="2" points={scalePoints(data.map((r: any) => ({ x: Number(r.time), y: Number(r[sensor]) })), 820, 260)} />)}</svg><div className="legend-row">{sensors.map((s: string, i: number) => <span key={s}><i style={{ background: colors[i % colors.length] }} />{s}</span>)}</div></div>;
}

function LineChart({ data, 阈值 }: any) { return <svg className="chart small" viewBox="0 0 820 180" preserveAspectRatio="none"><Grid />{阈值 !== undefined && <line x1="0" x2="820" y1={180 - 阈值 * 150} y2={180 - 阈值 * 150} stroke="#d4911f" strokeDasharray="6 5" />}<polyline fill="none" stroke="#7d3c98" strokeWidth="2" points={scalePoints(data, 820, 180)} />{data.filter((d: any) => d.label).slice(0, 80).map((d: any, i: number) => { const [x, y] = pointToXY(d, data, 820, 180); return <circle key={i} cx={x} cy={y} r="2.8" fill="#d64b4b" opacity="0.55" />; })}</svg>; }

function RelationSvg({ graph }: any) {
  const map = new Map(graph.nodes.map((n: any) => [n.id, n]));
  return <svg className="relation-svg" viewBox="0 0 820 480">{graph.edges.map((e: any) => { const a: any = map.get(e.source); const b: any = map.get(e.target); return a && b ? <line key={`${e.source}-${e.target}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={e.degradation > 0.7 ? "#d64b4b" : "#7d93a6"} strokeWidth={2 + e.degradation * 5} opacity="0.8" /> : null; })}{graph.nodes.map((n: any) => <g key={n.id}><circle cx={n.x} cy={n.y} r={28 + n.分数 * 8} fill="#f8fbfd" stroke={n.分数 > 0.7 ? "#d64b4b" : "#4f7b8f"} strokeWidth="3" /><text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="13" fontWeight="700" fill="#22313d">{n.label}</text></g>)}</svg>;
}

function Grid() { return <g opacity="0.45">{[0, 1, 2, 3, 4, 5].map((i) => <line key={`h${i}`} x1="0" x2="820" y1={20 + i * 38} y2={20 + i * 38} stroke="#d6e0e6" />)}{[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => <line key={`v${i}`} y1="12" y2="245" x1={20 + i * 95} x2={20 + i * 95} stroke="#d6e0e6" />)}</g>; }
function scalePoints(points: any[], width: number, height: number) { if (!points.length) return ""; const xs = points.map((p) => p.x); const ys = points.map((p) => p.y); const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys); return points.map((p) => { const x = ((p.x - minX) / (maxX - minX || 1)) * (width - 40) + 20; const y = height - (((p.y - minY) / (maxY - minY || 1)) * (height - 40) + 20); return `${x.toFixed(1)},${y.toFixed(1)}`; }).join(" "); }
function pointToXY(point: any, points: any[], width: number, height: number) { const xs = points.map((p) => p.x); const ys = points.map((p) => p.y); const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys); const x = ((point.x - minX) / (maxX - minX || 1)) * (width - 40) + 20; const y = height - (((point.y - minY) / (maxY - minY || 1)) * (height - 40) + 20); return [x, y]; }

createRoot(document.getElementById("root")!).render(<App />);



