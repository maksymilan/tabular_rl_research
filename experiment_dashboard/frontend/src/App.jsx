import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  Bot,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Clock3,
  Cpu,
  Database,
  FileJson,
  FlaskConical,
  Gauge,
  HardDrive,
  Menu,
  PanelLeftClose,
  Play,
  Power,
  RefreshCw,
  Save,
  Server,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AttributionPanel,
  attributeRecord,
  BUCKET_META,
  ConstructionPanel,
  TrainingBoard,
  TrajectoryView,
} from "./Trajectory";
import { StructuredRecord } from "./JsonViewer";

const api = {
  async request(path, options) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
    return payload;
  },
  experiments: () => api.request("/api/experiments"),
  records: (id, source, page, pageSize = 8) =>
    api.request(
      `/api/experiments/${id}/records?source=${source}&page=${page}&page_size=${pageSize}`,
    ),
  refresh: (id) => api.request(`/api/experiments/${id}/refresh`, { method: "POST", body: "{}" }),
  patch: (id, body) =>
    api.request(`/api/experiments/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  playgroundConfig: () => api.request("/api/playground/config"),
  models: () => api.request("/api/vllm/models"),
  server: () => api.request("/api/vllm/server"),
  startServer: (body) =>
    api.request("/api/vllm/server/start", { method: "POST", body: JSON.stringify(body) }),
  stopServer: () =>
    api.request("/api/vllm/server/stop", { method: "POST", body: "{}" }),
  chat: (body) =>
    api.request("/api/vllm/chat", { method: "POST", body: JSON.stringify(body) }),
};

const pct = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—");
const fixed = (value, digits = 3) =>
  Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
const compact = new Intl.NumberFormat("zh-CN", { notation: "compact", maximumFractionDigits: 1 });
const modelFamily = (model = "") => {
  const match = String(model).match(/Qwen(\d+(?:\.\d+)?)-([^-/\s]+)/i);
  return match ? `Qwen${match[1]}-${match[2]}` : model || "Unknown model";
};
const modelAbbrev = (model = "") => modelFamily(model).replace(/^Qwen/, "Q");
const isDirectSqlBaseline = (item) =>
  item.kind === "baseline" && (item.tags || []).includes("direct-sql");
const duration = (seconds) => {
  if (!Number.isFinite(seconds)) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
};
const dateTime = (value) =>
  value
    ? new Intl.DateTimeFormat("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }).format(new Date(value))
    : "—";
function StatusDot({ status }) {
  return (
    <span className={`status status-${status}`}>
      <span />
      {status === "completed" ? "已完成" : status === "running" ? "训练中" : status}
    </span>
  );
}

function Metric({ label, value, detail, icon: Icon }) {
  return (
    <div className="metric">
      <div className="metric-icon">{Icon ? <Icon size={17} /> : null}</div>
      <div>
        <div className="metric-value">{value}</div>
        <div className="metric-label">{label}</div>
        {detail ? <div className="metric-detail">{detail}</div> : null}
      </div>
    </div>
  );
}

function Empty({ title, detail }) {
  return (
    <div className="empty">
      <CircleAlert size={22} />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function Overview({ experiments, onOpen }) {
  const families = useMemo(() => {
    const groups = new Map();
    for (const item of experiments) {
      const family = modelFamily(item.model);
      if (!groups.has(family)) groups.set(family, []);
      groups.get(family).push(item);
    }
    return [...groups.entries()]
      .map(([family, items]) => ({
        family,
        items,
        directSql: items.find(isDirectSqlBaseline),
      }))
      .sort((a, b) => a.family.localeCompare(b.family, "zh-CN", { numeric: true }));
  }, [experiments]);
  const preferredFamily = useMemo(
    () => families.find((group) => group.family.startsWith("Qwen3.5"))?.family || families[0]?.family || "",
    [families],
  );
  const [selectedFamily, setSelectedFamily] = useState("");
  useEffect(() => {
    if (!families.length) return;
    if (!selectedFamily || !families.some((group) => group.family === selectedFamily)) {
      setSelectedFamily(preferredFamily);
    }
  }, [families, preferredFamily, selectedFamily]);
  const activeFamily = selectedFamily || preferredFamily;
  const visibleFamilies = families.filter((group) => group.family === activeFamily);
  const visibleExperiments = visibleFamilies.flatMap((group) => group.items);
  const directSqlByFamily = useMemo(
    () => new Map(visibleFamilies.map((group) => [group.family, group.directSql]).filter(([, item]) => item)),
    [visibleFamilies],
  );
  const chartData = visibleExperiments.map((item) => ({
    name: `${modelAbbrev(item.model)} ${item.short_name}`,
    accuracy: Number(((item.evaluation_summary?.accuracy || 0) * 100).toFixed(2)),
    legal: Number.isFinite(item.evaluation_summary?.legal_rate)
      ? Number((item.evaluation_summary.legal_rate * 100).toFixed(2))
      : null,
    evalLoss: item.training_metrics?.summary?.last_eval_loss,
    kind: item.kind || "sft",
    family: modelFamily(item.model),
  }));
  const trainedRuns = visibleExperiments.filter((item) => item.training_metrics?.available).length;
  const best = [...visibleExperiments].sort(
    (a, b) => (b.evaluation_summary?.accuracy || 0) - (a.evaluation_summary?.accuracy || 0),
  )[0];

  return (
    <main className="content">
      <header className="page-header">
        <div>
          <p className="eyebrow">Research workspace</p>
          <h1>训练实验总览</h1>
          <p>当前展示 {activeFamily || "模型"} 的 baseline、SFT 与工具评测</p>
        </div>
        <div className="header-date">
          <Clock3 size={16} />
          数据更新于本地实验产物
        </div>
      </header>

      <section className="model-switch-panel">
        <div>
          <span>模型家族</span>
          <strong>{activeFamily || "未选择"}</strong>
        </div>
        <div className="segmented-control">
          {families.map((group) => (
            <button
              key={group.family}
              className={group.family === activeFamily ? "active" : ""}
              onClick={() => setSelectedFamily(group.family)}
            >
              {group.family}
              <span>{group.items.length}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="metric-strip">
        <Metric
          label="实验与基线"
          value={visibleExperiments.length}
          detail={`${visibleExperiments.filter((item) => item.kind === "baseline").length} 个基线`}
          icon={FlaskConical}
        />
        <Metric label="SFT 训练" value={trainedRuns} icon={Check} />
        <Metric
          label="最佳执行准确率"
          value={best ? pct(best.evaluation_summary?.accuracy) : "—"}
          detail={best?.short_name}
          icon={Gauge}
        />
        <Metric
          label="累计评测案例"
          value={compact.format(
            visibleExperiments.reduce((sum, item) => sum + (item.evaluation_summary?.total || 0), 0),
          )}
          icon={Database}
        />
      </section>

      <section className="overview-grid">
        <div className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <h2>评测表现</h2>
              <p>执行准确率与合法回答率</p>
            </div>
            <span className="unit">%</span>
          </div>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} barGap={4}>
                <CartesianGrid vertical={false} stroke="#e1e5e0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} axisLine={false} tickLine={false} width={34} />
                <Tooltip contentStyle={{ borderRadius: 6, borderColor: "#cfd5cf" }} />
                <Legend />
                {visibleFamilies
                  .filter((group) => group.directSql?.evaluation_summary?.available)
                  .map((group, index) => (
                    <ReferenceLine
                      key={group.family}
                      y={group.directSql.evaluation_summary.accuracy * 100}
                      stroke={index ? "#7c3aed" : "#b45309"}
                      strokeDasharray={index ? "3 3" : "5 4"}
                      label={{ value: `${group.family} SQL`, fill: index ? "#6d28d9" : "#8a4b09", fontSize: 10 }}
                    />
                  ))}
                <Bar dataKey="accuracy" name="执行准确率" fill="#087f5b" radius={[3, 3, 0, 0]}>
                  {chartData.map((item) => (
                    <Cell key={item.name} fill={item.kind === "baseline" ? "#b45309" : "#087f5b"} />
                  ))}
                </Bar>
                <Bar dataKey="legal" name="合法回答率" fill="#4976a8" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel run-list-panel">
          <div className="panel-heading">
            <div>
              <h2>实验记录</h2>
              <p>按版本查看数据、训练与评测</p>
            </div>
          </div>
          <div className="run-list">
            {visibleFamilies.map((group) => (
              <div className="run-family" key={group.family}>
                <div className="run-family-heading">
                  <strong>{group.family}</strong>
                  <span>
                    {group.directSql ? "含同系列 Direct SQL baseline" : "缺少同系列 Direct SQL baseline"}
                  </span>
                </div>
                {group.items.map((item) => (
                  <button className="run-row" key={item.id} onClick={() => onOpen(item.id)}>
                    <div className={`run-marker ${item.kind === "baseline" ? "baseline" : ""}`} />
                    <div className="run-main">
                      <div className="run-title">
                        <strong>{item.name}</strong>
                        {item.kind === "baseline" ? <span className="type-badge">Baseline</span> : <StatusDot status={item.status} />}
                      </div>
                      <span>{item.dataset_version}</span>
                    </div>
                    <div className="run-score">
                      <strong>{pct(item.evaluation_summary?.accuracy)}</strong>
                      <span>{item.evaluation_summary?.correct || 0}/{item.evaluation_summary?.total || 0}</span>
                    </div>
                    <ChevronRight size={18} />
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel baseline-map">
        <div className="panel-heading">
          <div>
            <h2>Baseline 对齐</h2>
            <p>每个模型家族只和自己的 Direct SQL baseline 对比，避免跨模型误读</p>
          </div>
        </div>
        <div className="baseline-grid">
          {visibleFamilies.map((group) => (
            <div className="baseline-card" key={group.family}>
              <div>
                <strong>{group.family}</strong>
                <span>{group.items.length} 个实验</span>
              </div>
              {group.directSql ? (
                <div className="baseline-score">
                  <span>Direct SQL</span>
                  <strong>{pct(group.directSql.evaluation_summary?.accuracy)}</strong>
                </div>
              ) : (
                <div className="baseline-missing">
                  当前没有同系列 Direct SQL baseline
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="panel comparison-table">
        <div className="panel-heading">
          <div>
            <h2>关键指标对比</h2>
            <p>训练数据规模、loss 与推理表现</p>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>实验</th>
                <th>模型家族</th>
                <th>类型</th>
                <th>训练样本</th>
                <th>Train loss</th>
                <th>Eval loss</th>
                <th>执行准确率</th>
                <th>相对同系列 Direct SQL</th>
                <th>平均步骤</th>
                <th>评测完整性</th>
              </tr>
            </thead>
            <tbody>
              {visibleExperiments.map((item) => {
                const reference = directSqlByFamily.get(modelFamily(item.model));
                return (
                  <tr key={item.id} onClick={() => onOpen(item.id)}>
                    <td><strong>{item.short_name}</strong></td>
                    <td>{modelFamily(item.model)}</td>
                    <td><span className={`type-badge ${item.kind === "baseline" ? "baseline" : ""}`}>{item.kind === "baseline" ? "Baseline" : "SFT"}</span></td>
                    <td>{item.training_metrics?.available ? compact.format(item.dataset_summary?.train?.kept || 0) : "—"}</td>
                    <td>{fixed(item.training_metrics?.summary?.train_loss)}</td>
                    <td>{fixed(item.training_metrics?.summary?.last_eval_loss)}</td>
                    <td>{pct(item.evaluation_summary?.accuracy)}</td>
                    <td>
                      {reference && item.id !== reference.id
                        ? `${((item.evaluation_summary?.accuracy - reference.evaluation_summary.accuracy) * 100).toFixed(2)} pp`
                        : reference ? "reference" : "缺少同系列 baseline"}
                    </td>
                    <td>{fixed(item.evaluation_summary?.average_steps, 2)}</td>
                    <td>
                      {item.evaluation_summary?.complete_dev ? (
                        <span className="inline-ok"><Check size={14} /> 1034/1034</span>
                      ) : (
                        <span className="inline-warn">
                          <CircleAlert size={14} /> {item.evaluation_summary?.total || 0}/1034
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function LossChart({ metrics }) {
  const combined = useMemo(() => {
    const points = new Map();
    for (const point of metrics?.train || []) {
      points.set(point.step, { step: point.step, train: point.loss });
    }
    for (const point of metrics?.eval || []) {
      points.set(point.step, { ...(points.get(point.step) || { step: point.step }), eval: point.loss });
    }
    return [...points.values()].sort((a, b) => a.step - b.step);
  }, [metrics]);

  if (!metrics?.available) return <Empty title="没有训练指标" detail="trainer_state.json 不可用" />;
  return (
    <div className="chart loss-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={combined}>
          <CartesianGrid vertical={false} stroke="#e1e5e0" />
          <XAxis dataKey="step" axisLine={false} tickLine={false} />
          <YAxis axisLine={false} tickLine={false} width={42} />
          <Tooltip contentStyle={{ borderRadius: 6, borderColor: "#cfd5cf" }} />
          <Legend />
          <Line
            type="monotone"
            dataKey="train"
            name="Train loss"
            stroke="#087f5b"
            dot={false}
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="eval"
            name="Eval loss"
            stroke="#d97706"
            dot={{ r: 3 }}
            connectNulls
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function JsonBlock({ value }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

function RecordBrowser({ experiment }) {
  const sources = useMemo(() => experiment.data_sources || [], [experiment.data_sources]);
  const [source, setSource] = useState(sources[0]?.id || "");
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(0);
  const [view, setView] = useState("visual");

  useEffect(() => {
    if (!source) return;
    setData(null);
    setError("");
    api.records(experiment.id, source, page)
      .then((payload) => {
        setData(payload);
        setSelected(0);
      })
      .catch((reason) => setError(reason.message));
  }, [experiment.id, source, page]);

  useEffect(() => {
    if (!sources.some((item) => item.id === source)) {
      setSource(sources[0]?.id || "");
      setPage(1);
    }
  }, [source, sources]);

  const changeSource = (next) => {
    setSource(next);
    setPage(1);
  };
  const active = data?.records?.[selected];

  return (
    <div className="record-browser">
      <div className="source-toolbar">
        <label>
          <span>数据资产</span>
          <select value={source} onChange={(event) => changeSource(event.target.value)}>
            {["dataset", "evaluation", "training", "metadata"].map((group) => {
              const items = sources.filter((item) => item.group === group);
              if (!items.length) return null;
              const labels = {
                dataset: "训练数据",
                evaluation: "评测案例",
                training: "训练状态",
                metadata: "配置与汇总",
              };
              return (
                <optgroup label={labels[group]} key={group}>
                  {items.map((item) => (
                    <option value={item.id} key={item.id}>{item.label}</option>
                  ))}
                </optgroup>
              );
            })}
          </select>
        </label>
        <div className="source-summary">
          <strong>{sources.length}</strong>
          <span>个可浏览数据资产</span>
        </div>
        <div className="view-toggle">
          <button className={view === "visual" ? "active" : ""} onClick={() => setView("visual")}>
            <Activity size={13} /> 可视化
          </button>
          <button className={view === "json" ? "active" : ""} onClick={() => setView("json")}>
            <FileJson size={13} /> 原始 JSON
          </button>
        </div>
      </div>
      {data?.source ? <div className="source-path">{data.source.path}</div> : null}
      {error ? <Empty title="读取失败" detail={error} /> : null}
      {!data && !error ? <div className="loading-line" /> : null}
      {data ? (
        <>
          <div className="record-layout">
            <div className="record-index">
              {data.records.map((item, index) => {
                const isEval = Array.isArray(item.record.turns);
                const bucket =
                  isEval && item.record.correct === false
                    ? BUCKET_META[attributeRecord(item.record).bucket] || BUCKET_META.unknown
                    : null;
                return (
                  <button
                    key={item.index}
                    className={selected === index ? "active" : ""}
                    onClick={() => setSelected(index)}
                  >
                    <span>#{item.index + 1}</span>
                    <strong>
                      {item.record.question ||
                        item.record.trajectory_id ||
                        `Record ${item.index + 1}`}
                    </strong>
                    {typeof item.record.correct === "boolean" ? (
                      item.record.correct ? <Check size={15} /> : <X size={15} />
                    ) : null}
                    {bucket ? (
                      <em className="case-bucket" style={{ color: bucket.color }}>
                        {bucket.label}
                      </em>
                    ) : null}
                  </button>
                );
              })}
            </div>
            <div className={`record-json ${view === "visual" ? "light" : ""}`}>
              <div className="json-toolbar">
                <span>
                  {view === "visual" ? <Activity size={15} /> : <FileJson size={15} />}
                  {view === "visual" ? "Structured view" : "Raw JSON"} · record {active ? active.index + 1 : "—"}
                </span>
                <span>{data.total} records</span>
              </div>
              {active && view === "visual" ? (
                Array.isArray(active.record.turns) ? (
                  <TrajectoryView record={active.record} />
                ) : (
                  <StructuredRecord record={active.record} />
                )
              ) : null}
              {active && view === "json" ? <JsonBlock value={active.record} /> : null}
            </div>
          </div>
          <div className="pagination">
            <button
              className="icon-button"
              title="上一页"
              disabled={page <= 1}
              onClick={() => setPage((value) => value - 1)}
            >
              <ChevronLeft size={18} />
            </button>
            <span>第 {data.page} / {data.pages} 页</span>
            <button
              className="icon-button"
              title="下一页"
              disabled={page >= data.pages}
              onClick={() => setPage((value) => value + 1)}
            >
              <ChevronRight size={18} />
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}

function ExperimentDetail({ experiment, onBack, onUpdated }) {
  const hasTraining = experiment.training_metrics?.available;
  const [tab, setTab] = useState(hasTraining ? "training" : "evaluation");
  const [refreshing, setRefreshing] = useState(false);
  const [notes, setNotes] = useState(experiment.notes || "");
  const [saving, setSaving] = useState(false);
  const metrics = experiment.training_metrics;
  const evaluation = experiment.evaluation_summary;
  const trainManifest = experiment.dataset_summary?.train || {};
  const tabs = [
    ...(hasTraining ? [["training", "训练过程"]] : []),
    ...(hasTraining ? [["training_data", "训练数据"]] : []),
    ["data", `全部数据 (${experiment.data_sources?.length || 0})`],
    ["evaluation", "评测结果"],
    ...(evaluation?.available ? [["attribution", "错误归因"]] : []),
    ["settings", "实验记录"],
  ];

  const refresh = async () => {
    setRefreshing(true);
    try {
      await api.refresh(experiment.id);
      await onUpdated();
    } finally {
      setRefreshing(false);
    }
  };
  const saveNotes = async () => {
    setSaving(true);
    try {
      await api.patch(experiment.id, { notes });
      await onUpdated();
    } finally {
      setSaving(false);
    }
  };

  return (
    <main className="content">
      <header className="detail-header">
        <button className="icon-button" onClick={onBack} title="返回">
          <ArrowLeft size={19} />
        </button>
        <div className="detail-title">
          <div>
            <div className="title-line">
              <h1>{experiment.name}</h1>
              <StatusDot status={experiment.status} />
            </div>
            <p>{experiment.description}</p>
          </div>
          {hasTraining ? (
            <button className="button secondary" onClick={refresh} disabled={refreshing}>
              <RefreshCw size={16} className={refreshing ? "spin" : ""} />
              刷新远程指标
            </button>
          ) : null}
        </div>
      </header>

      <section className="metric-strip detail-metrics">
        {hasTraining ? (
          <>
            <Metric
              label="训练样本"
              value={compact.format(trainManifest.kept || trainManifest.source_trajectories || 0)}
              detail={`${trainManifest.dropped_overlong || 0} 条超长过滤`}
              icon={Database}
            />
            <Metric
              label="Train loss"
              value={fixed(metrics?.summary?.train_loss)}
              detail={`${metrics?.summary?.global_step || 0} steps`}
              icon={Activity}
            />
            <Metric
              label="Eval loss"
              value={fixed(metrics?.summary?.last_eval_loss)}
              detail={`${metrics?.summary?.epoch || experiment.training?.epochs} epochs`}
              icon={Sparkles}
            />
          </>
        ) : (
          <>
            <Metric label="实验类型" value="Baseline" detail={experiment.method} icon={FlaskConical} />
            <Metric label="模型" value={modelFamily(experiment.model)} detail={experiment.model} icon={Bot} />
            <Metric
              label="平均耗时"
              value={`${fixed(evaluation?.average_elapsed_seconds, 2)}s`}
              detail={`${evaluation?.total || 0} cases`}
              icon={Clock3}
            />
          </>
        )}
        <Metric
          label="执行准确率"
          value={pct(evaluation?.accuracy)}
          detail={`${evaluation?.correct || 0} / ${evaluation?.total || 0}`}
          icon={Gauge}
        />
      </section>

      <div className="tabs">
        {tabs.map(([value, label]) => (
          <button key={value} className={tab === value ? "active" : ""} onClick={() => setTab(value)}>
            {label}
          </button>
        ))}
      </div>

      {tab === "training_data" ? (
        <section className="panel browser-panel">
          <div className="panel-heading">
            <div>
              <h2>训练数据看板</h2>
              <p>模型实际训练的工具轨迹:开场目录、每步 think、工具调用与观测</p>
            </div>
          </div>
          <TrainingBoard experiment={experiment} />
        </section>
      ) : null}

      {tab === "attribution" ? (
        <section className="panel browser-panel">
          <div className="panel-heading">
            <div>
              <h2>错误归因</h2>
              <p>按错误发生的位置归因,并诊断模型是否在决策点主动查表</p>
            </div>
          </div>
          <AttributionPanel experimentId={experiment.id} />
        </section>
      ) : null}

      {tab === "training" ? (
        <section className="detail-grid">
          <div className="panel wide-panel">
            <div className="panel-heading">
              <div>
                <h2>Loss 曲线</h2>
                <p>训练与验证损失随优化步变化</p>
              </div>
              <span className="unit">step</span>
            </div>
            <LossChart metrics={metrics} />
          </div>
          <div className="panel config-panel">
            <div className="panel-heading">
              <div>
                <h2>训练配置</h2>
                <p>{experiment.method}</p>
              </div>
            </div>
            <dl className="definition-list">
              <div><dt>Base model</dt><dd>{experiment.model}</dd></div>
              <div><dt>Dataset</dt><dd>{experiment.dataset_version}</dd></div>
              <div><dt>Epochs</dt><dd>{experiment.training.epochs}</dd></div>
              <div><dt>Cutoff</dt><dd>{experiment.training.cutoff_len}</dd></div>
              <div><dt>Effective batch</dt><dd>{experiment.training.effective_batch_size}</dd></div>
              <div><dt>Learning rate</dt><dd>{experiment.training.learning_rate}</dd></div>
              <div><dt>LoRA rank / alpha</dt><dd>{experiment.training.lora_rank} / {experiment.training.lora_alpha}</dd></div>
              <div><dt>开始时间</dt><dd>{dateTime(experiment.started_at)}</dd></div>
              <div><dt>结束时间</dt><dd>{dateTime(experiment.completed_at)}</dd></div>
              <div><dt>训练耗时</dt><dd>{duration(metrics?.summary?.train_runtime)}</dd></div>
              <div><dt>Checkpoint</dt><dd className="path-value">{experiment.training.checkpoint}</dd></div>
            </dl>
          </div>
        </section>
      ) : null}

      {tab === "data" ? (
        <section className="panel browser-panel">
          <div className="panel-heading">
            <div>
              <h2>JSON 数据浏览器</h2>
              <p>检查训练样本与逐案例评测产物</p>
            </div>
          </div>
          <RecordBrowser experiment={experiment} />
        </section>
      ) : null}

      {tab === "evaluation" ? (
        <>
          <section className="evaluation-layout">
            <div className="panel eval-summary">
            <div className="panel-heading">
              <div>
                <h2>评测概况</h2>
                <p>{evaluation?.total || 0} 个 Spider dev 案例</p>
              </div>
            </div>
            <div className="eval-score">
              <strong>{pct(evaluation?.accuracy)}</strong>
              <span>execution accuracy</span>
            </div>
            <dl className="definition-list">
              <div><dt>合法回答率</dt><dd>{Number.isFinite(evaluation?.legal_rate) ? pct(evaluation.legal_rate) : "不适用"}</dd></div>
              <div><dt>平均工具步数</dt><dd>{fixed(evaluation?.average_steps, 2)}</dd></div>
              <div><dt>平均耗时</dt><dd>{fixed(evaluation?.average_elapsed_seconds, 2)}s</dd></div>
              <div>
                <dt>完整性</dt>
                <dd className={evaluation?.complete_dev ? "text-ok" : "text-warn"}>
                  {evaluation?.complete_dev ? "完整" : `${evaluation?.total || 0}/1034`}
                </dd>
              </div>
            </dl>
            </div>
            <div className="panel failure-panel">
            <div className="panel-heading">
              <div>
                <h2>失败分布</h2>
                <p>按最终 failure_type 汇总</p>
              </div>
            </div>
            <div className="failure-bars">
              {Object.entries(evaluation?.failure_types || {}).map(([key, count]) => {
                const totalFailures = Math.max(1, (evaluation?.total || 0) - (evaluation?.correct || 0));
                return (
                  <div key={key}>
                    <div className="failure-label"><span>{key}</span><strong>{count}</strong></div>
                    <div className="progress"><span style={{ width: `${(count / totalFailures) * 100}%` }} /></div>
                  </div>
                );
              })}
            </div>
            </div>
          </section>
          {experiment.id !== "qwen25-7b-base-sql" ? (
            <section className="panel attribution-panel">
              <div className="panel-heading">
                <div>
                  <h2>错误归因与感知行为</h2>
                  <p>从完整 rollout 中区分协议、执行、结果形状和主动查表问题</p>
                </div>
              </div>
              <AttributionPanel experimentId={experiment.id} />
            </section>
          ) : null}
        </>
      ) : null}

      {tab === "settings" ? (
        <section className="settings-layout">
          <div className="panel notes-panel">
            <div className="panel-heading">
              <div>
                <h2>实验备注</h2>
                <p>记录观察、异常与下一步判断</p>
              </div>
            </div>
            <textarea value={notes} onChange={(event) => setNotes(event.target.value)} rows={10} />
            <div className="form-actions">
              <button className="button primary" onClick={saveNotes} disabled={saving}>
                <Save size={16} />
                保存备注
              </button>
            </div>
          </div>
          <div className="panel manifest-panel">
            <div className="panel-heading">
              <div>
                <h2>实验登记信息</h2>
                <p>{experiment.data_sources?.length || 0} 个本地数据资产</p>
              </div>
            </div>
            <StructuredRecord record={experiment} />
          </div>
        </section>
      ) : null}
    </main>
  );
}

function Playground() {
  const [config, setConfig] = useState(null);
  const [serverInfo, setServerInfo] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [launchModel, setLaunchModel] = useState("");
  const [gpuIndex, setGpuIndex] = useState("");
  const [question, setQuestion] = useState("");
  const [system, setSystem] = useState("");
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [status, setStatus] = useState("checking");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [managing, setManaging] = useState(false);

  const loadServer = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setError("");
    try {
      const payload = await api.server();
      setServerInfo(payload);
      const available = payload.gpus?.find((gpu) => gpu.available);
      setGpuIndex((current) => current || (available ? String(available.index) : ""));
      return payload;
    } catch (reason) {
      if (!quiet) setError(reason.message);
      return null;
    }
  }, []);

  const checkModels = useCallback(async ({ quiet = false } = {}) => {
    setStatus("checking");
    try {
      const payload = await api.models();
      const list = payload.data?.data || [];
      setModels(list);
      setModel((current) => current || list[0]?.id || "");
      setStatus("online");
    } catch (reason) {
      setStatus("offline");
      if (!quiet) setError(reason.message);
    }
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([api.playgroundConfig(), loadServer()])
      .then(([nextConfig]) => {
        if (!active) return;
        setConfig(nextConfig);
        setSystem(nextConfig.system_prompt || "");
        setLaunchModel(nextConfig.models?.[0]?.experiment_id || "");
      })
      .catch((reason) => {
        if (active) setError(reason.message);
      })
      .finally(() => {
        if (active) checkModels({ quiet: true });
      });
    const timer = window.setInterval(() => {
      loadServer({ quiet: true }).then((info) => {
        if (info?.ready) checkModels({ quiet: true });
      });
    }, 10000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [checkModels, loadServer]);

  const startServer = async () => {
    setManaging(true);
    setError("");
    try {
      await api.startServer({
        experiment_id: launchModel,
        gpu_index: Number(gpuIndex),
      });
      setStatus("checking");
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => window.setTimeout(resolve, 3000));
        const info = await loadServer({ quiet: true });
        if (info?.ready) {
          await checkModels();
          setModel(launchModel);
          return;
        }
        if (info?.managed && !info.running) {
          throw new Error(info.log_tail?.slice(-1)[0] || "vLLM 启动后异常退出");
        }
      }
      throw new Error("vLLM 在 180 秒内未就绪，请检查服务日志");
    } catch (reason) {
      setError(reason.message);
      await loadServer({ quiet: true });
    } finally {
      setManaging(false);
    }
  };

  const stopServer = async () => {
    setManaging(true);
    setError("");
    try {
      await api.stopServer();
      setModels([]);
      setModel("");
      setStatus("offline");
      await loadServer({ quiet: true });
    } catch (reason) {
      setError(reason.message);
    } finally {
      setManaging(false);
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    setRunning(true);
    setError("");
    setResponse(null);
    try {
      const payload = await api.chat({
        model,
        messages: [
          ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
          { role: "user", content: question },
        ],
        temperature,
        max_tokens: maxTokens,
      });
      setResponse(payload);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setRunning(false);
    }
  };

  const answer = response?.data?.choices?.[0]?.message?.content;

  return (
    <main className="content">
      <header className="page-header">
        <div>
          <p className="eyebrow">OpenAI-compatible endpoint</p>
          <h1>模型 Playground</h1>
          <p>通过本地后端代理访问 SSH 隧道后的 vLLM 服务</p>
        </div>
        <button className="button secondary" onClick={checkModels}>
          <RefreshCw size={16} />
          检查服务
        </button>
      </header>

      <div className="server-strip">
        <div className={`server-state ${status}`}>
          <Server size={17} />
          <span>{status === "online" ? "vLLM 在线" : status === "checking" ? "正在连接" : "vLLM 离线"}</span>
        </div>
        <span>{models.length ? `${models.length} 个可用模型` : "等待模型列表"}</span>
      </div>

      <section className="panel service-panel">
        <div className="panel-heading">
          <div>
            <h2>模型服务</h2>
            <p>在 NewGNN 的空闲 GPU 上启动已登记的 LoRA 模型</p>
          </div>
          <span className={`service-badge ${serverInfo?.running ? "running" : ""}`}>
            {serverInfo?.ready ? "Ready" : serverInfo?.running ? "Loading" : "Stopped"}
          </span>
        </div>
        <div className="service-controls">
          <label>
            <span>训练模型</span>
            <select
              value={launchModel}
              onChange={(event) => setLaunchModel(event.target.value)}
              disabled={serverInfo?.running || managing}
            >
              {(config?.models || []).map((item) => (
                <option key={item.experiment_id} value={item.experiment_id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>GPU</span>
            <select
              value={gpuIndex}
              onChange={(event) => setGpuIndex(event.target.value)}
              disabled={serverInfo?.running || managing}
            >
              {!serverInfo?.gpus?.length ? <option value="">正在读取 GPU</option> : null}
              {(serverInfo?.gpus || []).map((gpu) => (
                <option key={gpu.index} value={gpu.index} disabled={!gpu.available}>
                  GPU {gpu.index} · {gpu.memory_used_mib}/{gpu.memory_total_mib} MiB
                  {gpu.available ? " · 空闲" : ` · ${gpu.processes?.[0]?.user || "占用"}`}
                </option>
              ))}
            </select>
          </label>
          {serverInfo?.running ? (
            <button className="button danger" type="button" onClick={stopServer} disabled={managing}>
              {managing ? <RefreshCw size={16} className="spin" /> : <Square size={15} />}
              停止服务
            </button>
          ) : (
            <button
              className="button primary"
              type="button"
              onClick={startServer}
              disabled={managing || !launchModel || gpuIndex === ""}
            >
              {managing ? <RefreshCw size={16} className="spin" /> : <Power size={16} />}
              {managing ? "启动中" : "启动服务"}
            </button>
          )}
        </div>
        <div className="gpu-grid">
          {(serverInfo?.gpus || []).map((gpu) => (
            <div className={`gpu-item ${gpu.available ? "available" : "busy"}`} key={gpu.index}>
              <div className="gpu-title">
                <span><Cpu size={15} /> GPU {gpu.index}</span>
                <strong>{gpu.utilization_percent}%</strong>
              </div>
              <div className="gpu-memory">
                <span style={{ width: `${Math.min(100, (gpu.memory_used_mib / gpu.memory_total_mib) * 100)}%` }} />
              </div>
              <div className="gpu-detail">
                <span><HardDrive size={13} /> {gpu.memory_used_mib} / {gpu.memory_total_mib} MiB</span>
                <span>{gpu.available ? "可用" : gpu.processes?.map((item) => item.user).join(", ") || "占用中"}</span>
              </div>
            </div>
          ))}
        </div>
        {serverInfo?.managed ? (
          <div className="service-meta">
            <span>PID {serverInfo.pid || "—"}</span>
            <span>GPU {serverInfo.gpu_index ?? "—"}</span>
            <span>{serverInfo.served_model_name || "—"}</span>
            <span>{serverInfo.ready ? "OpenAI API 已就绪" : "正在加载权重"}</span>
          </div>
        ) : null}
        {serverInfo?.log_tail?.length ? (
          <details className="service-log">
            <summary>查看服务日志</summary>
            <pre className="json-block">{serverInfo.log_tail.join("")}</pre>
          </details>
        ) : null}
      </section>

      <section className="playground-layout">
        <form className="panel prompt-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div>
              <h2>请求</h2>
              <p>发送一次非流式 Chat Completions 调用</p>
            </div>
          </div>
          <label>
            <span>模型</span>
            <select value={model} onChange={(event) => setModel(event.target.value)} required>
              {!models.length ? <option value="">未发现模型</option> : null}
              {models.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
            </select>
          </label>
          <label>
            <span>
              System prompt · {config?.protocol_version || "loading"}
              {config?.protocol_hash ? ` · ${config.protocol_hash}` : ""}
            </span>
            <textarea value={system} onChange={(event) => setSystem(event.target.value)} rows={8} />
          </label>
          <label>
            <span>问题</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={8}
              placeholder="输入要测试的问题..."
              required
            />
          </label>
          <div className="form-row">
            <label>
              <span>Temperature</span>
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Max tokens</span>
              <input
                type="number"
                min="1"
                max="4096"
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </label>
          </div>
          <button className="button primary submit-button" disabled={running || status !== "online"}>
            {running ? <RefreshCw size={17} className="spin" /> : <Play size={17} />}
            {running ? "生成中" : "运行模型"}
          </button>
        </form>

        <div className="panel response-panel">
          <div className="panel-heading">
            <div>
              <h2>响应</h2>
              <p>{response ? `${response.latency_seconds}s` : "等待请求"}</p>
            </div>
            <Bot size={20} />
          </div>
          {error ? <Empty title="请求失败" detail={error} /> : null}
          {!response && !error ? (
            <div className="response-placeholder">
              <Sparkles size={24} />
              <span>模型输出会显示在这里</span>
            </div>
          ) : null}
          {answer ? <div className="answer-block">{answer}</div> : null}
          {response ? (
            <details>
              <summary>查看原始响应 JSON</summary>
              <JsonBlock value={response.data} />
            </details>
          ) : null}
        </div>
      </section>
    </main>
  );
}

function Sidebar({ view, onView, collapsed, onToggle, mobileOpen, onMobileClose }) {
  return (
    <>
      {mobileOpen ? <button className="backdrop" onClick={onMobileClose} aria-label="关闭菜单" /> : null}
      <aside className={`sidebar ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><FlaskConical size={20} /></div>
          {!collapsed ? <div><strong>Tabular RL</strong><span>Experiment Console</span></div> : null}
        </div>
        <nav>
          <button className={view === "overview" ? "active" : ""} onClick={() => onView("overview")}>
            <Activity size={19} />
            {!collapsed ? <span>实验总览</span> : null}
          </button>
          <button className={view === "playground" ? "active" : ""} onClick={() => onView("playground")}>
            <Bot size={19} />
            {!collapsed ? <span>模型测试</span> : null}
          </button>
          <button className={view === "construction" ? "active" : ""} onClick={() => onView("construction")}>
            <FlaskConical size={19} />
            {!collapsed ? <span>数据构造</span> : null}
          </button>
        </nav>
        <button className="collapse-button" onClick={onToggle} title={collapsed ? "展开侧栏" : "收起侧栏"}>
          <PanelLeftClose size={18} className={collapsed ? "flip" : ""} />
        </button>
      </aside>
    </>
  );
}

export default function App() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [view, setView] = useState("overview");
  const [selectedId, setSelectedId] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const load = async () => {
    try {
      setError("");
      setExperiments(await api.experiments());
    } catch (reason) {
      setError(reason.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const selected = experiments.find((item) => item.id === selectedId);
  const navigate = (next) => {
    setSelectedId(null);
    setView(next);
    setMobileOpen(false);
  };

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        view={selected ? "detail" : view}
        onView={navigate}
        collapsed={collapsed}
        onToggle={() => setCollapsed((value) => !value)}
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />
      <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="打开菜单">
        <Menu size={21} />
      </button>
      {loading ? (
        <main className="content"><div className="loading-page"><div className="loader" />加载实验记录</div></main>
      ) : error ? (
        <main className="content"><Empty title="控制台加载失败" detail={error} /></main>
      ) : selected ? (
        <ExperimentDetail experiment={selected} onBack={() => setSelectedId(null)} onUpdated={load} />
      ) : view === "playground" ? (
        <Playground />
      ) : view === "construction" ? (
        <ConstructionPanel />
      ) : (
        <Overview experiments={experiments} onOpen={setSelectedId} />
      )}
    </div>
  );
}
