import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  Bot,
  Check,
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
import { PaginationControls } from "./PaginationControls";
import {
  AttributionPanel,
  attributeRecord,
  BUCKET_META,
  ConstructionPanel,
  PassKRecordView,
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
  researchSummary: () => api.request("/api/research-summary"),
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
const BENCHMARKS = {
  spider: {
    label: "Spider",
    scope: "评测：Spider dev；数据构造：Spider train",
    description: "Spider 上的 baseline、SFT 训练与工具调用评测。",
  },
  bird: {
    label: "BIRD",
    scope: "评测：BIRD public Dev；数据构造：BIRD train-compatible",
    description: "BIRD 大型数据库上的 baseline 与数据构造实验。",
  },
};
const benchmarkFor = (item) => {
  if (BENCHMARKS[item.benchmark]) return item.benchmark;
  const fingerprint = [item.id, item.dataset_version, ...(item.tags || [])].join(" ").toLowerCase();
  return fingerprint.includes("bird") ? "bird" : "spider";
};
const combineRunSummaries = (runs = []) => {
  const summaries = runs.map((run) => run.summary).filter((summary) => summary?.available);
  if (!summaries.length) return { available: false };
  const total = summaries.reduce((sum, summary) => sum + (summary.total || 0), 0);
  const correct = summaries.reduce((sum, summary) => sum + (summary.correct || 0), 0);
  const passKeys = [...new Set(summaries.flatMap((summary) => Object.keys(summary.pass_at || {})))]
    .sort((a, b) => Number(a) - Number(b));
  const pass_at = {};
  for (const key of passKeys) {
    const passTotal = summaries.reduce((sum, summary) => sum + (summary.pass_at?.[key]?.total || 0), 0);
    const passCorrect = summaries.reduce((sum, summary) => sum + (summary.pass_at?.[key]?.correct || 0), 0);
    pass_at[key] = {
      correct: passCorrect,
      total: passTotal,
      rate: passTotal ? passCorrect / passTotal : 0,
    };
  }
  const legalObserved = summaries.reduce((sum, summary) => sum + (summary.legal_observed || 0), 0);
  const legal = summaries.reduce((sum, summary) => sum + (summary.legal || 0), 0);
  return {
    available: true,
    total,
    correct,
    accuracy: total ? correct / total : 0,
    legal: legalObserved ? legal : null,
    legal_rate: legalObserved ? legal / legalObserved : null,
    legal_observed: legalObserved,
    complete: null,
    pass_at: Object.keys(pass_at).length ? pass_at : undefined,
  };
};
const displayEvaluationSummary = (item) =>
  item.evaluation_summary?.available
    ? item.evaluation_summary
    : combineRunSummaries(item.evaluation_runs || []);
const primaryMetric = (summary) => {
  if (!summary?.available) return { label: "Acc", value: null, correct: 0, total: 0 };
  const entries = Object.entries(summary.pass_at || {}).sort(([a], [b]) => Number(a) - Number(b));
  if (entries.length) {
    const [key, value] = entries[entries.length - 1];
    return {
      label: `pass@${key}`,
      value: value?.rate,
      correct: value?.correct || 0,
      total: value?.total || 0,
    };
  }
  return {
    label: "Acc",
    value: summary.accuracy,
    correct: summary.correct || 0,
    total: summary.total || 0,
  };
};
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

const artifactCategoryLabels = {
  all: "全部",
  reports: "正式报告",
  evaluation: "评测结果",
  trajectories: "轨迹数据",
  "eval-inputs": "冻结输入",
  table_rl: "服务器产物",
  workspace: "工作区",
};

function ResearchOverview({ summary }) {
  const [artifactCategory, setArtifactCategory] = useState("all");
  const [artifactQuery, setArtifactQuery] = useState("");
  const [diagnosticFamily, setDiagnosticFamily] = useState("all");
  const [copiedPath, setCopiedPath] = useState("");

  const fullDevData = (summary?.full_dev_runs || []).map((item) => ({
    ...item,
    accuracyPct: Number((item.accuracy * 100).toFixed(2)),
    validPct: Number(((item.valid / item.total) * 100).toFixed(2)),
  }));
  const baseline = fullDevData.find((item) => item.id === "sft2");
  const directSqlData = (summary?.direct_sql_runs || []).map((item) => ({
    ...item,
    name: item.id === "coder-direct-greedy" ? "greedy" : `pass@${item.k}`,
    accuracyPct: Number((item.accuracy * 100).toFixed(2)),
  }));
  const coderToolSftData = (summary?.coder_tool_sft_runs || []).map((item) => ({
    ...item,
    accuracyPct: Number((item.accuracy * 100).toFixed(2)),
  }));
  const coderComparisonData = ["greedy", "pass@1", "pass@2", "pass@4"].map((name) => {
    const directSql = directSqlData.find((item) => item.name === name);
    const toolSft = coderToolSftData.find((item) => item.name === name);
    return {
      name,
      directSqlPct: directSql?.accuracyPct,
      directSqlCorrect: directSql?.correct,
      toolSftPct: toolSft?.accuracyPct,
      toolSftCorrect: toolSft?.correct,
      total: directSql?.total || toolSft?.total,
    };
  });
  const omnisql = summary?.omnisql_comparison || {};
  const omnisqlGreedy = omnisql.greedy || [];
  const omnisqlPaired = omnisql.paired || [];
  const omnisqlFailureTypes = omnisql.greedy_failure_types || [];
  const omnisqlPassK = omnisql.pass_k || [];
  const fixed200Data = (summary?.fixed200_runs || [])
    .filter((item) => item.total === 200)
    .map((item) => ({
      ...item,
      accuracyPct: Number(((item.correct / item.total) * 100).toFixed(1)),
      legalPct: Number.isFinite(item.legal)
        ? Number(((item.legal / item.total) * 100).toFixed(1))
        : null,
    }));
  const diagnosticFamilies = [
    "all",
    ...new Set((summary?.diagnostic_runs || []).map((item) => item.family)),
  ];
  const diagnostics = (summary?.diagnostic_runs || []).filter(
    (item) => diagnosticFamily === "all" || item.family === diagnosticFamily,
  );
  const papersById = new Map((summary?.rl_papers || []).map((paper) => [paper.id, paper]));
  const normalizedQuery = artifactQuery.trim().toLowerCase();
  const artifacts = (summary?.artifacts || []).filter((item) => (
    (artifactCategory === "all" || item.category === artifactCategory)
      && (!normalizedQuery || item.path.toLowerCase().includes(normalizedQuery))
  ));
  const copyPath = async (path) => {
    await navigator.clipboard.writeText(path);
    setCopiedPath(path);
    window.setTimeout(() => setCopiedPath(""), 1400);
  };

  if (!summary) {
    return <main className="content"><Empty title="研究统计不可用" detail="没有加载到统一实验摘要。" /></main>;
  }

  return (
    <main className="content research-overview">
      <header className="page-header">
        <div>
          <p className="eyebrow">Research evidence map</p>
          <h1>{summary.title}</h1>
          <p>{summary.period.start} 至 {summary.period.end}；正式结果、监督效率与原始产物统一索引。</p>
        </div>
        <div className="header-date">
          <Clock3 size={16} />
          更新于 {summary.period.updated}
        </div>
      </header>

      <section className="research-contract">
        <strong>完整评测口径</strong>
        <span>{summary.evaluation_contract}</span>
        <code>{summary.source_index}</code>
      </section>

      <section className="metric-strip">
        <Metric
          label="Teacher-union60 完整评测"
          value={`${summary.headline.best_full_dev_correct}/${summary.headline.best_full_dev_total}`}
          detail={`${summary.headline.best_full_dev_run} · ${pct(summary.headline.best_full_dev_correct / summary.headline.best_full_dev_total)}`}
          icon={Gauge}
        />
        <Metric
          label="相对 SFT2"
          value={`+${summary.headline.best_full_dev_correct - summary.headline.sft2_correct}`}
          detail="union60 的单次完整评测净分数"
          icon={Activity}
        />
        <Metric
          label="已结构化正式结果"
          value={summary.coverage.full_dev_runs + summary.coverage.diagnostic_runs}
          detail={`${summary.coverage.full_dev_runs} 个 full-dev · ${summary.coverage.diagnostic_runs} 个诊断`}
          icon={FlaskConical}
        />
        <Metric
          label="数据与报告入口"
          value={summary.coverage.artifact_paths}
          detail={`${summary.coverage.local_artifacts_available} 个本地可用 · ${summary.coverage.remote_artifacts} 个远端`}
          icon={HardDrive}
        />
      </section>

      <section className="finding-grid">
        {summary.findings.map((item, index) => (
          <article className="finding-card" key={item.title}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{item.title}</strong>
            <p>{item.detail}</p>
          </article>
        ))}
      </section>

      <section className="panel rl-lineage-panel">
        <div className="panel-heading">
          <div>
            <h2>RL 方法谱系与论文来源</h2>
            <p>记录已经实际训练的方法、监督信号、信用粒度、评测数字及对应论文来源。</p>
          </div>
          <span className="unit">{summary.rl_method_families.length} families · {summary.rl_experiments.length} runs</span>
        </div>
        <div className="rl-boundary-note">
          <strong>{summary.rl_implementation_note.title}</strong>
          <span>{summary.rl_implementation_note.exp1}</span>
          <span>{summary.rl_implementation_note.process}</span>
          <span>{summary.rl_implementation_note.other}</span>
          <span>{summary.rl_implementation_note.rank}</span>
          <code>docs/reports/rl/RL_METHODS_AND_PAPER_PROVENANCE_20260805_ZH.md</code>
        </div>
        <div className="rl-method-grid">
          {summary.rl_method_families.map((method) => (
            <article key={method.id}>
              <header>
                <div>
                  <span>{method.experiments.join(" · ")}</span>
                  <strong>{method.label}</strong>
                </div>
                <b>{method.provenance}</b>
              </header>
              <dl>
                <div><dt>训练信号</dt><dd>{method.signal}</dd></div>
                <div><dt>信用粒度</dt><dd>{method.credit}</dd></div>
                <div><dt>实测数据</dt><dd>{method.result}</dd></div>
              </dl>
              <footer>
                {method.paper_ids.map((paperId) => {
                  const paper = papersById.get(paperId);
                  return paper ? <a key={paper.id} href={paper.url} target="_blank" rel="noreferrer">{paper.short} ↗</a> : null;
                })}
              </footer>
            </article>
          ))}
        </div>
        <details className="rl-experiment-details">
          <summary>展开 Exp0–18 与 teacher-union 的算法、训练配置和实测结果</summary>
          <div className="table-scroll research-table">
            <table>
              <thead><tr><th>实验</th><th>方法</th><th>实际算法</th><th>训练数据 / 更新</th><th>优化配置</th><th>信号 / mask</th><th>评测口径</th><th>结果</th><th>数值比较 / 记录</th></tr></thead>
              <tbody>
                {summary.rl_experiments.map((item) => (
                  <tr key={item.id}>
                    <td><strong>{item.id}</strong></td>
                    <td>{item.method}</td>
                    <td>{item.algorithm}</td>
                    <td>{item.training}</td>
                    <td><code>{item.optimization}</code></td>
                    <td>{item.change}</td>
                    <td><code>{item.scope}</code></td>
                    <td>{item.result}</td>
                    <td>{item.comparison}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <div className="paper-library">
          <div className="paper-library-heading">
            <div><strong>原始论文索引</strong><span>点击标题打开 DOI / arXiv 原始页面。</span></div>
            <b>{summary.rl_papers.length} papers</b>
          </div>
          <div className="paper-grid">
            {summary.rl_papers.map((paper) => (
              <a key={paper.id} href={paper.url} target="_blank" rel="noreferrer">
                <span>{paper.year}</span>
                <strong>{paper.title}</strong>
                <p>{paper.relation}</p>
              </a>
            ))}
          </div>
        </div>
      </section>

      <section className="research-chart-grid">
        <div className="panel research-chart-panel">
          <div className="panel-heading">
            <div>
              <h2>SFT2 → RL 完整评测</h2>
              <p>同一模型、同一 BIRD-dev1534 greedy@1 口径；绿色虚线为 SFT2。</p>
            </div>
            <span className="unit">accuracy %</span>
          </div>
          <div className="research-tall-chart">
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 760, height: 510 }}>
              <BarChart data={fullDevData} layout="vertical" margin={{ top: 12, right: 28, bottom: 10, left: 6 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" domain={[49, 52]} tickFormatter={(value) => `${value}%`} />
                <YAxis type="category" dataKey="label" width={116} tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(value, name, item) => [
                    `${value}% · ${item.payload.correct}/${item.payload.total}`,
                    "执行准确率",
                  ]}
                  labelFormatter={(label, items) => `${label} · ${items?.[0]?.payload?.method || ""}`}
                />
                <ReferenceLine x={baseline?.accuracyPct} stroke="#087f5b" strokeDasharray="5 4" />
                <Bar dataKey="accuracyPct" radius={[0, 4, 4, 0]}>
                  {fullDevData.map((item) => (
                    <Cell
                      key={item.id}
                      fill={item.id === "union60" ? "#087f5b" : item.id === "sft2" ? "#7a8d81" : "#82aa9b"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel research-chart-panel">
          <div className="panel-heading">
            <div>
              <h2>Coder-7B Direct SQL / Tool SFT</h2>
              <p>BIRD-dev 1534 · greedy + sampled K=4 · strict official EX</p>
            </div>
            <span className="unit">BIRD-dev 1534</span>
          </div>
          <div className="research-short-chart">
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 520, height: 320 }}>
              <LineChart data={coderComparisonData} margin={{ top: 22, right: 30, bottom: 12, left: 6 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[45, 70]} tickFormatter={(value) => `${value}%`} />
                <Tooltip
                  formatter={(value, name, item) => {
                    const isDirect = name === "directSqlPct";
                    const correct = isDirect ? item.payload.directSqlCorrect : item.payload.toolSftCorrect;
                    return [`${value}% · ${correct}/${item.payload.total}`, isDirect ? "Direct SQL" : "Tool SFT"];
                  }}
                />
                <Legend formatter={(value) => value === "directSqlPct" ? "Direct SQL" : "Tool SFT"} />
                <Line type="monotone" dataKey="directSqlPct" stroke="#4976a8" strokeWidth={3} dot={{ r: 5 }} />
                <Line type="monotone" dataKey="toolSftPct" stroke="#087f5b" strokeWidth={3} dot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div><h2>OmniSQL · Greedy</h2></div>
          <span className="unit">{omnisql.dataset} · {omnisql.total} 题</span>
        </div>
        <div className="table-scroll research-table">
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>训练</th>
                <th>接口</th>
                <th>评测协议 / 输入</th>
                <th>Correct</th>
                <th>Accuracy</th>
                <th>Valid / Legal</th>
                <th>平均步骤</th>
              </tr>
            </thead>
            <tbody>
              {omnisqlGreedy.map((item) => (
                <tr key={item.id}>
                  <td><strong>{item.label}</strong><br /><span>{item.model}</span></td>
                  <td>{item.training}</td>
                  <td>{item.interface}</td>
                  <td><code>{item.protocol}</code></td>
                  <td>{item.correct}/{item.total}</td>
                  <td>{fixed(item.accuracy * 100, 2)}%</td>
                  <td>{Number.isFinite(item.valid) ? `${item.valid}/${item.total} · ${fixed((item.valid / item.total) * 100, 2)}%` : "—"}</td>
                  <td>{fixed(item.average_steps, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="research-chart-grid">
        <div className="panel">
          <div className="panel-heading">
            <div><h2>OmniSQL SFT 后 · Greedy 逐题配对</h2></div>
            <span className="unit">{omnisql.total} 题</span>
          </div>
          <div className="table-scroll research-table">
            <table>
              <thead>
                <tr>
                  <th>对照</th>
                  <th>都正确</th>
                  <th>OmniSQL SFT 后独有</th>
                  <th>对照独有</th>
                  <th>都错误</th>
                  <th>Net</th>
                  <th>Δ pp</th>
                </tr>
              </thead>
              <tbody>
                {omnisqlPaired.map((item) => (
                  <tr key={item.reference}>
                    <td><strong>{item.reference}</strong></td>
                    <td>{item.both_correct}</td>
                    <td>{item.omnisql_after_only}</td>
                    <td>{item.reference_only}</td>
                    <td>{item.both_wrong}</td>
                    <td>{item.net_correct > 0 ? "+" : ""}{item.net_correct}</td>
                    <td>{item.delta_pp > 0 ? "+" : ""}{fixed(item.delta_pp, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div><h2>OmniSQL · Pass@k</h2></div>
            <span className="unit">{omnisql.dataset}</span>
          </div>
          <div className="table-scroll research-table">
            <table>
              <thead>
                <tr>
                  <th>模型</th>
                  <th>协议</th>
                  <th>Pass@1</th>
                  <th>Pass@2</th>
                  <th>Pass@4</th>
                  <th>Sample valid</th>
                  <th>平均步骤</th>
                </tr>
              </thead>
              <tbody>
                {omnisqlPassK.map((item) => (
                  <tr key={item.id}>
                    <td><strong>{item.label}</strong></td>
                    <td><code>{item.protocol}</code></td>
                    {["1", "2", "4"].map((k) => (
                      <td key={k}>{item.pass_at?.[k] ? `${item.pass_at[k].correct}/${item.pass_at[k].total} · ${fixed(item.pass_at[k].accuracy * 100, 2)}%` : "—"}</td>
                    ))}
                    <td>{Number.isFinite(item.sample_valid) ? `${item.sample_valid}/${item.sample_total} · ${fixed(item.sample_valid_rate * 100, 2)}%` : "—"}</td>
                    <td>{fixed(item.average_steps, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div><h2>Greedy 终止类型</h2></div>
          <span className="unit">count</span>
        </div>
        <div className="table-scroll research-table">
          <table>
            <thead><tr><th>failure_type</th><th>OmniSQL SFT 后</th><th>SFT2 checkpoint-1682</th></tr></thead>
            <tbody>
              {omnisqlFailureTypes.map((item) => (
                <tr key={item.failure_type}>
                  <td><code>{item.failure_type}</code></td>
                  <td>{item.omnisql_after_sft}</td>
                  <td>{item.sft2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel supervision-panel">
        <div className="panel-heading">
          <div>
            <h2>Teacher-union 信用分配与数据效率</h2>
            <p>真实 streaming student rollout；dense MOPD 与 verified same-prefix repair DPO 分开统计。</p>
          </div>
          <span className="unit">60 → 120 tasks</span>
        </div>
        <div className="supervision-grid">
          {summary.supervision_runs.map((run) => {
            const repairRate = run.repair_generated ? run.repair_correct / run.repair_generated : 0;
            const dpoRate = run.student_failed ? run.strong_dpo_tasks / run.student_failed : 0;
            return (
              <article className="supervision-card" key={run.id}>
                <header>
                  <div><strong>{run.label}</strong><span>{run.questions} 题 · {run.optimizer_updates} 次更新</span></div>
                  <b>{run.student_correct}/{run.questions}</b>
                </header>
                <div className="supervision-flow">
                  <div><span>失败题</span><strong>{run.student_failed}</strong></div>
                  <i>→</i>
                  <div><span>生成 repair</span><strong>{run.repair_generated}</strong></div>
                  <i>→</i>
                  <div><span>验证正确</span><strong>{run.repair_correct}</strong></div>
                  <i>→</i>
                  <div className="accent"><span>强 DPO 题</span><strong>{run.strong_dpo_tasks}</strong></div>
                </div>
                <div className="efficiency-row">
                  <label><span>Repair 正确率</span><strong>{pct(repairRate)}</strong><i style={{ width: `${repairRate * 100}%` }} /></label>
                  <label><span>失败题→强 DPO</span><strong>{pct(dpoRate)}</strong><i style={{ width: `${dpoRate * 100}%` }} /></label>
                  <label><span>Dense teacher 覆盖</span><strong>{pct(run.dense_teacher_tasks / run.questions)}</strong><i style={{ width: `${(run.dense_teacher_tasks / run.questions) * 100}%` }} /></label>
                </div>
                <footer>
                  <span>合法 repair {run.repair_legal}</span>
                  <span>弱 / 无 repair {run.weak_repair_tasks} / {run.no_repair_tasks}</span>
                  <span>策略 token {compact.format(run.active_policy_tokens)}</span>
                  <span>mean A {run.mean_dense_advantage.toExponential(2)}</span>
                </footer>
              </article>
            );
          })}
        </div>
        <div className="table-scroll research-table">
          <table>
            <thead><tr><th>配对比较</th><th>Gains</th><th>Regressions</th><th>Net</th><th>Exact paired p</th></tr></thead>
            <tbody>
              {summary.paired_comparisons.map((item) => (
                <tr key={item.label}>
                  <td><strong>{item.label}</strong></td>
                  <td className="text-ok">{item.gains}</td>
                  <td className="text-warn">{item.regressions}</td>
                  <td className={item.net > 0 ? "text-ok" : "text-warn"}>{item.net > 0 ? "+" : ""}{item.net}</td>
                  <td>{fixed(item.exact_p, 5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="research-chart-grid">
        <div className="panel research-chart-panel">
          <div className="panel-heading">
            <div>
              <h2>固定 200 题：Atomic 与 Native</h2>
              <p>相同 frozen cohort；正确率与合法终止率并列，token 成本在表内保留。</p>
            </div>
            <span className="unit">fixed200</span>
          </div>
          <div className="research-short-chart">
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 680, height: 320 }}>
              <BarChart data={fixed200Data} margin={{ top: 22, right: 16, bottom: 28, left: 2 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" interval={0} tick={{ fontSize: 10 }} />
                <YAxis domain={[60, 102]} tickFormatter={(value) => `${value}%`} />
                <Tooltip formatter={(value, name) => [`${value}%`, name === "accuracyPct" ? "正确率" : "合法终止率"]} />
                <Legend formatter={(value) => value === "accuracyPct" ? "正确率" : "合法终止率"} />
                <Bar dataKey="accuracyPct" fill="#087f5b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="legalPct" fill="#9ab4a8" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="native-cost-list">
            {fixed200Data.map((item) => (
              <div key={item.id}>
                <strong>{item.label}</strong>
                <span>{item.correct}/200 correct · {Number.isFinite(item.errors) ? `${item.errors} errors` : "errors 未报告"}</span>
                <code>{Number.isFinite(item.tokens) ? `${compact.format(item.tokens)} tokens` : "tokens 未报告"}</code>
              </div>
            ))}
          </div>
        </div>

        <div className="panel diagnostics-panel">
          <div className="panel-heading">
            <div>
              <h2>工具设计实验数据矩阵</h2>
              <p>每行列出实验 cohort、correct、reference 和数值差。</p>
            </div>
            <span className="unit">{diagnostics.length} runs</span>
          </div>
          <div className="diagnostic-filter">
            {diagnosticFamilies.map((family) => (
              <button key={family} className={diagnosticFamily === family ? "active" : ""} onClick={() => setDiagnosticFamily(family)}>
                {family === "all" ? "全部" : family}
              </button>
            ))}
          </div>
          <div className="diagnostic-list">
            {diagnostics.map((item) => {
              const delta = item.correct - item.reference_correct;
              return (
                <article key={`${item.label}-${item.cohort}`}>
                  <div><span>{item.family} · {item.cohort}</span><strong>{item.label}</strong></div>
                  <div className="diagnostic-score"><strong>{item.correct}/{item.total}</strong><span>vs {item.reference} {item.reference_correct}/{item.total}</span></div>
                  <b className={delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral"}>{delta > 0 ? "+" : ""}{delta}</b>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section className="panel artifact-panel">
        <div className="panel-heading artifact-heading">
          <div>
            <h2>全部实验数据与报告入口</h2>
            <p>由两周总索引自动抽取并去重；远端路径不在页面加载时执行 SSH。</p>
          </div>
          <span className="unit">显示 {artifacts.length} / {summary.artifacts.length}</span>
        </div>
        <div className="artifact-toolbar">
          <div className="diagnostic-filter">
            {Object.entries(artifactCategoryLabels).map(([category, label]) => (
              <button key={category} className={artifactCategory === category ? "active" : ""} onClick={() => setArtifactCategory(category)}>{label}</button>
            ))}
          </div>
          <input value={artifactQuery} onChange={(event) => setArtifactQuery(event.target.value)} placeholder="搜索目录、报告或实验名" />
        </div>
        <div className="artifact-list">
          {artifacts.map((item) => (
            <article key={item.path}>
              <span className={`artifact-state ${item.remote ? "remote" : item.available ? "available" : "missing"}`}>
                {item.remote ? <Server size={13} /> : item.category === "reports" ? <FileJson size={13} /> : <Database size={13} />}
                {item.remote ? "REMOTE" : item.available ? item.kind.toUpperCase() : "MISSING"}
              </span>
              <code title={item.path}>{item.path}</code>
              <button onClick={() => copyPath(item.path)}>{copiedPath === item.path ? "已复制" : "复制路径"}</button>
            </article>
          ))}
          {!artifacts.length ? <Empty title="没有匹配路径" detail="清除筛选或换一个关键词。" /> : null}
        </div>
      </section>
    </main>
  );
}

function Overview({ experiments, onOpen }) {
  const boards = useMemo(() => {
    const groups = new Map();
    for (const item of experiments) {
      const benchmark = benchmarkFor(item);
      if (!groups.has(benchmark)) groups.set(benchmark, []);
      groups.get(benchmark).push(item);
    }
    return [...groups.entries()].map(([id, items]) => ({
      id,
      ...(BENCHMARKS[id] || { label: id, scope: "", description: "" }),
      items,
    }));
  }, [experiments]);
  const [selectedBoard, setSelectedBoard] = useState("");
  const preferredBoard = boards.find((board) => board.id === "spider")?.id || boards[0]?.id || "";
  useEffect(() => {
    if (!boards.length) return;
    if (!selectedBoard || !boards.some((board) => board.id === selectedBoard)) {
      setSelectedBoard(preferredBoard);
    }
  }, [boards, preferredBoard, selectedBoard]);
  const activeBoard = boards.find((board) => board.id === (selectedBoard || preferredBoard));
  const boardExperiments = useMemo(() => activeBoard?.items || [], [activeBoard]);
  const families = useMemo(() => {
    const groups = new Map();
    for (const item of boardExperiments) {
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
  }, [boardExperiments]);
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
  const chartData = visibleExperiments
    .map((item) => ({ item, summary: displayEvaluationSummary(item) }))
    .filter(({ summary }) => summary?.available)
    .map(({ item, summary }) => {
      const metric = primaryMetric(summary);
      return {
        name: `${modelAbbrev(item.model)} ${item.short_name}`,
        accuracy: Number(((metric.value || 0) * 100).toFixed(2)),
        metricLabel: metric.label,
        legal: Number.isFinite(summary?.legal_rate)
          ? Number((summary.legal_rate * 100).toFixed(2))
          : null,
        evalLoss: item.training_metrics?.summary?.last_eval_loss,
        kind: item.kind || "sft",
        family: modelFamily(item.model),
      };
    });
  const passKeys = ["1", "2", "4", "8", "16", "32"];
  const overviewPassData = visibleExperiments.flatMap((item) =>
    (item.evaluation_runs || [])
      .filter((run) => run.summary?.available && run.summary.pass_at)
      .map((run) => {
        const row = {
          name: `${item.short_name} · ${run.label}`,
          family: modelFamily(item.model),
          kind: item.kind || "sft",
        };
        for (const key of passKeys) {
          if (run.summary.pass_at?.[key]) {
            row[`pass${key}`] = Number((run.summary.pass_at[key].rate * 100).toFixed(2));
          }
        }
        return row;
      })
  );
  const trainedRuns = visibleExperiments.filter((item) => item.training_metrics?.available).length;
  const best = [...visibleExperiments].sort(
    (a, b) => (displayEvaluationSummary(b)?.accuracy || 0) - (displayEvaluationSummary(a)?.accuracy || 0),
  )[0];

  return (
    <main className="content">
      <header className="page-header">
        <div>
          <p className="eyebrow">Research workspace</p>
          <h1>{activeBoard?.label || "实验"} 看板</h1>
          <p>{activeBoard?.description || "按数据集隔离的 baseline、SFT 与工具评测。"}</p>
        </div>
        <div className="header-date">
          <Clock3 size={16} />
          数据更新于本地实验产物
        </div>
      </header>

      <section className="model-switch-panel benchmark-switch-panel">
        <div>
          <span>评测与构造数据集</span>
          <strong>{activeBoard?.label || "未选择"}</strong>
          <small>{activeBoard?.scope}</small>
        </div>
        <div className="segmented-control" aria-label="选择数据集看板">
          {boards.map((board) => (
            <button
              key={board.id}
              className={board.id === activeBoard?.id ? "active" : ""}
              onClick={() => setSelectedBoard(board.id)}
            >
              {board.label}
              <span>{board.items.length}</span>
            </button>
          ))}
        </div>
      </section>

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
          value={best ? pct(displayEvaluationSummary(best)?.accuracy) : "—"}
          detail={best?.short_name}
          icon={Gauge}
        />
        <Metric
          label="累计评测案例"
          value={compact.format(
            visibleExperiments.reduce((sum, item) => sum + (displayEvaluationSummary(item)?.total || 0), 0),
          )}
          icon={Database}
        />
      </section>

      <section className="overview-grid">
        <div className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <h2>评测表现</h2>
              <p>{activeBoard?.scope}；执行准确率与合法回答率</p>
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
                <Bar dataKey="accuracy" name="Acc / pass@k" fill="#087f5b" radius={[3, 3, 0, 0]}>
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
              <p>{activeBoard?.label} 数据集下的版本、训练与评测</p>
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
                      {(() => {
                        const metric = primaryMetric(displayEvaluationSummary(item));
                        return (
                          <>
                            <strong>{pct(metric.value)}</strong>
                            <span>{metric.label} · {metric.correct}/{metric.total}</span>
                          </>
                        );
                      })()}
                    </div>
                    <ChevronRight size={18} />
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      </section>

      {overviewPassData.length ? (
        <section className="panel passk-overview">
          <div className="panel-heading">
            <div>
              <h2>Pass@ 采样对比</h2>
              <p>{activeBoard?.label} 内同一模型家族的采样评测与 baseline 对比</p>
            </div>
            <span className="unit">%</span>
          </div>
          <div className="chart passk-overview-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={overviewPassData} barGap={3}>
                <CartesianGrid vertical={false} stroke="#e1e5e0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} interval={0} angle={-12} height={58} />
                <YAxis domain={[0, 100]} axisLine={false} tickLine={false} width={34} />
                <Tooltip contentStyle={{ borderRadius: 6, borderColor: "#cfd5cf" }} />
                <Legend />
                <Bar dataKey="pass1" name="pass@1" fill="#4976a8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pass2" name="pass@2" fill="#087f5b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pass4" name="pass@4" fill="#d97706" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pass8" name="pass@8" fill="#7c3aed" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pass16" name="pass@16" fill="#be185d" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pass32" name="pass@32" fill="#334155" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      ) : null}

      <section className="panel baseline-map">
        <div className="panel-heading">
          <div>
            <h2>Baseline 对齐</h2>
              <p>{activeBoard?.label} 内每个模型家族只和自己的 Direct SQL baseline 对比</p>
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
              <p>{activeBoard?.label} 看板中的训练数据规模、loss 与推理表现</p>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>实验</th>
                <th>模型家族</th>
                <th>评测数据集</th>
                <th>类型</th>
                <th>训练样本</th>
                <th>Train loss</th>
                <th>Eval loss</th>
                <th>Acc / pass@k</th>
                <th>相对同系列 Direct SQL</th>
                <th>平均步骤</th>
                <th>评测完整性</th>
              </tr>
            </thead>
            <tbody>
              {visibleExperiments.map((item) => {
                const reference = directSqlByFamily.get(modelFamily(item.model));
                const summary = displayEvaluationSummary(item);
                const metric = primaryMetric(summary);
                return (
                  <tr key={item.id} onClick={() => onOpen(item.id)}>
                    <td><strong>{item.short_name}</strong></td>
                    <td>{modelFamily(item.model)}</td>
                    <td>{item.dataset_version || activeBoard?.label || "—"}</td>
                    <td><span className={`type-badge ${item.kind === "baseline" ? "baseline" : ""}`}>{item.kind === "baseline" ? "Baseline" : "SFT"}</span></td>
                    <td>{item.training_metrics?.available ? compact.format(item.dataset_summary?.train?.kept || 0) : "—"}</td>
                    <td>{fixed(item.training_metrics?.summary?.train_loss)}</td>
                    <td>{fixed(item.training_metrics?.summary?.last_eval_loss)}</td>
                    <td>
                      <strong>{pct(metric.value)}</strong>
                      <span className="metric-label-inline">{metric.label}</span>
                    </td>
                    <td>
                      {reference && item.id !== reference.id
                        ? `${((summary?.accuracy - reference.evaluation_summary.accuracy) * 100).toFixed(2)} pp`
                        : reference ? "reference" : "缺少同系列 baseline"}
                    </td>
                    <td>{fixed(item.evaluation_summary?.average_steps, 2)}</td>
                    <td>
                      {summary?.complete ? (
                        <span className="inline-ok"><Check size={14} /> {summary.expected_total}/{summary.expected_total}</span>
                      ) : (
                        <span className="inline-warn">
                          <CircleAlert size={14} /> {summary?.expected_total ? `${summary.total || 0}/${summary.expected_total}` : `${summary?.total || 0}/—`}
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
                const isEval = Array.isArray(item.record.turns) || Array.isArray(item.record.samples);
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
                Array.isArray(active.record.samples) ? (
                  <PassKRecordView record={active.record} />
                ) : Array.isArray(active.record.turns) ? (
                  <TrajectoryView record={active.record} />
                ) : (
                  <StructuredRecord record={active.record} />
                )
              ) : null}
              {active && view === "json" ? <JsonBlock value={active.record} /> : null}
            </div>
          </div>
          <PaginationControls page={data.page} pages={data.pages} onPageChange={setPage} />
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
  const evaluationRuns = experiment.evaluation_runs || [];
  const evaluation = displayEvaluationSummary(experiment);
  const passChartData = evaluationRuns
    .filter((run) => run.summary?.available)
    .map((run) => {
      const entries = Object.entries(run.summary.pass_at || {}).sort(
        ([a], [b]) => Number(a) - Number(b),
      );
      const [passKey, passValue] = entries[entries.length - 1] || [
        "acc",
        {
          correct: run.summary.correct,
          total: run.summary.total,
          rate: run.summary.accuracy,
        },
      ];
      return {
        name: run.label,
        passKey: passKey === "acc" ? "acc" : `pass@${passKey}`,
        score: Number(((passValue?.rate || 0) * 100).toFixed(2)),
        legal: Number.isFinite(run.summary.legal_rate)
          ? Number((run.summary.legal_rate * 100).toFixed(2))
          : null,
        correct: passValue?.correct ?? run.summary.correct,
        total: passValue?.total ?? run.summary.total,
        directory: run.directory,
      };
    });
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
                <p>{evaluation?.total || 0} 个 {experiment.dataset_version || "评测"} 案例</p>
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
                <dd className={evaluation?.complete ? "text-ok" : "text-warn"}>
                  {evaluation?.complete ? "完整" : evaluation?.expected_total ? `${evaluation?.total || 0}/${evaluation.expected_total}` : "未声明"}
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
          {passChartData.length ? (
            <section className="panel passk-panel">
              <div className="panel-heading">
                <div>
                  <h2>Pass@ 对比</h2>
                  <p>主评测和采样评测的正确率、合法回答率</p>
                </div>
              </div>
              <div className="passk-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={passChartData} margin={{ top: 18, right: 18, left: 4, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="passKey" tickLine={false} axisLine={false} />
                    <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} width={42} />
                    <Tooltip
                      formatter={(value, name) => [
                        `${Number(value).toFixed(2)}%`,
                        name === "score" ? "Pass / Acc" : "Legal",
                      ]}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ""}
                    />
                    <Legend />
                    <Bar dataKey="score" name="Pass / Acc" fill="#087f5b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="legal" name="Legal" fill="#4c78a8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="passk-run-list">
                {passChartData.map((run) => (
                  <div key={run.name} className="passk-run">
                    <div>
                      <strong>{run.name}</strong>
                      <span>{run.directory}</span>
                    </div>
                    <dl>
                      <div><dt>{run.passKey}</dt><dd>{run.score.toFixed(2)}%</dd></div>
                      <div><dt>正确</dt><dd>{run.correct}/{run.total}</dd></div>
                      <div><dt>合法</dt><dd>{Number.isFinite(run.legal) ? `${run.legal.toFixed(2)}%` : "—"}</dd></div>
                    </dl>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
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
          <button className={view === "research" ? "active" : ""} onClick={() => onView("research")}>
            <Gauge size={19} />
            {!collapsed ? <span>研究总览</span> : null}
          </button>
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
  const [researchSummary, setResearchSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [view, setView] = useState("research");
  const [selectedId, setSelectedId] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const load = async () => {
    try {
      setError("");
      const [nextExperiments, nextResearchSummary] = await Promise.all([
        api.experiments(),
        api.researchSummary(),
      ]);
      setExperiments(nextExperiments);
      setResearchSummary(nextResearchSummary);
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
      ) : view === "research" ? (
        <ResearchOverview summary={researchSummary} />
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
