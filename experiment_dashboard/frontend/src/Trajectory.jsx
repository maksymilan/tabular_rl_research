/* eslint-disable react-refresh/only-export-components */
import { useEffect, useMemo, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Clock,
  Eye,
  FileJson,
  Search,
  Table2,
  Target,
  Wrench,
  X,
} from "lucide-react";
import { ConversationView } from "./JsonViewer";

// Perception tools (read-only probes). Highlighted distinctly in the trajectory because the
// research question is whether the model probes the data at a DECISION point or only as a
// scripted pre-answer ritual.
const PERCEPTION = new Set(["describe_table", "inspect_column", "read_subtable"]);
const TOOL_ICON = {
  describe_table: Table2,
  inspect_column: Search,
  read_subtable: Eye,
  answer_from_context: Target,
};

// Bucket → display label + accent colour (CSS theme variables from styles.css).
export const BUCKET_META = {
  correct: { label: "正确", color: "var(--green)" },
  empty_pred: { label: "空结果 · 过滤过头", color: "var(--red)" },
  arity_mismatch: { label: "投影错列 · 列数不符", color: "var(--amber)" },
  cardinality_mismatch: { label: "行数不符", color: "var(--blue)" },
  value_mismatch: { label: "取值错 · 列/值/聚合", color: "var(--red)" },
  execution_error: { label: "执行错误", color: "var(--amber)" },
  protocol_error: { label: "协议/格式错误", color: "var(--muted)" },
  api_error: { label: "API 错误", color: "var(--muted)" },
  max_steps: { label: "超出最大步数", color: "var(--muted)" },
  unknown: { label: "未知", color: "var(--muted)" },
};

function sampleShape(sample) {
  if (!sample || !sample.length) return [0, 0];
  const rows = sample.map((r) => (Array.isArray(r) ? r : [r]));
  return [rows.length, Math.max(...rows.map((r) => r.length))];
}

// Client-side mirror of backend/attribution.py:attribute_record — classifies one eval case
// from its own fields so each browsed record can show a bucket badge without an extra fetch.
export function attributeRecord(record) {
  const tools = (record.turns || [])
    .map((t) => (t.parsed && t.parsed.tool) || null)
    .filter(Boolean);
  const usedRead = tools.includes("read_subtable");
  const decisionRead = tools.some((t, i) => t === "read_subtable" && i < tools.length - 2);
  const base = { tools, usedRead, decisionRead };
  if (record.correct === true) return { ...base, bucket: "correct" };
  if (record.failure_type !== "wrong_answer") {
    return { ...base, bucket: record.failure_type || "unknown" };
  }
  const predShape = sampleShape(record.pred_sample);
  const goldShape = sampleShape(record.gold_sample);
  let bucket;
  if (predShape[0] === 0 && goldShape[0] > 0) bucket = "empty_pred";
  else if (predShape[1] !== goldShape[1]) bucket = "arity_mismatch";
  else if (predShape[0] !== goldShape[0]) bucket = "cardinality_mismatch";
  else bucket = "value_mismatch";
  return { ...base, bucket, predShape, goldShape };
}

function outputSummary(output) {
  if (output == null) return "";
  if (Array.isArray(output)) return `${output.length} 项`;
  if (typeof output !== "object") return String(output);
  if (Array.isArray(output.tables)) return `表 ${output.tables.map((t) => t.table_name).join(", ")}`;
  if (Array.isArray(output.rows)) return `${output.row_count ?? output.rows.length} 行`;
  if (output.row_count != null) {
    const cols = Array.isArray(output.columns) ? output.columns.length : "?";
    return `${output.kind || output.table || "result"} · ${output.row_count} 行 · ${cols} 列`;
  }
  if (output.distinct_count != null) return `distinct ${output.distinct_count}`;
  return "";
}

function normalizeMessage(message) {
  if (!message) return null;
  return {
    role: message.role || message.from || "user",
    content: message.content ?? message.value ?? "",
  };
}

function contextDiff(current, next) {
  if (!Array.isArray(current) || !Array.isArray(next) || next.length <= current.length) {
    return [];
  }
  return next.slice(current.length).map(normalizeMessage).filter(Boolean);
}

function observationMessage(turn, index) {
  if (turn.context_added?.length) return null;
  if (turn.tool_output == null && !turn.execution_error && !turn.api_error) return null;
  const payload = turn.execution_error || turn.api_error
    ? {
        step_id: `step_${index + 1}`,
        status: "error",
        error: {
          type: turn.execution_error_type || turn.api_error_type || "error",
          message: turn.execution_error || turn.api_error,
        },
      }
    : { step_id: `step_${index + 1}`, status: "success", output: turn.tool_output };
  return { role: "user", content: JSON.stringify(payload) };
}

function ContextPanel({ title, messages, empty = "无新增上下文" }) {
  if (!messages?.length) {
    return (
      <details className="traj-context">
        <summary>{title} · 0 条</summary>
        <p className="traj-empty-inline">{empty}</p>
      </details>
    );
  }
  return (
    <details className="traj-context">
      <summary>{title} · {messages.length} 条</summary>
      <ConversationView messages={messages} />
    </details>
  );
}

function Sample({ title, rows, tone }) {
  const preview = (rows || []).slice(0, 6).map((r) => (Array.isArray(r) ? r : [r]));
  return (
    <div className={`traj-sample ${tone}`}>
      <span className="traj-sample-title">{title}</span>
      {preview.length ? (
        <table>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell === null ? "NULL" : String(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <em>空结果</em>
      )}
      {(rows || []).length > 6 ? <span className="traj-more">… 共 {rows.length} 行</span> : null}
    </div>
  );
}

// The opening DATASET OVERVIEW catalog (lazy: table names + row counts + FK relations only).
function CatalogPreview({ overview }) {
  const tables = overview.tables || [];
  const relations = overview.relations || [];
  return (
    <details className="traj-catalog">
      <summary>
        开场目录 · {tables.length} 表{relations.length ? ` · ${relations.length} 关系` : ""}
      </summary>
      <div className="catalog-tables">
        {tables.map((t, i) => (
          <span key={i} className="catalog-table">
            {t.table_name}
            <em>{t.num_rows ?? t.row_count ?? "?"} 行</em>
          </span>
        ))}
      </div>
      {relations.length ? (
        <div className="catalog-rels">
          {relations.map((r, i) => (
            <code key={i}>
              {r.from} → {r.to}
            </code>
          ))}
        </div>
      ) : null}
    </details>
  );
}

// Step-by-step view of one rollout: think → tool call → tool output, per turn.
export function TrajectoryView({ record }) {
  const attr = attributeRecord(record);
  const meta = BUCKET_META[attr.bucket] || BUCKET_META.unknown;
  const turns = record.turns || [];
  return (
    <div className="trajectory">
      <div className="traj-head">
        {record.isTraining ? (
          <span className="traj-verdict train">gold 轨迹</span>
        ) : (
          <>
            <span className={`traj-verdict ${record.correct ? "ok" : "bad"}`}>
              {record.correct ? <Check size={13} /> : <X size={13} />}
              {record.correct ? "正确" : "失败"}
            </span>
            <span className="bucket-badge" style={{ "--badge": meta.color }}>
              {meta.label}
            </span>
          </>
        )}
        {record.db_id ? <span className="traj-db">{record.db_id}</span> : null}
        <span className="traj-stepcount">{turns.length} 步</span>
      </div>
      <p className="traj-question">{record.question}</p>
      {record.gold_sql ? <code className="traj-gold">{record.gold_sql}</code> : null}
      {record.db_overview ? <CatalogPreview overview={record.db_overview} /> : null}
      {!record.correct ? (
        <div className="traj-samples">
          <Sample title={`预测 ${attr.predShape ? attr.predShape.join("×") : ""}`} rows={record.pred_sample} tone="bad" />
          <Sample title={`答案 ${attr.goldShape ? attr.goldShape.join("×") : ""}`} rows={record.gold_sample} tone="ok" />
        </div>
      ) : null}
      <ol className="traj-steps">
        {turns.map((turn, i) => {
          const nextTurn = turns[i + 1];
          const modelInput = Array.isArray(turn.model_input) ? turn.model_input.map(normalizeMessage).filter(Boolean) : [];
          const addedFromNext = contextDiff(turn.model_input, nextTurn?.model_input);
          const fallbackObservation = observationMessage(turn, i);
          const contextAdded = turn.context_added?.length
            ? turn.context_added.map(normalizeMessage).filter(Boolean)
            : addedFromNext.length
              ? addedFromNext
              : fallbackObservation ? [fallbackObservation] : [];
          const tool = turn.parsed && turn.parsed.tool;
          const Icon = TOOL_ICON[tool] || Wrench;
          const isError = !!turn.error_attempt;
          const isPerception = !isError && (turn.perception != null ? turn.perception : PERCEPTION.has(tool));
          const isRitualRead = tool === "read_subtable" && i >= turns.length - 2;
          const cls = isError ? " error-attempt" : isPerception ? " perception" : "";
          return (
            <li key={i} className={`traj-step${cls}`}>
              <div className="traj-step-head">
                <Icon size={13} />
                <strong>{tool || "—"}</strong>
                {isError ? (
                  <span className="traj-tag err">纠错 · 错误尝试</span>
                ) : isPerception ? (
                  <span className={`traj-tag${isRitualRead ? " ritual" : ""}`}>
                    {tool === "read_subtable" ? (isRitualRead ? "答案前固定读" : "决策点主动读") : "感知"}
                  </span>
                ) : null}
                <span className="traj-step-no">#{i}</span>
              </div>
              {turn.parsed && turn.parsed.think ? <p className="traj-think">{turn.parsed.think}</p> : null}
              {turn.parsed && turn.parsed.arguments ? (
                <code className="traj-args">{JSON.stringify(turn.parsed.arguments)}</code>
              ) : null}
              {modelInput.length ? (
                <ContextPanel title="本步前模型看到的上下文" messages={modelInput} empty="该记录没有保存 model_input" />
              ) : null}
              <ContextPanel title="本步后加入上下文" messages={contextAdded} />
              {turn.tool_output != null ? (
                <details className="traj-output">
                  <summary>{outputSummary(turn.tool_output) || "tool_output"}</summary>
                  <pre>{JSON.stringify(turn.tool_output, null, 2)}</pre>
                </details>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

// Aggregate error-attribution panel for a whole run: bucket distribution + the probe diagnosis.
export function AttributionPanel({ experimentId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setData(null);
    setError("");
    fetch(`/api/experiments/${experimentId}/attribution`)
      .then((r) => r.json())
      .then((payload) => {
        if (payload.error) throw new Error(payload.error);
        setData(payload);
      })
      .catch((reason) => setError(reason.message));
  }, [experimentId]);

  if (error) return <p className="traj-empty">归因端点不可用:{error}</p>;
  if (!data) return <div className="loading-line" />;
  if (!data.available) return <p className="traj-empty">该实验暂无评测产物(all.jsonl)。</p>;

  const wa = data.wrong_answer;
  const failTotal = data.total - (data.buckets.correct || 0);
  const failBuckets = Object.entries(data.buckets)
    .filter(([key]) => key !== "correct")
    .sort((a, b) => b[1] - a[1]);
  const pct = (n) => (wa.total ? Math.round((n / wa.total) * 100) : 0);

  return (
    <div className="attribution">
      <div className="attr-headline">
        <div>
          <strong>{data.buckets.correct || 0}</strong> / {data.total} 正确
        </div>
        <span>失败 {failTotal} 例,按错误发生的位置归因</span>
      </div>
      <div className="failure-bars">
        {failBuckets.map(([key, count]) => {
          const meta = BUCKET_META[key] || BUCKET_META.unknown;
          return (
            <div key={key}>
              <div className="failure-label">
                <span>{meta.label}</span>
                <strong>{count}</strong>
              </div>
              <div className="progress">
                <span style={{ width: `${(count / Math.max(1, failTotal)) * 100}%`, background: meta.color }} />
              </div>
            </div>
          );
        })}
      </div>
      <div className="insight-callout">
        <Eye size={16} />
        <div>
          <strong>主动查表诊断（wrong_answer 子集）</strong>
          {wa.read_subtable === 0 ? (
            <p>
              该数据形态下模型不调用 read_subtable —— 中间结果以行内联呈现,无需主动读。
              {wa.total} 个错误答案均为推理/取值错,而非可见性缺口。
            </p>
          ) : (
            <p>
              {wa.total} 个错误答案中,{wa.read_subtable} 个（{pct(wa.read_subtable)}%）调用过 read_subtable,
              但只有 <b>{wa.decision_point_read}</b> 个（{pct(wa.decision_point_read)}%）在<b>决策点</b>读表;
              其余都是答案前的固定读,救不了上游已经选错的列/过滤。inspect_column 使用 {wa.inspect_column} 次。
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- Training data (LLaMA-Factory sharegpt) → trajectory ---------------------------------------
// The SFT record is {system, conversations:[{from:"human"|"gpt"|"observation", value}]}. The first
// human turn is "DATASET OVERVIEW\n{json}\n\nQUESTION\n{q}"; each gpt turn is
// "<think>..</think><tool_call>{json}</tool_call>"; each observation is the tool output envelope.

function parseAssistantTurn(value) {
  const think = (value.match(/<think>([\s\S]*?)<\/think>/) || [])[1] || "";
  const call = (value.match(/<tool_call>([\s\S]*?)<\/tool_call>/) || [])[1];
  let tool = null;
  let args = null;
  if (call) {
    try {
      const parsed = JSON.parse(call.trim());
      tool = parsed.tool;
      args = parsed.arguments;
    } catch {
      /* leave unparsed */
    }
  }
  return { think: think.trim(), tool, arguments: args };
}

export function extractQuestion(record) {
  const human = (record.conversations || []).find((c) => c.from === "human");
  if (!human) return record.question || "训练样例";
  const match = human.value.match(/QUESTION\s*\n([\s\S]*)$/);
  return match ? match[1].trim() : human.value.slice(0, 80);
}

// Normalise a sharegpt training record into the shape TrajectoryView already renders.
export function sftToTrajectory(record) {
  const conv = record.conversations || [];
  const human = conv.find((c) => c.from === "human");
  let question = "";
  let overview = null;
  if (human) {
    const q = human.value.match(/QUESTION\s*\n([\s\S]*)$/);
    question = q ? q[1].trim() : human.value.slice(0, 200);
    const o = human.value.match(/DATASET OVERVIEW\s*\n([\s\S]*?)\n\s*QUESTION/);
    if (o) {
      try {
        overview = JSON.parse(o[1].trim());
      } catch {
        overview = null;
      }
    }
  }
  const turns = [];
  const context = [
    ...(record.system ? [{ role: "system", content: record.system }] : []),
    ...(human ? [{ role: "user", content: human.value }] : []),
  ];
  for (let i = 0; i < conv.length; i += 1) {
    if (conv[i].from !== "gpt") continue;
    const parsed = parseAssistantTurn(conv[i].value);
    const assistant = { role: "assistant", content: conv[i].value };
    let tool_output = null;
    let observation = null;
    const next = conv[i + 1];
    if (next && next.from === "observation") {
      observation = { role: "observation", content: next.value };
      try {
        const envelope = JSON.parse(next.value);
        tool_output = envelope.output ?? envelope;
      } catch {
        tool_output = next.value;
      }
    }
    turns.push({
      model_input: context.map((message) => ({ ...message })),
      context_added: [assistant, observation].filter(Boolean),
      parsed,
      tool_output,
    });
    context.push(assistant);
    if (observation) context.push(observation);
  }
  return { question, db_overview: overview, correct: true, isTraining: true, turns };
}

function LengthHistogram({ hist, active, onSelect }) {
  const entries = Object.entries(hist || {})
    .map(([k, v]) => [Number(k), v])
    .sort((a, b) => a[0] - b[0]);
  if (!entries.length) return null;
  const max = Math.max(1, ...entries.map((e) => e[1]));
  return (
    <div className="len-hist">
      <span className="len-hist-title">轨迹步数分布{onSelect ? " · 点击筛选" : ""}</span>
      <div className="len-hist-bars">
        {entries.map(([len, count]) => (
          <button
            key={len}
            type="button"
            className={`len-bar${active === len ? " active" : ""}`}
            title={`${len} 步 · ${count} 条`}
            onClick={onSelect ? () => onSelect(len) : undefined}
          >
            <span style={{ height: `${(count / max) * 100}%` }} />
            <em>{len}</em>
          </button>
        ))}
      </div>
    </div>
  );
}

// Training-data board: manifest aggregates + a per-example trajectory browser (train / dev).
export function TrainingBoard({ experiment }) {
  const manifest = experiment.dataset_summary?.train || {};
  const sourceOptions = useMemo(() => {
    const availableSourceIds = new Set((experiment.data_sources || []).map((item) => item.id));
    return [
      ["train", "训练集"],
      ["dev", "验证集"],
    ].filter(([value]) => availableSourceIds.has(value));
  }, [experiment.data_sources]);
  const [source, setSource] = useState(sourceOptions[0]?.[0] || "train");
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(0);
  const [view, setView] = useState("traj");
  const [lengthFilter, setLengthFilter] = useState(null); // exact trajectory length, or null

  useEffect(() => {
    if (!sourceOptions.some(([value]) => value === source)) {
      setSource(sourceOptions[0]?.[0] || "");
      setPage(1);
      return;
    }
    if (!source) return;
    setData(null);
    setError("");
    const lengthQuery = lengthFilter == null ? "" : `&min_steps=${lengthFilter}&max_steps=${lengthFilter}`;
    fetch(`/api/experiments/${experiment.id}/records?source=${source}&page=${page}&page_size=8${lengthQuery}`)
      .then((r) => r.json())
      .then((payload) => {
        if (payload.error) throw new Error(payload.error);
        setData(payload);
        setSelected(0);
      })
      .catch((reason) => setError(reason.message));
  }, [experiment.id, source, page, lengthFilter, sourceOptions]);

  const selectLength = (len) => {
    setLengthFilter((current) => (current === len ? null : len));
    setPage(1);
  };

  const active = data?.records?.[selected];
  const traj = active ? sftToTrajectory(active.record) : null;
  const tokens = manifest.est_tokens || {};
  const keptRecords = manifest.kept ?? manifest.kept_records ?? manifest.source_trajectories ?? 0;
  const droppedOverlong = manifest.dropped_overlong ?? manifest.dropped_over_max_tokens ?? 0;
  const tokenMax = tokens.max ?? manifest.max_tokens ?? "—";

  return (
    <div className="training-board">
      <div className="train-stats">
        <div className="train-stat">
          <strong>{keptRecords.toLocaleString()}</strong>
          <span>训练轨迹</span>
        </div>
        <div className="train-stat">
          <strong>{tokens.p50 ?? "—"}</strong>
          <span>tokens p50</span>
        </div>
        <div className="train-stat">
          <strong>{tokenMax}</strong>
          <span>tokens max</span>
        </div>
        <div className="train-stat">
          <strong>{droppedOverlong}</strong>
          <span>超长丢弃</span>
        </div>
        <LengthHistogram
          hist={manifest.trajectory_length_hist}
          active={lengthFilter}
          onSelect={selectLength}
        />
      </div>

      <div className="train-controls">
        <div className="segmented" aria-label="数据划分">
          {sourceOptions.map(([value, label]) => (
            <button
              key={value}
              className={source === value ? "active" : ""}
              onClick={() => {
                setSource(value);
                setPage(1);
              }}
            >
              {label}
            </button>
          ))}
        </div>
        {lengthFilter != null ? (
          <button type="button" className="length-chip" onClick={() => selectLength(lengthFilter)}>
            仅 {lengthFilter} 步轨迹
            {data ? ` · ${data.total} 条` : ""}
            <X size={12} />
          </button>
        ) : null}
      </div>

      {error ? <p className="traj-empty">读取失败:{error}</p> : null}
      {!data && !error ? <div className="loading-line" /> : null}
      {data ? (
        <>
          <div className="record-layout">
            <div className="record-index">
              {data.records.map((item, index) => (
                <button
                  key={item.index}
                  className={selected === index ? "active" : ""}
                  onClick={() => setSelected(index)}
                >
                  <span>#{item.index + 1}</span>
                  <strong>{extractQuestion(item.record)}</strong>
                </button>
              ))}
            </div>
            <div className={view === "traj" ? "record-json light" : "record-json"}>
              <div className="json-toolbar">
                <div className="view-toggle">
                  <button className={view === "traj" ? "active" : ""} onClick={() => setView("traj")}>
                    动作轨迹
                  </button>
                  <button className={view === "json" ? "active" : ""} onClick={() => setView("json")}>
                    JSON
                  </button>
                </div>
                <span>{data.total} 条</span>
              </div>
              {active ? (
                view === "traj" ? (
                  <TrajectoryView record={traj} />
                ) : (
                  <pre className="json-block">{JSON.stringify(active.record, null, 2)}</pre>
                )
              ) : null}
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

// ---- Data-construction review: LLM-enriched reflection/perception trajectories ------------------
const MODE_META = {
  full: { label: "感知+纠错", color: "var(--green)" },
  perception_only: { label: "仅感知", color: "var(--blue)" },
  semantic_rewrite: { label: "语义重写", color: "var(--blue)" },
  skeleton: { label: "未富化·骨架", color: "var(--muted)" },
};

function enrichmentField(record, field, fallback = undefined) {
  return record?.[field] ?? record?.enrichment?.[field] ?? fallback;
}

function normalizeConstructionFile(item) {
  return typeof item === "string" ? { name: item } : item;
}

function finiteNumber(value) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatBytes(bytes) {
  const value = finiteNumber(bytes);
  if (value == null) return "未知大小";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatFileTime(item) {
  if (item.mtime_iso) return item.mtime_iso;
  const value = finiteNumber(item.mtime);
  if (value == null) return "未知时间";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value * 1000));
}

function GeneratorTrace({ record }) {
  const enrichment = record?.enrichment || {};
  const generator = enrichment.generator || {};
  const history = enrichment.annotation_history || [];
  if (!generator.model && !history.length) return null;
  return (
    <details className="generator-trace">
      <summary>
        生成模型 · {generator.model || "unknown"}
        {generator.mode_requested ? ` · ${generator.mode_requested}` : ""}
      </summary>
      {history.map((item, index) => (
        <div className="generator-attempt" key={index}>
          <div className="generator-attempt-head">
            <strong>Attempt {item.attempt ?? index + 1}</strong>
            {item.ok ? <span className="text-ok">accepted</span> : null}
            {item.issues ? <span className="text-warn">{item.issues}</span> : null}
            {item.error ? <span className="text-warn">{item.error}</span> : null}
          </div>
          {item.usage ? (
            <code className="traj-args">usage {JSON.stringify(item.usage)}</code>
          ) : null}
          {item.model_output ? (
            <details className="traj-output" open={history.length === 1}>
              <summary>外部模型原始输出</summary>
              <pre>{item.model_output}</pre>
            </details>
          ) : null}
        </div>
      ))}
    </details>
  );
}

function stepsToTrajectory(record, steps, extra = {}) {
  const turns = [];
  const context = [];
  if (record.system) context.push({ role: "system", content: record.system });
  if (record.question || record.initial_state?.dataset_overview) {
    context.push({
      role: "user",
      content: [
        record.initial_state?.dataset_overview
          ? `DATASET OVERVIEW\n${JSON.stringify(record.initial_state.dataset_overview)}`
          : "",
        record.question ? `QUESTION\n${record.question}` : "",
      ].filter(Boolean).join("\n\n"),
    });
  }
  for (const s of steps || []) {
    const assistant = {
      role: "assistant",
      content: [
        s.think ? `<think>${s.think}</think>` : "",
        s.tool_call ? `<tool_call>${JSON.stringify(s.tool_call)}</tool_call>` : "",
      ].filter(Boolean).join("\n"),
    };
    const observation = s.tool_output == null ? null : {
      role: "observation",
      content: JSON.stringify({
        step_id: s.step_id,
        status: "success",
        output: s.tool_output,
      }),
    };
    turns.push({
      model_input: context.map((message) => ({ ...message })),
      context_added: [assistant, observation].filter(Boolean),
      parsed: {
        think: s.think,
        tool: s.tool_call?.tool,
        arguments: s.tool_call?.arguments,
      },
      tool_output: s.tool_output,
      perception: !!s.perception,
      error_attempt: !!s.error_attempt,
    });
    context.push(assistant);
    if (observation) context.push(observation);
  }
  return {
    question: record.question,
    db_id: record.db_id || record.source?.db_id,
    gold_sql: record.source?.gold_sql,
    db_overview: record.initial_state?.dataset_overview,
    isTraining: true,
    correct: extra.correct ?? true,
    turns,
  };
}

function RejectedCandidates({ record }) {
  const candidates = record?.enrichment?.rejected_candidates || [];
  if (!candidates.length) return null;
  return (
    <details className="rejected-candidates">
      <summary>被拒候选轨迹 · {candidates.length}</summary>
      {candidates.map((item, index) => (
        <div className="rejected-candidate" key={`${item.attempt}-${item.stage}-${index}`}>
          <div className="generator-attempt-head">
            <strong>Attempt {item.attempt ?? index + 1}</strong>
            {item.stage ? <span className="traj-tag err">{item.stage}</span> : null}
            {item.issues ? <span className="text-warn">{item.issues}</span> : null}
          </div>
          <div className="rejected-checks">
            {["l1_ok", "l2_ok", "q_ok", "ok"].map((key) => (
              item[key] == null ? null : (
                <span key={key} className={item[key] ? "text-ok" : "text-warn"}>
                  {key}:{String(item[key])}
                </span>
              )
            ))}
          </div>
          {item.candidate_steps?.length ? (
            <TrajectoryView record={stepsToTrajectory(record, item.candidate_steps, { correct: false })} />
          ) : (
            <p className="traj-empty">该失败发生在形成可 replay 轨迹之前；请查看上方外部模型原始输出。</p>
          )}
        </div>
      ))}
    </details>
  );
}

// Enriched record {steps:[{step_id,think,tool_call,tool_output,perception?,error_attempt?}]} -> the
// shape TrajectoryView renders, carrying per-step perception/error flags for highlighting.
export function enrichedToTrajectory(record) {
  return stepsToTrajectory(record, record.steps || [], { correct: true });
}

export function ConstructionPanel() {
  const [files, setFiles] = useState([]);
  const [file, setFile] = useState("");
  const [data, setData] = useState(null);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/construction")
      .then((r) => r.json())
      .then((d) => {
        const fs = (d.files || []).map(normalizeConstructionFile).filter((item) => item.name);
        setFiles(fs);
        setFile((cur) => cur || (fs.length ? fs[0].name : ""));
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!file) return;
    setData(null);
    setError("");
    fetch(`/api/construction?file=${encodeURIComponent(file)}&page=${page}&page_size=8`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) throw new Error(d.error);
        setData(d);
        setSelected(0);
      })
      .catch((e) => setError(e.message));
  }, [file, page]);

  const active = data?.records?.[selected]?.record;
  const modeName = active ? enrichmentField(active, "mode", "skeleton") : null;
  const mode = active ? MODE_META[modeName] || MODE_META.skeleton : null;

  return (
    <main className="content">
      <header className="page-header">
        <div>
          <p className="eyebrow">DATA CONSTRUCTION</p>
          <h1>数据构造审核</h1>
          <p>反思/感知富化轨迹:开场目录 → describe_table 取列 → inspect_column 确认字面量 → 纠错(错误→观察→改正) → 执行</p>
        </div>
        <div className="construction-picker">
          <span className="construction-picker-title">
            <FileJson size={14} />
            生成数据
          </span>
          <div className="construction-file-list" role="listbox" aria-label="构造数据文件">
            {files.map((item) => (
              <button
                key={item.name}
                type="button"
                className={`construction-file${file === item.name ? " active" : ""}`}
                title={item.name}
                onClick={() => {
                  setFile(item.name);
                  setPage(1);
                }}
              >
                <span className="construction-file-name">{item.name}</span>
                <span className="construction-file-meta">
                  <Clock size={12} />
                  {formatFileTime(item)} · {formatBytes(item.size_bytes)}
                </span>
              </button>
            ))}
          </div>
        </div>
      </header>
      <section className="panel browser-panel">
        {error ? <p className="traj-empty">读取失败:{error}</p> : null}
        {!data && !error ? <div className="loading-line" /> : null}
        {data ? (
          <>
            <div className="record-layout">
              <div className="record-index">
                {data.records.map((item, index) => {
                  const r = item.record;
                  const m = MODE_META[enrichmentField(r, "mode", "skeleton")] || MODE_META.skeleton;
                  return (
                    <button
                      key={item.index}
                      className={selected === index ? "active" : ""}
                      onClick={() => setSelected(index)}
                    >
                      <span>#{item.index + 1}</span>
                      <strong>{r.question || r.trajectory_id}</strong>
                      <em className="case-bucket" style={{ color: m.color }}>
                        {m.label} · 感{enrichmentField(r, "n_perception", 0)}/纠{enrichmentField(r, "n_error", 0)}
                      </em>
                    </button>
                  );
                })}
              </div>
              <div className="record-json light">
                <div className="json-toolbar">
                  <span>
                    {active ? (
                      <>
                        <span className="bucket-badge" style={{ "--badge": mode.color }}>{mode.label}</span>
                        {" "}感知 {enrichmentField(active, "n_perception", 0)} · 纠错 {enrichmentField(active, "n_error", 0)}
                        {active.enrichment?.generator?.model ? ` · ${active.enrichment.generator.model}` : ""}
                      </>
                    ) : "—"}
                  </span>
                  <span>{data.total} 条</span>
                </div>
                {active ? (
                  <>
                    <GeneratorTrace record={active} />
                    <RejectedCandidates record={active} />
                    <TrajectoryView record={enrichedToTrajectory(active)} />
                  </>
                ) : null}
              </div>
            </div>
            <div className="pagination">
              <button className="icon-button" disabled={page <= 1} onClick={() => setPage((v) => v - 1)}>
                <ChevronLeft size={18} />
              </button>
              <span>第 {data.page} / {data.pages} 页</span>
              <button className="icon-button" disabled={page >= data.pages} onClick={() => setPage((v) => v + 1)}>
                <ChevronRight size={18} />
              </button>
            </div>
          </>
        ) : null}
      </section>
    </main>
  );
}
