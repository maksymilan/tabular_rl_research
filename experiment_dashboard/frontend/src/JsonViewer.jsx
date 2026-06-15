import { useState } from "react";
import {
  Bot,
  Braces,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Database,
  MessageSquare,
  User,
  X,
} from "lucide-react";

const isPrimitive = (value) =>
  value === null || ["string", "number", "boolean"].includes(typeof value);

const primitiveText = (value) => {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
};

function Primitive({ value }) {
  return <span className={`json-value type-${value === null ? "null" : typeof value}`}>{primitiveText(value)}</span>;
}

function ObjectTable({ rows }) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))].slice(0, 10);
  return (
    <div className="object-table-wrap">
      <table className="object-table">
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>
                  {isPrimitive(row[column]) ? (
                    <Primitive value={row[column]} />
                  ) : (
                    <span className="json-complex">{Array.isArray(row[column]) ? `Array(${row[column].length})` : "Object"}</span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TreeNode({ name, value, depth = 0, defaultOpen = depth < 2 }) {
  const [open, setOpen] = useState(defaultOpen);
  if (isPrimitive(value)) {
    return (
      <div className="tree-row leaf">
        {name !== undefined ? <span className="json-key">{name}</span> : null}
        <Primitive value={value} />
      </div>
    );
  }

  const entries = Array.isArray(value) ? value.map((item, index) => [index, item]) : Object.entries(value);
  const label = Array.isArray(value) ? `Array(${entries.length})` : `Object(${entries.length})`;
  const tabular =
    Array.isArray(value) &&
    value.length > 0 &&
    value.length <= 100 &&
    value.every((item) => item && typeof item === "object" && !Array.isArray(item));

  return (
    <div className="tree-node">
      <button className="tree-toggle" type="button" onClick={() => setOpen((current) => !current)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {name !== undefined ? <span className="json-key">{name}</span> : null}
        <span className="json-shape">{label}</span>
      </button>
      {open ? (
        <div className="tree-children">
          {tabular ? (
            <ObjectTable rows={value} />
          ) : (
            entries.map(([key, child]) => (
              <TreeNode key={key} name={key} value={child} depth={depth + 1} />
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}

export function JsonTree({ value }) {
  return (
    <div className="json-tree">
      <TreeNode value={value} />
    </div>
  );
}

function maybeJson(content) {
  if (typeof content !== "string") return null;
  const trimmed = content.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function roleMeta(role) {
  if (role === "assistant" || role === "gpt") return { label: "Assistant", icon: Bot, tone: "assistant" };
  if (role === "system") return { label: "System", icon: Code2, tone: "system" };
  if (role === "tool" || role === "observation") return { label: "Observation", icon: Database, tone: "tool" };
  return { label: "User", icon: User, tone: "user" };
}

export function ConversationView({ messages, system }) {
  const normalized = [
    ...(system ? [{ role: "system", content: system }] : []),
    ...(messages || []).map((message) => ({
      role: message.role || message.from || "user",
      content: message.content ?? message.value ?? message,
    })),
  ];
  return (
    <div className="conversation-view">
      {normalized.map((message, index) => {
        const meta = roleMeta(message.role);
        const Icon = meta.icon;
        const parsed = maybeJson(message.content);
        return (
          <article className={`message-card ${meta.tone}`} key={index}>
            <header>
              <span><Icon size={14} /> {meta.label}</span>
              <small>#{index + 1}</small>
            </header>
            {parsed ? <JsonTree value={parsed} /> : <div className="message-content">{String(message.content)}</div>}
          </article>
        );
      })}
    </div>
  );
}

function ResultTable({ title, rows, tone }) {
  const values = (rows || []).map((row) => (Array.isArray(row) ? row : [row]));
  return (
    <div className={`result-table ${tone}`}>
      <header>
        <span>{title}</span>
        <strong>{values.length} rows</strong>
      </header>
      {values.length ? (
        <div className="object-table-wrap">
          <table className="object-table">
            <tbody>
              {values.slice(0, 20).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => <td key={cellIndex}><Primitive value={cell} /></td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div className="empty-result">空结果</div>}
    </div>
  );
}

export function SqlEvaluationView({ record }) {
  return (
    <div className="sql-eval-view">
      <div className="record-verdict">
        <span className={record.correct ? "inline-ok" : "inline-bad"}>
          {record.correct ? <Check size={15} /> : <X size={15} />}
          {record.correct ? "执行正确" : record.failure_type || "执行失败"}
        </span>
        <span>{record.db_id}</span>
        <span>{record.elapsed_seconds}s</span>
      </div>
      <h3>{record.question}</h3>
      <div className="sql-compare">
        <div>
          <span>模型 SQL</span>
          <pre>{record.predicted_sql || record.model_output || "—"}</pre>
        </div>
        <div>
          <span>Gold SQL</span>
          <pre>{record.gold_sql || "—"}</pre>
        </div>
      </div>
      <div className="result-compare">
        <ResultTable title="预测结果" rows={record.predicted_sample || record.pred_sample} tone="pred" />
        <ResultTable title="Gold 结果" rows={record.gold_sample} tone="gold" />
      </div>
      {record.model_input ? (
        <details className="embedded-section">
          <summary><MessageSquare size={14} /> 查看完整模型输入</summary>
          <ConversationView messages={record.model_input} />
        </details>
      ) : null}
    </div>
  );
}

export function StructuredRecord({ record }) {
  if (Array.isArray(record?.conversations)) {
    return <ConversationView system={record.system} messages={record.conversations} />;
  }
  if (Array.isArray(record?.model_input) && ("predicted_sql" in record || "model_output" in record)) {
    return <SqlEvaluationView record={record} />;
  }
  if (Array.isArray(record?.initial_model_input) && !Array.isArray(record?.turns)) {
    return <ConversationView messages={record.initial_model_input} />;
  }
  return (
    <div className="generic-record">
      <div className="generic-record-title"><Braces size={15} /> 结构化对象</div>
      <JsonTree value={record} />
    </div>
  );
}
