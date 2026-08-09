#!/usr/bin/env python3
"""Build vector assets for the checkpoint-relalg OmniGraffle visual system.

The generated SVG files are imported into OmniGraffle and saved as editable
`.graffle` source documents. Equations are rendered by LaTeX and embedded as
vector SVG data; no equation is approximated with an ordinary text box.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
EQUATION_DIR = HERE / "equations"
STANDARD_VERSION = "checkpoint-relalg-visual-standard-v1"
PROTOCOL_VERSION = "checkpoint-relalg-v1"

COLORS = {
    "canvas": "#F7F8FC",
    "white": "#FFFFFF",
    "ink": "#172033",
    "muted": "#667085",
    "border": "#CBD5E1",
    "provider": "#6658D3",
    "protocol": "#246BCE",
    "state": "#138A7E",
    "execution": "#D97706",
    "checkpoint": "#8B5CF6",
    "success": "#26845B",
    "error": "#C2414B",
    "hidden": "#475467",
    "warning": "#B7791F",
    "light_blue": "#EAF2FF",
    "light_teal": "#E7F7F4",
    "light_violet": "#F1EDFF",
    "light_amber": "#FFF5DE",
    "light_red": "#FDECEC",
    "light_gray": "#EEF2F6",
}


FORMULAS = {
    "F01": r"M_t=\operatorname{Render}\!\left(Q,K,T_t,C_{0:k},S_t,E_t\right)",
    "F02": r"a_t\sim\pi_\theta(\,\cdot\mid M_t),\qquad\left|\operatorname{AuthoredActions}_t\right|=1",
    "F03": r"\mathcal{C}_{\mathrm{task}}=Q\oplus K,\qquad\mathcal{F}_{\mathrm{db}}=S_t,\qquad C_{0:k}\not\vdash\text{database facts}",
    "F04": r"\mathcal{T}_{D}=\mathcal{P}\cup\{\operatorname{execute\_sql}\}\cup\mathcal{C},\quad\mathcal{T}_{A}=\mathcal{P}\cup\mathcal{R}\cup\mathcal{C},\quad\mathcal{T}_{H}=\mathcal{T}_{D}\cup\mathcal{T}_{A}",
    "F05": r"A=\left\langle h,k,\left\langle c_i:\tau_i\right\rangle_{i=1}^{m},n,\omega,d\right\rangle",
    "F06": r"\sigma_p(R),\quad\pi_{\langle e_i\to a_i\rangle_{i=1}^{m}}(R),\quad R\bowtie_\theta S,\quad\gamma_{G;\langle f_i(c_i)\to a_i\rangle}(R),\quad\tau_K(R)",
    "F07": r"\mu_{R\bowtie_\theta S}(r,s)=\mu_R(r)\,\mu_S(s)\,\mathbf{1}[\theta(r,s)]",
    "F08": r"C_k=\langle D_k,A_k,O_k,U_k,h_k\rangle,\qquad h_k=\operatorname{SHA256}\!\left(\operatorname{canon}(S_k)\right)",
    "F09": r"\operatorname{Restore}(C_j):\quad S\leftarrow S(C_j),\qquad\mathcal{P}_{\mathrm{active}}\leftarrow\operatorname{Ancestors}(C_j)\cup\{C_{\mathrm{recovery}}\}",
    "F10": r"S_{t+1}=\begin{cases}\operatorname{Apply}(S_t,a_t),&\operatorname{valid}(a_t),\\S_t,&\operatorname{rejected}(a_t)\lor\operatorname{failed}(a_t),\end{cases}\qquad H(S_{t+1})=H(S_t)\ \text{on failure}",
    "F11": r"\widehat{R}=\operatorname{Artifact}(a_T),\qquad\operatorname{Acc}_{\mathrm{official}}=\mathbf{1}\!\left[\operatorname{Set}(\widehat{R})=\operatorname{Set}(R^*)\right]",
    "F12": r"\operatorname{Strict}(\widehat{R},R^*)=\operatorname{Schema}\land\operatorname{Rows}_{\mathrm{multiset/sequence}}\land\operatorname{NULL}\land\operatorname{Multiplicity}",
}


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )


def render_equations() -> dict[str, str]:
    """Render every canonical formula to path-only SVG and return data URIs."""

    EQUATION_DIR.mkdir(parents=True, exist_ok=True)
    data_uris: dict[str, str] = {}
    for formula_id, source in FORMULAS.items():
        with tempfile.TemporaryDirectory(prefix=f"{formula_id.lower()}-", dir=HERE) as tmp:
            temp_dir = Path(tmp)
            tex = temp_dir / f"{formula_id}.tex"
            tex.write_text(
                "\n".join(
                    [
                        r"\documentclass[12pt]{standalone}",
                        r"\usepackage{mathptmx}",
                        r"\usepackage{amsmath,amssymb}",
                        r"\begin{document}",
                        rf"$\displaystyle {source}$",
                        r"\end{document}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            _run(["latex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=temp_dir)
            output = EQUATION_DIR / f"{formula_id}.svg"
            _run(
                [
                    "dvisvgm",
                    "--no-fonts",
                    "--exact-bbox",
                    f"--output={output}",
                    f"{formula_id}.dvi",
                ],
                cwd=temp_dir,
            )
        encoded = base64.b64encode(output.read_bytes()).decode("ascii")
        data_uris[formula_id] = f"data:image/svg+xml;base64,{encoded}"
    return data_uris


def render_formula_sheet() -> Path:
    """Build a vector PDF companion from the canonical LaTeX source."""

    source = HERE / "checkpoint_relalg_formulas.tex"
    output = HERE / "checkpoint_relalg_formulas.pdf"
    with tempfile.TemporaryDirectory(prefix="formula-sheet-", dir=HERE) as tmp:
        temp_dir = Path(tmp)
        _run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={temp_dir}",
                str(source),
            ],
            cwd=HERE,
        )
        shutil.copy2(temp_dir / "checkpoint_relalg_formulas.pdf", output)
    return output


class Canvas:
    def __init__(self, title: str, formulas: dict[str, str]) -> None:
        self.title = title
        self.formulas = formulas
        self.parts: list[str] = []

    @staticmethod
    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "#FFFFFF",
        stroke: str = "#CBD5E1",
        sw: float = 1.5,
        rx: float = 12,
        dash: str | None = None,
        opacity: float = 1.0,
        ident: str | None = None,
    ) -> None:
        attrs = [
            f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
            f'rx="{rx}"', f'fill="{fill}"', f'stroke="{stroke}"',
            f'stroke-width="{sw}"', f'opacity="{opacity}"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if ident:
            attrs.append(f'id="{self.esc(ident)}"')
        self.add(f"<rect {' '.join(attrs)}/>")

    def line(
        self,
        points: Iterable[tuple[float, float]],
        *,
        color: str,
        sw: float = 2.25,
        dash: str | None = None,
        marker: str = "arrow",
    ) -> None:
        pts = " ".join(f"{x},{y}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        self.add(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"'
            f'{dash_attr}{marker_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        cls: str = "body",
        fill: str = COLORS["ink"],
        anchor: str = "start",
        weight: str | None = None,
        style: str = "",
    ) -> None:
        weight_attr = f' font-weight="{weight}"' if weight else ""
        self.add(
            f'<text x="{x}" y="{y}" class="{cls}" fill="{fill}" '
            f'text-anchor="{anchor}"{weight_attr} style="{self.esc(style)}">'
            f'{self.esc(value)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: Iterable[str],
        *,
        cls: str = "body",
        fill: str = COLORS["ink"],
        step: float = 20,
        anchor: str = "start",
    ) -> None:
        for index, line in enumerate(lines):
            self.text(x, y + step * index, line, cls=cls, fill=fill, anchor=anchor)

    def title_block(self, title: str, subtitle: str, *, status: str = "DIAGNOSTIC ONLY") -> None:
        self.text(64, 58, title, cls="title", weight="bold")
        self.text(64, 88, subtitle, cls="subtitle", fill=COLORS["muted"])
        self.chip(1640, 42, 216, 34, status, COLORS["warning"], "#FFF7E3")
        self.line([(64, 112), (1856, 112)], color=COLORS["border"], sw=1.25, marker="")

    def group(self, x: float, y: float, w: float, h: float, title: str, color: str, *, dash: str | None = None) -> None:
        self.rect(x, y, w, h, fill=COLORS["white"], stroke=color, sw=1.8, rx=16, dash=dash)
        self.rect(x, y, w, 38, fill=color, stroke=color, sw=0, rx=16, opacity=0.96)
        self.rect(x, y + 22, w, 16, fill=color, stroke=color, sw=0, rx=0, opacity=0.96)
        self.text(x + 18, y + 26, title, cls="section", fill=COLORS["white"], weight="bold")

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: Iterable[str],
        *,
        color: str,
        fill: str,
        icon: str | None = None,
        dash: str | None = None,
    ) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=color, sw=1.5, rx=12, dash=dash)
        tx = x + 18
        if icon:
            self.icon(icon, x + 18, y + 15, 24, color)
            tx = x + 52
        self.text(tx, y + 31, title, cls="node", fill=color, weight="bold")
        self.multiline(x + 18, y + 55, lines, cls="body", fill=COLORS["ink"], step=18)

    def chip(self, x: float, y: float, w: float, h: float, value: str, color: str, fill: str) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=color, sw=1.2, rx=h / 2)
        self.text(x + w / 2, y + h / 2 + 4, value, cls="chip", fill=color, anchor="middle", weight="bold")

    def formula(self, formula_id: str, x: float, y: float, w: float, h: float, *, caption: str | None = None) -> None:
        self.rect(x, y, w, h, fill="#FFFFFF", stroke=COLORS["border"], sw=1.2, rx=10)
        if caption:
            self.text(x + 12, y + 20, f"{formula_id} · {caption}", cls="foot", fill=COLORS["muted"], weight="bold")
            image_y = y + 25
            image_h = h - 34
        else:
            image_y = y + 8
            image_h = h - 16
        self.add(
            f'<image x="{x + 12}" y="{image_y}" width="{w - 24}" height="{image_h}" '
            f'preserveAspectRatio="xMidYMid meet" href="{self.formulas[formula_id]}" '
            f'data-latex="{self.esc(FORMULAS[formula_id])}"/>'
        )

    def icon(self, name: str, x: float, y: float, size: float, color: str) -> None:
        self.add(
            f'<use href="#icon-{self.esc(name)}" x="{x}" y="{y}" width="{size}" height="{size}" '
            f'style="color:{color}"/>'
        )

    def number(self, x: float, y: float, value: int, color: str = COLORS["protocol"]) -> None:
        self.add(f'<circle cx="{x}" cy="{y}" r="15" fill="{color}"/>')
        self.text(x, y + 5, str(value), cls="chip", fill="#FFFFFF", anchor="middle", weight="bold")

    def svg(self) -> str:
        symbols = f"""
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="context-stroke"/></marker>
  <marker id="stop" markerWidth="10" markerHeight="10" refX="7" refY="5" orient="auto"><path d="M6,0 L6,10" stroke="context-stroke" stroke-width="2.5"/></marker>
  <symbol id="icon-document" viewBox="0 0 24 24"><path d="M5 2h9l5 5v15H5z M14 2v6h5 M8 12h8 M8 16h8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></symbol>
  <symbol id="icon-stack" viewBox="0 0 24 24"><path d="M12 3 3 8l9 5 9-5z M3 12l9 5 9-5 M3 16l9 5 9-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></symbol>
  <symbol id="icon-model" viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="4" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="8" cy="12" r="1.4" fill="currentColor"/><circle cx="16" cy="12" r="1.4" fill="currentColor"/><path d="M8 16h8 M12 2v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
  <symbol id="icon-shield" viewBox="0 0 24 24"><path d="M12 2 20 5v6c0 5-3.4 9-8 11-4.6-2-8-6-8-11V5z M8 12l2.5 2.5L16.5 8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></symbol>
  <symbol id="icon-database" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5 M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" fill="none" stroke="currentColor" stroke-width="1.8"/></symbol>
  <symbol id="icon-table" viewBox="0 0 24 24"><rect x="2.5" y="4" width="19" height="16" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M2.5 9h19 M9 4v16 M15 4v16" fill="none" stroke="currentColor" stroke-width="1.5"/></symbol>
  <symbol id="icon-eye" viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="12" cy="12" r="2.8" fill="none" stroke="currentColor" stroke-width="1.8"/></symbol>
  <symbol id="icon-checkpoint" viewBox="0 0 24 24"><path d="M5 22V3 M5 4h12l-2 4 2 4H5 M9 16h10v5H9z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></symbol>
  <symbol id="icon-terminal" viewBox="0 0 24 24"><rect x="2.5" y="4" width="19" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="m6 9 3 3-3 3 M11 16h6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="icon-audit" viewBox="0 0 24 24"><circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="m14.5 14.5 6 6 M7 10l2 2 4-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
  <symbol id="icon-lock" viewBox="0 0 24 24"><rect x="4" y="10" width="16" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M8 10V7a4 4 0 0 1 8 0v3 M12 14v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
  <symbol id="icon-error" viewBox="0 0 24 24"><path d="M12 2 22 20H2z" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 8v5 M12 17h.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></symbol>
</defs>
"""
        css = """
<style>
text { font-family: 'Times New Roman', Times, serif; }
.title { font-size: 34px; }
.subtitle { font-size: 16px; }
.section { font-size: 18px; }
.node { font-size: 15px; }
.body { font-size: 12.5px; }
.edge { font-size: 11px; font-style: italic; }
.chip { font-size: 10.5px; }
.foot { font-size: 10px; }
</style>
"""
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="1920" height="1080" viewBox="0 0 1920 1080">\n'
            f'<title>{self.esc(self.title)}</title>\n{symbols}{css}'
            f'<rect width="1920" height="1080" fill="{COLORS["canvas"]}"/>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


def architecture(formulas: dict[str, str]) -> Canvas:
    c = Canvas("Checkpointed Relational Agent — System Architecture", formulas)
    c.title_block(
        "Checkpointed Relational Agent — System Architecture",
        "checkpoint-relalg-v1 · Direct / Atomic / Hybrid · authority, state, execution, and no-leak boundaries",
    )
    c.chip(1230, 72, 190, 28, "CURRENT: HYBRID", COLORS["protocol"], COLORS["light_blue"])
    c.chip(1430, 72, 205, 28, "ARM A: TEXT-JSON", COLORS["provider"], COLORS["light_violet"])

    c.group(48, 138, 286, 316, "TASK CONTRACT", COLORS["ink"])
    c.card(70, 196, 242, 92, "QUESTION", ["Requested semantics", "Output grain and shape"], color=COLORS["ink"], fill="#F4F6F8", icon="document")
    c.card(70, 306, 242, 92, "EXTERNAL KNOWLEDGE", ["Task-given mappings", "Constants and definitions"], color=COLORS["ink"], fill="#F4F6F8", icon="document")
    c.text(70, 428, "Task authority only", cls="chip", fill=COLORS["ink"], weight="bold")

    c.group(356, 138, 360, 316, "PROMPT ASSEMBLY", COLORS["protocol"])
    c.card(378, 190, 316, 104, "Static contract", ["Shared core · Mode contract", "Teacher checkpoint guidance", "Exact mode tool schemas"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="stack")
    c.card(378, 310, 316, 124, "Dynamic context renderer", ["QUESTION → EXTERNAL KNOWLEDGE", "PHASE TARGETS → CHECKPOINT HISTORY", "ENVIRONMENT STATE → LAST ERROR"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="document")

    c.group(738, 138, 338, 316, "MODEL / PROVIDER", COLORS["provider"])
    c.card(760, 190, 294, 88, "Official DeepSeek", ["Chat Completions · thinking high", "Provider-private reasoning: audit only"], color=COLORS["provider"], fill=COLORS["light_violet"], icon="model")
    c.card(760, 294, 142, 112, "Arm A", ["Text-JSON", "one raw action", "current rollout"], color=COLORS["provider"], fill="#F7F3FF")
    c.card(912, 294, 142, 112, "Arm B", ["Native tool_call", "one call", "ablation"], color=COLORS["muted"], fill="#F4F6F8", dash="5 4")
    c.text(907, 429, "Exactly one authored action", cls="chip", fill=COLORS["provider"], anchor="middle", weight="bold")

    c.group(1098, 138, 402, 316, "HARNESS CONTROL PLANE", COLORS["protocol"])
    c.card(1120, 188, 166, 102, "Carrier parser", ["Strict envelope", "No semantic repair"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="shield")
    c.card(1302, 188, 176, 102, "Schema validator", ["Closed arguments", "Mode capability"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="shield")
    c.card(1120, 306, 166, 102, "State validator", ["Active handles", "Exact columns"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="shield")
    c.card(1302, 306, 176, 102, "Runtime dispatcher", ["Perception · SQL", "Atomic · Control"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="terminal")

    c.group(1522, 138, 350, 316, "STATE / EXECUTION", COLORS["state"])
    c.card(1544, 188, 306, 94, "EnvironmentState", ["Active membership + immutable records", "Current phase · checkpoint · targets"], color=COLORS["state"], fill=COLORS["light_teal"], icon="stack")
    c.card(1544, 298, 144, 112, "CheckpointStore", ["Snapshots", "Active path", "Restore branch"], color=COLORS["checkpoint"], fill=COLORS["light_violet"], icon="checkpoint")
    c.card(1704, 298, 146, 112, "Artifact registry", ["Immutable tables", "Schema · order", "Derivation"], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")

    c.line([(334, 256), (356, 256)], color=COLORS["protocol"])
    c.line([(716, 256), (738, 256)], color=COLORS["protocol"])
    c.line([(1076, 256), (1098, 256)], color=COLORS["protocol"])
    c.line([(1500, 256), (1522, 256)], color=COLORS["state"])

    c.group(48, 486, 430, 344, "AUTHORITY AND EVIDENCE", COLORS["ink"])
    c.card(70, 542, 386, 70, "Task authority", ["QUESTION + EXTERNAL KNOWLEDGE"], color=COLORS["ink"], fill="#F4F6F8", icon="document")
    c.card(70, 626, 386, 70, "Database authority", ["CURRENT ENVIRONMENT STATE + artifacts"], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")
    c.card(70, 710, 386, 92, "Non-evidence", ["Checkpoint prose = working memory", "Reasoning and assistant prose = audit only"], color=COLORS["warning"], fill=COLORS["light_amber"], icon="lock")

    c.group(500, 486, 612, 344, "UNIFIED EXECUTION AND PUBLICATION", COLORS["execution"])
    c.card(522, 542, 178, 106, "Perception", ["describe_table", "inspect_column", "read_rows"], color=COLORS["state"], fill=COLORS["light_teal"], icon="eye")
    c.card(718, 542, 178, 106, "Direct SQL", ["execute_sql", "read-only SELECT / WITH", "authorizer + timeout"], color=COLORS["execution"], fill=COLORS["light_amber"], icon="terminal")
    c.card(914, 542, 176, 106, "Atomic algebra", ["9 typed operators", "canonical types", "bag semantics"], color=COLORS["execution"], fill=COLORS["light_amber"], icon="table")
    c.card(522, 674, 270, 120, "Source SQLite database", ["Read-only · registered relations", "BINARY text collation", "row / byte / cell / time limits"], color=COLORS["execution"], fill=COLORS["light_amber"], icon="database")
    c.card(812, 674, 278, 120, "Atomic publish", ["Validate → execute → materialize", "Success: publish artifact + step", "Failure: rollback; no artifact"], color=COLORS["success"], fill="#EAF8F0", icon="shield")
    c.line([(611, 648), (611, 674)], color=COLORS["execution"])
    c.line([(807, 648), (807, 660), (670, 660), (670, 674)], color=COLORS["execution"])
    c.line([(1002, 648), (1002, 674)], color=COLORS["execution"])

    c.group(1134, 486, 738, 344, "ISOLATED SCORING AND AUDIT", COLORS["hidden"], dash="8 5")
    c.text(1158, 526, "NO-LEAK BOUNDARY — no evaluator edge returns to the model", cls="chip", fill=COLORS["error"], weight="bold")
    c.card(1158, 550, 208, 104, "Terminal artifact", ["answer(handle)", "Exact full relation", "Model does not copy values"], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")
    c.card(1386, 550, 216, 104, "Hidden evaluator", ["Reference executes only", "after termination", "Never provider-visible"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="lock")
    c.card(1622, 550, 226, 104, "Two evaluation views", ["Official bird-set", "Strict artifact audit", "schema · order · NULL · bags"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="audit")
    c.card(1158, 682, 318, 110, "Immutable run artifact", ["Provider history · state hashes", "Checkpoint path · dependencies", "Errors · budgets · identity"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="document")
    c.card(1498, 682, 350, 110, "Fresh replay and no-leak", ["Re-render every causal input", "Re-execute model-authored actions", "Diagnostic trajectory only"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="audit")
    c.line([(1366, 602), (1386, 602)], color=COLORS["state"])
    c.line([(1602, 602), (1622, 602)], color=COLORS["hidden"])
    c.line([(1735, 654), (1735, 670), (1673, 670), (1673, 682)], color=COLORS["hidden"])
    c.line([(1476, 737), (1498, 737)], color=COLORS["hidden"])

    c.line([(1697, 454), (1697, 470), (812, 470), (812, 486)], color=COLORS["state"])
    c.line([(1697, 486), (1697, 464), (536, 464), (536, 434)], color=COLORS["state"], dash="6 5")
    c.text(1110, 463, "Harness re-renders the causal state", cls="edge", fill=COLORS["state"], anchor="middle")
    c.text(972, 842, "Bounded client retry = transport event, not a semantic action", cls="foot", fill=COLORS["warning"], anchor="middle")

    c.formula("F01", 64, 866, 850, 136, caption="Causal context")
    c.formula("F02", 936, 866, 920, 136, caption="Single-action invariant")
    c.text(64, 1042, "Source of truth: executable registry + frozen implementation specification. Current status remains diagnostic-only.", cls="foot", fill=COLORS["muted"])
    return c


def tool_surface(formulas: dict[str, str]) -> Canvas:
    c = Canvas("Unified Tool Surface and Extended Bag Relational Algebra", formulas)
    c.title_block(
        "Unified Tool Surface and Extended Bag Relational Algebra",
        "One immutable artifact system across Direct, Atomic, and Hybrid modes",
    )
    c.chip(64, 132, 260, 34, "DIRECT = P + SQL + C", COLORS["execution"], COLORS["light_amber"])
    c.chip(340, 132, 280, 34, "ATOMIC = P + R + C", COLORS["state"], COLORS["light_teal"])
    c.chip(636, 132, 332, 34, "HYBRID = DIRECT ∪ ATOMIC", COLORS["protocol"], COLORS["light_blue"])
    c.text(994, 154, "Current experiment mode · SQL and atomic artifacts interoperate", cls="chip", fill=COLORS["protocol"], weight="bold")

    c.group(48, 190, 300, 426, "SHARED PERCEPTION · P", COLORS["state"])
    for idx, (name, detail) in enumerate([
        ("describe_table", "Discover source schema"),
        ("inspect_column", "Bounded value profile"),
        ("read_rows", "Observation only; no artifact"),
    ]):
        c.card(70, 246 + idx * 104, 256, 82, name, [detail], color=COLORS["state"], fill=COLORS["light_teal"], icon="eye")
    c.text(70, 586, "Outputs: schemas and bounded observations", cls="foot", fill=COLORS["state"], weight="bold")

    c.group(370, 190, 276, 426, "DIRECT SQL", COLORS["execution"])
    c.card(392, 248, 232, 128, "execute_sql", ["One read-only SELECT / WITH", "Registered relation namespace", "Top-level ORDER BY is explicit"], color=COLORS["execution"], fill=COLORS["light_amber"], icon="terminal")
    c.card(392, 398, 232, 126, "SQL safety envelope", ["Authorizer · timeout", "canonical dynamic types", "row / byte / cell limits"], color=COLORS["execution"], fill=COLORS["light_amber"], icon="shield")
    c.text(508, 570, "Produces relation artifact", cls="foot", fill=COLORS["execution"], anchor="middle", weight="bold")

    c.group(668, 190, 700, 426, "ATOMIC OPERATORS · R", COLORS["execution"])
    atomic = [
        ("filter_rows", "rows only · preserve order"), ("project", "columns / expressions"),
        ("join", "binary · bag multiplicity"), ("aggregate", "group and metrics"),
        ("distinct", "all visible columns"), ("set_operation", "aligned schemas"),
        ("sort", "establish order"), ("limit", "ordered input required"),
        ("add_rank", "add ranking column"),
    ]
    for idx, (name, detail) in enumerate(atomic):
        col = idx % 3
        row = idx // 3
        c.card(690 + col * 220, 244 + row * 112, 202, 92, name, [detail], color=COLORS["execution"], fill=COLORS["light_amber"], icon="table")

    c.group(1390, 190, 482, 426, "SHARED CONTROL · C", COLORS["checkpoint"])
    c.card(1412, 244, 212, 112, "commit_checkpoint", ["Meaningful milestone", "summary · uncertainty", "1–3 next targets"], color=COLORS["checkpoint"], fill=COLORS["light_violet"], icon="checkpoint")
    c.card(1638, 244, 212, 112, "restore_checkpoint", ["Explicit contradiction", "restore exact snapshot", "create recovery child"], color=COLORS["checkpoint"], fill=COLORS["light_violet"], icon="checkpoint")
    c.card(1412, 382, 438, 112, "answer", ["Consume exactly one active relation artifact", "The exact full relation is the answer evidence"], color=COLORS["success"], fill="#EAF8F0", icon="table")
    c.text(1631, 548, "Zero checkpoints are legal for simple tasks", cls="foot", fill=COLORS["checkpoint"], anchor="middle")

    c.line([(348, 408), (370, 408)], color=COLORS["state"], dash="6 5")
    c.line([(646, 408), (668, 408)], color=COLORS["execution"])
    c.line([(1368, 408), (1390, 408)], color=COLORS["state"])

    c.card(446, 642, 1028, 132, "IMMUTABLE RELATION ARTIFACT", [
        "handle · kind · ordered columns and canonical types · row_count · ordered_by · Harness derivation",
        "Every producer publishes atomically. Direct and atomic tools can consume the same active artifact.",
        "Hidden ordering metadata never enters visible columns, Direct SQL, context, or answer.",
    ], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")
    c.line([(508, 616), (508, 642)], color=COLORS["execution"])
    c.line([(1018, 616), (1018, 642)], color=COLORS["execution"])
    c.line([(1631, 494), (1631, 608), (1474, 608), (1474, 708)], color=COLORS["state"])
    c.line([(960, 642), (960, 620), (832, 620), (832, 616)], color=COLORS["state"], dash="5 4")

    c.formula("F04", 64, 808, 880, 104, caption="Mode surfaces")
    c.formula("F05", 966, 808, 890, 104, caption="Artifact contract")
    c.formula("F06", 64, 928, 880, 94, caption="Extended relational algebra")
    c.formula("F07", 966, 928, 890, 94, caption="Bag join multiplicity")
    c.text(64, 1050, "Ordering badge semantics: preserve · establish · clear. BLOB values are equality-only and never valid order keys.", cls="foot", fill=COLORS["muted"])
    return c


def checkpoint_reasoning(formulas: dict[str, str]) -> Canvas:
    c = Canvas("Phase Reasoning, Semantic Checkpoints, and Recovery", formulas)
    c.title_block(
        "Phase Reasoning, Semantic Checkpoints, and Recovery",
        "Checkpoints control active working memory; they do not create database evidence",
    )
    c.text(64, 148, "PHASE TIMELINE", cls="section", fill=COLORS["checkpoint"], weight="bold")
    phases = [
        (100, "P₀ · Bootstrap", "explore or answer directly", COLORS["protocol"]),
        (410, "C₁ · Commit", "meaningful milestone", COLORS["checkpoint"]),
        (690, "P₁(T₁)", "targeted investigation", COLORS["protocol"]),
        (970, "C₂ · Commit", "later assumption", COLORS["checkpoint"]),
        (1250, "Contradiction?", "explicit evidence", COLORS["error"]),
        (1505, "Restore C₁", "new recovery child", COLORS["checkpoint"]),
        (1740, "Answer", "active artifact", COLORS["success"]),
    ]
    for idx, (x, title, detail, color) in enumerate(phases):
        w = 220 if idx not in {4, 5, 6} else (190 if idx == 4 else 180)
        c.card(x, 180, w, 96, title, [detail], color=color, fill="#FFFFFF", icon="checkpoint" if "C" in title or "Restore" in title else ("error" if idx == 4 else "table"))
        if idx < len(phases) - 1:
            nx = phases[idx + 1][0]
            c.line([(x + w, 228), (nx, 228)], color=COLORS["checkpoint"] if idx in {1, 3, 4} else COLORS["protocol"], dash="8 5" if idx in {1, 3, 4} else None)
    c.line([(100, 288), (100, 312), (1770, 312), (1770, 276)], color=COLORS["muted"], dash="3 5")
    c.text(900, 307, "Simple tasks may take the zero-checkpoint path P₀ → answer", cls="edge", fill=COLORS["muted"], anchor="middle")

    c.text(64, 366, "CHECKPOINT GRAPH", cls="section", fill=COLORS["checkpoint"], weight="bold")
    c.card(80, 402, 170, 82, "root", ["bootstrap snapshot"], color=COLORS["checkpoint"], fill=COLORS["light_violet"], icon="checkpoint")
    c.card(360, 402, 190, 82, "C₁ · active ancestor", ["stable milestone"], color=COLORS["checkpoint"], fill=COLORS["light_violet"], icon="checkpoint")
    c.card(680, 350, 210, 82, "C₂ · abandoned", ["invalidated branch"], color=COLORS["muted"], fill=COLORS["light_gray"], icon="checkpoint", dash="4 4")
    c.card(680, 482, 238, 82, "C₃ · recovery child", ["new active phase"], color=COLORS["success"], fill="#EAF8F0", icon="checkpoint")
    c.card(1048, 482, 228, 82, "Answer artifact R*", ["active membership only"], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")
    c.line([(250, 443), (360, 443)], color=COLORS["checkpoint"])
    c.line([(550, 443), (610, 443), (610, 391), (680, 391)], color=COLORS["muted"], dash="3 5")
    c.line([(550, 443), (610, 443), (610, 523), (680, 523)], color=COLORS["checkpoint"], dash="8 5")
    c.line([(918, 523), (1048, 523)], color=COLORS["success"])

    c.card(1340, 350, 520, 214, "Structural-sharing snapshot", [
        "discovered_schema_ids", "active_artifact_ids", "active_observation_ids",
        "usable_step_ids", "environment_state_hash",
        "Tables are not copied; immutable records are shared.",
    ], color=COLORS["state"], fill=COLORS["light_teal"], icon="stack")
    c.line([(890, 391), (1320, 391), (1320, 418), (1340, 418)], color=COLORS["state"], dash="5 4")
    c.line([(918, 523), (1320, 523), (1320, 500), (1340, 500)], color=COLORS["state"], dash="5 4")

    c.group(48, 612, 570, 226, "COMMIT CONTRACT", COLORS["checkpoint"])
    c.multiline(74, 676, [
        "progress_summary · 1–5 verified milestone statements",
        "remaining_uncertainties · 0–5 unresolved semantic questions",
        "next_targets · 1–3 objectives for the next phase",
        "Commit only after answer-relevant semantic progress.",
    ], step=28)
    c.group(642, 612, 570, 226, "RESTORE CONTRACT", COLORS["checkpoint"])
    c.multiline(668, 676, [
        "checkpoint_id · an available snapshot or root",
        "reason · contradiction + invalidated assumption + why this node",
        "next_targets · 1–3 recovery objectives",
        "Old records remain immutable; inactive handles cannot be referenced.",
    ], step=28)
    c.group(1236, 612, 636, 226, "PHASE TRANSCRIPT RESET", COLORS["provider"])
    c.multiline(1262, 676, [
        "After commit or restore: clear the old provider phase transcript.",
        "Retain compact CHECKPOINT HISTORY and current EnvironmentState.",
        "Carry only grounded producer identity across phases.",
        "Provider-private reasoning is archived for audit, not re-injected as fact.",
    ], step=28)

    c.formula("F08", 64, 864, 870, 128, caption="Checkpoint snapshot and logical hash")
    c.formula("F09", 956, 864, 900, 128, caption="Exact restore and active path")
    c.text(64, 1030, "Global identifiers are monotonic and never reused, including after restore.", cls="foot", fill=COLORS["muted"])
    return c


def causal_execution(formulas: dict[str, str]) -> Canvas:
    c = Canvas("Causal Episode Loop, Safety Invariants, and Admission Boundary", formulas)
    c.title_block(
        "Causal Episode Loop, Safety Invariants, and Admission Boundary",
        "A real model↔Harness trajectory with state-preserving failures and isolated evaluation",
    )
    c.text(64, 146, "CAUSAL LOOP", cls="section", fill=COLORS["protocol"], weight="bold")
    steps = [
        (82, 184, "Render causal prefix", "state + targets + last error", "document", COLORS["protocol"]),
        (370, 184, "Official provider request", "one carrier response", "model", COLORS["provider"]),
        (658, 184, "Envelope + schema", "strict one-action validation", "shield", COLORS["protocol"]),
        (946, 184, "State validation", "active handles + exact columns", "shield", COLORS["protocol"]),
        (1234, 184, "Compile / authorize", "typed IR or read-only SQL", "terminal", COLORS["execution"]),
        (1522, 184, "SQLite execution", "timeout + resource limits", "database", COLORS["execution"]),
    ]
    for idx, (x, y, title, detail, icon, color) in enumerate(steps, start=1):
        c.number(x + 20, y - 14, idx, color)
        c.card(x, y, 250, 100, title, [detail], color=color, fill="#FFFFFF", icon=icon)
        if idx < len(steps):
            c.line([(x + 250, y + 50), (steps[idx][0], y + 50)], color=COLORS["protocol"] if idx < 4 else COLORS["execution"])

    lower = [
        (1522, 352, "Validate result", "types · rows · bytes · cells", "shield", COLORS["execution"]),
        (1234, 352, "Atomic publish", "artifact / observation + derivation", "table", COLORS["success"]),
        (946, 352, "Update EnvironmentState", "immutable record + active membership", "stack", COLORS["state"]),
        (658, 352, "Structured tool result", "causal feedback only", "document", COLORS["state"]),
    ]
    for idx, (x, y, title, detail, icon, color) in enumerate(lower, start=7):
        c.number(x + 20, y - 14, idx, color)
        c.card(x, y, 250, 100, title, [detail], color=color, fill="#FFFFFF", icon=icon)
    c.line([(1647, 284), (1647, 352)], color=COLORS["execution"])
    c.line([(1522, 402), (1484, 402)], color=COLORS["success"])
    c.line([(1234, 402), (1196, 402)], color=COLORS["state"])
    c.line([(946, 402), (908, 402)], color=COLORS["state"])
    c.line([(658, 402), (590, 402), (590, 322), (207, 322), (207, 284)], color=COLORS["protocol"])
    c.text(388, 314, "next semantic turn", cls="edge", fill=COLORS["protocol"], anchor="middle")

    c.group(48, 498, 610, 252, "STATE-PRESERVING ERROR LANE", COLORS["error"])
    c.card(72, 554, 194, 112, "Structured error", ["carrier · validation", "execution · resource", "stable code and path"], color=COLORS["error"], fill=COLORS["light_red"], icon="error")
    c.card(288, 554, 154, 112, "Rollback", ["no artifact", "no observation", "no checkpoint"], color=COLORS["error"], fill=COLORS["light_red"], icon="shield")
    c.card(464, 554, 170, 112, "LAST ERROR", ["latest causal feedback", "next turn may recover"], color=COLORS["warning"], fill=COLORS["light_amber"], icon="document")
    c.line([(266, 610), (288, 610)], color=COLORS["error"])
    c.line([(442, 610), (464, 610)], color=COLORS["error"])
    c.line([(658, 610), (686, 610), (686, 476), (207, 476), (207, 452)], color=COLORS["error"], dash="6 5")
    c.text(350, 478, "state hash unchanged", cls="edge", fill=COLORS["error"], anchor="middle")

    c.group(682, 498, 500, 252, "TERMINAL LANE", COLORS["success"])
    c.card(706, 554, 202, 112, "answer(active artifact)", ["exact relation evidence", "columns · rows · order"], color=COLORS["success"], fill="#EAF8F0", icon="table")
    c.card(932, 554, 226, 112, "Collect exact artifact", ["bounded full relation", "no model-authored values"], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")
    c.line([(908, 610), (932, 610)], color=COLORS["success"])

    c.group(1206, 498, 666, 252, "HIDDEN EVALUATION · NO-LEAK", COLORS["hidden"], dash="8 5")
    c.card(1230, 554, 184, 112, "Reference evaluator", ["runs after termination", "hidden SQL / rows"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="lock")
    c.card(1436, 554, 180, 112, "Official score", ["bird-set", "compatibility view"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="audit")
    c.card(1638, 554, 210, 112, "Strict artifact audit", ["schema · sequence/bag", "duplicates · NULL · empty"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="audit")
    c.line([(1158, 610), (1230, 610)], color=COLORS["hidden"])
    c.line([(1414, 610), (1436, 610)], color=COLORS["hidden"])
    c.line([(1616, 610), (1638, 610)], color=COLORS["hidden"])
    c.line([(1340, 536), (1340, 514), (1045, 514), (1045, 498)], color=COLORS["error"], dash="2 5", marker="stop")
    c.text(1210, 510, "forbidden return flow", cls="edge", fill=COLORS["error"], anchor="middle")

    c.group(48, 778, 1824, 174, "AUDIT, RESOURCE GUARDS, AND ADMISSION", COLORS["hidden"])
    c.card(72, 832, 390, 88, "Immutable audit record", ["provider identity/history · state hashes · checkpoint path", "artifact dependencies · error events · fresh replay · no-leak"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="document")
    c.card(484, 832, 394, 88, "Runtime and batch guards", ["model turns · calls · checkpoints · restores · SQL timeout", "artifact rows/bytes/cells · attempts · tokens · wall time · failure streaks"], color=COLORS["warning"], fill=COLORS["light_amber"], icon="shield")
    c.card(900, 832, 406, 88, "Diagnostic trajectory", ["structure + replay + identity + causal-history gates", "Correct trajectories are still not automatically training data"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="audit")
    c.card(1328, 832, 520, 88, "LOCKED: scheme-aware export and explicit admission", ["SFT/RL export remains disabled until a separate admission decision", "Never mix with atomic or native-tool-bundle result directories"], color=COLORS["error"], fill=COLORS["light_red"], icon="lock", dash="6 5")
    c.line([(462, 876), (484, 876)], color=COLORS["hidden"])
    c.line([(878, 876), (900, 876)], color=COLORS["hidden"])
    c.line([(1306, 876), (1328, 876)], color=COLORS["error"], dash="6 5")

    c.formula("F10", 64, 972, 900, 78, caption="Success / failure state invariant")
    c.formula("F12", 986, 972, 870, 78, caption="Independent strict artifact audit")
    return c


def visual_standard(formulas: dict[str, str]) -> Canvas:
    c = Canvas("Checkpoint-RelAlg Visual Standard", formulas)
    c.title_block(
        "Checkpoint-RelAlg Visual Standard",
        "Reusable OmniGraffle grammar · English Times New Roman · vector icons · LaTeX equations",
        status="STANDARD V1",
    )

    c.group(48, 138, 560, 300, "TYPOGRAPHY AND GRID", COLORS["ink"])
    typography = [
        ("Canvas title", "34 pt · Bold"), ("Section heading", "20 pt · Bold"),
        ("Node title", "15 pt · Bold"), ("Body", "12.5 pt · Regular"),
        ("Edge label", "11 pt · Italic"), ("Chip / footnote", "10.5 / 10 pt"),
    ]
    for idx, (label, spec) in enumerate(typography):
        y = 194 + idx * 34
        c.text(74, y, label, cls="body", weight="bold")
        c.text(330, y, spec, cls="body", fill=COLORS["muted"])
    c.multiline(74, 404, ["Font: Times New Roman", "Canvas: 1920 × 1080 · 64 px safe margin", "8 px base grid · 24 px major grid · 24 px node gap"], cls="foot", fill=COLORS["muted"], step=17)

    c.group(632, 138, 716, 300, "SEMANTIC COLOR TOKENS", COLORS["protocol"])
    swatches = [
        ("Provider", COLORS["provider"]), ("Protocol", COLORS["protocol"]),
        ("State / Artifact", COLORS["state"]), ("Execution", COLORS["execution"]),
        ("Checkpoint", COLORS["checkpoint"]), ("Success", COLORS["success"]),
        ("Error", COLORS["error"]), ("Hidden / Audit", COLORS["hidden"]),
        ("Warning", COLORS["warning"]), ("Ink", COLORS["ink"]),
    ]
    for idx, (name, color) in enumerate(swatches):
        col = idx % 2
        row = idx // 2
        x = 658 + col * 330
        y = 190 + row * 43
        c.rect(x, y, 36, 24, fill=color, stroke=color, rx=5)
        c.text(x + 48, y + 17, name, cls="body", weight="bold")
        c.text(x + 202, y + 17, color, cls="foot", fill=COLORS["muted"])

    c.group(1372, 138, 500, 300, "VECTOR ICON GRAMMAR", COLORS["state"])
    icons = [
        ("document", "Task / record"), ("model", "Model / provider"),
        ("shield", "Validator / guard"), ("database", "Source database"),
        ("table", "Relation artifact"), ("eye", "Observation"),
        ("checkpoint", "Snapshot / phase"), ("audit", "Audit / replay"),
        ("lock", "No-leak / disabled"), ("error", "Structured error"),
    ]
    for idx, (icon, label) in enumerate(icons):
        col = idx % 2
        row = idx // 2
        x = 1398 + col * 224
        y = 184 + row * 45
        c.icon(icon, x, y, 26, COLORS["state"] if icon not in {"error", "lock"} else COLORS["error"])
        c.text(x + 38, y + 19, label, cls="body")

    c.group(48, 466, 742, 286, "CONNECTOR GRAMMAR", COLORS["protocol"])
    connector_specs = [
        ("Successful semantic transition", COLORS["protocol"], None, "arrow"),
        ("Data / relation flow", COLORS["state"], None, "arrow"),
        ("Execution request", COLORS["execution"], None, "arrow"),
        ("Structured error / rollback", COLORS["error"], None, "arrow"),
        ("Checkpoint / restore control", COLORS["checkpoint"], "8 5", "arrow"),
        ("Optional / retry / diagnostic", COLORS["warning"], "6 5", "arrow"),
        ("Forbidden / no-leak direction", COLORS["hidden"], "2 5", "stop"),
    ]
    for idx, (label, color, dash, marker) in enumerate(connector_specs):
        y = 522 + idx * 30
        c.text(74, y + 5, label, cls="body")
        c.line([(420, y), (744, y)], color=color, dash=dash, marker=marker)

    c.group(814, 466, 500, 286, "SHAPE GRAMMAR", COLORS["state"])
    c.card(838, 520, 196, 72, "Process", ["rounded rectangle"], color=COLORS["protocol"], fill=COLORS["light_blue"], icon="shield")
    c.card(1056, 520, 234, 72, "Relation artifact", ["table card"], color=COLORS["state"], fill=COLORS["light_teal"], icon="table")
    c.card(838, 612, 196, 72, "Checkpoint", ["snapshot + flag"], color=COLORS["checkpoint"], fill=COLORS["light_violet"], icon="checkpoint")
    c.card(1056, 612, 234, 72, "Hidden evaluator", ["dashed secure zone"], color=COLORS["hidden"], fill=COLORS["light_gray"], icon="lock", dash="6 5")
    c.text(838, 718, "Use native vectors or SVG only. Never rasterize source icons.", cls="foot", fill=COLORS["muted"])

    c.group(1338, 466, 534, 286, "LATEX EQUATION POLICY", COLORS["ink"])
    c.multiline(1364, 522, [
        "1 · Store canonical source in checkpoint_relalg_formulas.tex.",
        "2 · Render with LaTeX and Times-compatible math fonts.",
        "3 · Convert fonts to vector paths with dvisvgm.",
        "4 · Place on the EQUATIONS layer; keep data-latex metadata.",
        "5 · Never approximate formulas with Unicode text boxes.",
    ], step=29)
    c.formula("F11", 1364, 676, 482, 58)

    c.group(48, 780, 1170, 218, "OMNIGRAFFLE LAYER AND CONSTRUCTION CONTRACT", COLORS["hidden"])
    c.multiline(74, 838, [
        "Layer order: BACKGROUND → GROUPS → CONNECTORS → NODES → ICONS → LABELS → EQUATIONS → CALLOUTS",
        "Build order: claim → lanes/groups → nodes → connectors → labels → vector icons → LaTeX equations → authority review",
        "Stable names: STATE.EnvironmentState · TOOL.aggregate · BOUNDARY.no_leak · FORMULA.F10",
        "Editable .graffle + LaTeX source are authoritative. SVG/PDF are vector exports; PNG is preview only.",
    ], step=32)

    c.group(1242, 780, 630, 218, "RELEASE CHECKLIST", COLORS["success"])
    c.multiline(1268, 836, [
        "✓ English Times New Roman throughout",
        "✓ Exact executable tool names and current protocol status",
        "✓ Authority, working-memory, and no-leak boundaries explicit",
        "✓ Failure path preserves logical state and creates no artifact",
        "✓ LaTeX source + vector rendering for every equation",
        "✓ OmniGraffle, SVG, PDF, PNG, and manifest verified",
    ], step=27)
    c.text(64, 1040, f"Standard identity: {STANDARD_VERSION}", cls="foot", fill=COLORS["muted"])
    return c


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_preview(svg_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if not converter:
        return
    _run([converter, "-f", "pdf", "-o", str(svg_path.with_suffix(".pdf")), str(svg_path)], cwd=HERE)
    _run([converter, "-f", "png", "-w", "1920", "-o", str(svg_path.with_suffix(".png")), str(svg_path)], cwd=HERE)


def main() -> int:
    formulas = render_equations()
    formula_sheet = render_formula_sheet()
    canvases = {
        "01_checkpoint_relalg_system_architecture": architecture(formulas),
        "02_checkpoint_relalg_tool_surface": tool_surface(formulas),
        "03_checkpoint_relalg_checkpoint_reasoning": checkpoint_reasoning(formulas),
        "04_checkpoint_relalg_causal_execution_audit": causal_execution(formulas),
        "05_checkpoint_relalg_visual_standard": visual_standard(formulas),
    }
    outputs: list[Path] = []
    for stem, canvas in canvases.items():
        path = HERE / f"{stem}.svg"
        path.write_text(canvas.svg(), encoding="utf-8")
        export_preview(path)
        outputs.extend(
            candidate for candidate in (path, path.with_suffix(".pdf"), path.with_suffix(".png"))
            if candidate.exists()
        )
    outputs.extend(
        candidate
        for candidate in (
            HERE / "CHECKPOINT_RELALG_VISUAL_STANDARD.md",
            HERE / "README.md",
            HERE / "checkpoint_relalg_formulas.tex",
            formula_sheet,
            *(EQUATION_DIR / f"{formula_id}.svg" for formula_id in FORMULAS),
            *(HERE / f"{stem}.graffle" for stem in canvases),
        )
        if candidate.exists()
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {
        "schema_version": "checkpoint-relalg-visual-artifact-manifest-v1",
        "visual_standard_version": STANDARD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": commit,
        "font_family": "Times New Roman",
        "equation_source": "checkpoint_relalg_formulas.tex",
        "equation_renderer": "latex + dvisvgm --no-fonts",
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)
        },
        "omnigraffle_sources": [f"{stem}.graffle" for stem in canvases],
    }
    (HERE / "visual_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"canvases": len(canvases), "outputs": len(outputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
