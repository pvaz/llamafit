// LlamaFit. Copyright (C) 2026 Paulo Vaz.
// SPDX-License-Identifier: AGPL-3.0-or-later
// This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
//
// The whole dashboard. It fetches JSON from this same server and draws it, and that is
// all it does: it computes no budget, no speed and no score, because every one of those
// arrived already computed and a second opinion here would be a second answer for the
// project to be wrong in.
//
// It holds no English either. Every word comes from GET /api/v1/ui, which builds it
// through the gettext catalogs the command line reads, so the page speaks whatever
// language LlamaFit was started in and a translator never has to open this file.
//
// The one thing it does repeat from Python is how a number is written: which unit a byte
// count wants and how many decimals a figure gets. The characters that vary by language --
// the group and decimal separators -- come from the server with everything else, so the
// repetition is arithmetic, not language. llamafit/units.py is the original, and
// tests/unit/test_web_strings.py pins the two together.
//
// No framework, no bundler, no dependency. Plain functions, one global `ui`, and
// textContent everywhere rather than innerHTML, so a model id or a file path from
// somebody's disk can never become markup.

"use strict";

const API = "/api/v1";
const BYTE_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"];

/** Everything the server told us to draw with: strings, labels, punctuation. */
let ui = null;

/** What the reader last asked the board for; also what Plan and Simulate feed into. */
const asked = {
  use_case: "general",
  require: [],
  prefer: "balanced",
  min_context: 0,
  max_download: "",
  all_quants: false,
  vision: true,
  limit: 10,
  sort: "score",
  profile: "",
  memory: "",
  ram: "",
  cpu_cores: "",
};

/** The last board, kept so a row can be expanded without asking for it again. */
let board = null;

// ---------------------------------------------------------------- words and numbers

/** One of the page's own strings, by the key strings.py filed it under. */
function t(key) {
  if (key.startsWith("columns.")) return col(key.slice(8));
  return (ui && ui.strings[key]) || key;
}

/** One column heading, by the name the server knows it as. */
function col(name) {
  return (ui && ui.columns[name]) || name;
}

/** The word for one enumerated value, or the value itself when nobody named it. */
function label(table, value) {
  const words = ui && ui.labels[table];
  return (words && words[value]) || value || "";
}

/** Fill "%(name)s" placeholders the way Python's percent formatting does. */
function format(template, values) {
  return template.replace(/%\((\w+)\)s/g, (whole, name) =>
    name in values ? String(values[name]) : whole,
  );
}

/** Swap the English separators for the ones this language writes numbers with. */
function localise(text) {
  const group = ui.format.group;
  const decimal = ui.format.decimal;
  if (group === "," && decimal === ".") return text;
  return text.replace(/[,.]/g, (c) => (c === "," ? group : decimal));
}

/** A figure with its digits grouped and this language's punctuation. */
function number(value, digits = 1) {
  const fixed = Number(value).toFixed(digits);
  const [whole, fraction] = fixed.split(".");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return localise(fraction === undefined ? grouped : `${grouped}.${fraction}`);
}

/** A byte count the way llamafit/units.py writes one. */
function bytes(n) {
  if (n === null || n === undefined) return ui.format.unknown_size;
  let value = Number(n);
  let index = 0;
  while (value >= 1024 && index < BYTE_UNITS.length - 1) {
    value /= 1024;
    index += 1;
  }
  if (index === 0) return `${n} B`;
  return `${number(value, 1)} ${BYTE_UNITS[index]}`;
}

/** A context length compactly -- 32K -- and plainly when it is not a round number. */
function context(tokens) {
  if (!tokens) return "";
  return tokens % 1024 === 0 ? `${tokens / 1024}K` : number(tokens, 0);
}

/** A share as a percentage, or the word for one nobody could compute. */
function percent(share) {
  if (share === null || share === undefined) return ui.format.unknown_size;
  return localise(`${(share * 100).toFixed(0)}%`);
}

// ---------------------------------------------------------------- DOM and transport

/** Build one element: tag, attributes, and children that are text or elements. */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === false || value === null || value === undefined) continue;
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

/** Replace an element's children with one set of new ones. */
function fill(target, children) {
  target.replaceChildren(...[].concat(children).filter(Boolean));
}

/** Fetch JSON from this server, turning LlamaFit's own error shape into an exception. */
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = (body && body.error) || {};
    const failure = new Error(error.message || `${response.status}`);
    failure.hint = error.hint;
    failure.command = error.command;
    throw failure;
  }
  return body;
}

/** Show a failure where the reader is looking, with the hint that would fix it. */
function showFailure(error) {
  const banner = document.getElementById("failure");
  const lines = [`${t("app.error")}: ${error.message}`];
  if (error.command) lines.push(`${t("app.command")}: ${error.command}`);
  if (error.hint) lines.push(`${t("app.hint")}: ${error.hint}`);
  banner.textContent = lines.join("\n");
  banner.hidden = false;
}

/** Clear the failure banner before trying something new. */
function clearFailure() {
  document.getElementById("failure").hidden = true;
}

/** Run something that talks to the server, saying so while it does. */
async function working(target, run) {
  if (target) fill(target, el("p", { class: "muted", text: t("app.loading") }));
  clearFailure();
  try {
    return await run();
  } catch (error) {
    showFailure(error);
    if (target) fill(target, []);
    return null;
  }
}

// ---------------------------------------------------------------- tables

/** A table from a list of column definitions and a list of rows. */
function table(columns, rows, options = {}) {
  const head = el(
    "tr",
    {},
    columns.map((column) =>
      el("th", {
        text: column.heading,
        class: column.number ? "number" : column.wrap ? "wrap" : null,
        scope: "col",
        "data-sort": column.sort || null,
      }),
    ),
  );
  const body = rows.map((row) =>
    el(
      "tr",
      row.attrs || {},
      columns.map((column) =>
        el("td", {
          text: row.cells[column.key] === undefined ? "" : row.cells[column.key],
          class: [column.number ? "number" : column.wrap ? "wrap" : "", row.classes?.[column.key] || ""]
            .filter(Boolean)
            .join(" ") || null,
        }),
      ),
    ),
  );
  const parts = [el("thead", {}, head), el("tbody", {}, body)];
  if (options.caption) parts.unshift(el("caption", { text: options.caption }));
  return el("table", {}, parts);
}

/** A two-column list of facts, for the host and for a plan's files. */
function facts(pairs) {
  const nodes = [];
  for (const [term, value] of pairs) {
    if (value === null || value === undefined || value === "") continue;
    nodes.push(el("dt", { text: term }), el("dd", { text: value }));
  }
  return el("dl", { class: "facts" }, nodes);
}

// ---------------------------------------------------------------- the board

const BOARD_COLUMNS = [
  { key: "rank", name: "rank", number: true },
  { key: "model", name: "model" },
  { key: "quant", name: "quant" },
  { key: "score", name: "score", number: true, sort: "score" },
  { key: "gen", name: "gen", number: true, sort: "speed" },
  { key: "confidence", name: "confidence" },
  { key: "verdict", name: "verdict" },
  { key: "mode", name: "mode" },
  { key: "context", name: "context", number: true, sort: "context" },
  { key: "quality", name: "quality", number: true, sort: "quality" },
  { key: "vram", name: "vram", number: true },
  { key: "prompt", name: "prompt", number: true },
  { key: "size", name: "size", number: true, sort: "size" },
  { key: "ram", name: "ram", number: true },
];

/** One board row's cells, read straight off the document the API returned. */
function boardCells(row) {
  const candidate = row.candidate;
  const placement = candidate.placement;
  const budget = placement && placement.budget;
  const speed = candidate.speed;
  const score = candidate.score;
  return {
    rank: row.rank === null ? "" : number(row.rank, 0),
    model: row.model_id,
    quant: row.quant,
    score: score ? number(score.total) : "",
    gen: speed ? number(speed.gen_tps) : "",
    confidence: speed ? label("confidence", speed.confidence) : "",
    verdict: budget ? label("verdict", budget.verdict) : "",
    mode: placement ? label("mode", placement.mode) : "",
    context: placement ? context(placement.max_context_fit) : "",
    quality: score ? number(score.quality, 0) : "",
    vram: budget ? bytes(budget.vram_required) : "",
    prompt: speed ? number(speed.pp_tps, 0) : "",
    size: bytes(row.download_bytes),
    ram: budget ? bytes(budget.ram_required) : "",
  };
}

/** Draw the ranked board, its captions, and the candidates that did not qualify. */
function renderBoard() {
  const target = document.getElementById("board-table");
  if (!board.rows.length) {
    fill(target, el("p", { text: t("board.empty") }));
  } else {
    const columns = BOARD_COLUMNS.map((column) => ({ ...column, heading: col(column.name) }));
    const rows = board.rows.map((row) => ({
      cells: boardCells(row),
      attrs: { class: "row", "data-model": row.model_id, "data-quant": row.quant },
      classes: {
        verdict:
          row.candidate.placement &&
          `verdict-${row.candidate.placement.budget.verdict}`,
      },
    }));
    const drawn = table(columns, rows);
    drawn.addEventListener("click", onBoardClick);
    fill(target, drawn);
  }
  renderCaptions();
  renderExcluded();
  fillModelChoices();
}

/** What the board's numbers are, said once under the table rather than on every row. */
function renderCaptions() {
  const same = board.planned_context === board.requested_context;
  const first = format(t(same ? "board.caption" : "board.caption_split"), {
    working: context(board.working_context),
    planned: context(board.planned_context),
    requested: context(board.requested_context),
    use_case: board.needs.use_case,
  });
  const weights = Object.entries(board.weights)
    .map(([part, weight]) => `${label("score_part", part)} ${number(weight, 2)}`)
    .join(", ");
  fill(document.getElementById("board-captions"), [
    el("p", { text: first }),
    el("p", { text: format(t("board.weights"), { weights }) }),
  ]);
}

/** The candidates that were not ranked, each with the reason it was not. */
function renderExcluded() {
  const columns = [
    { key: "model", heading: col("model") },
    { key: "quant", heading: col("quant") },
    { key: "why", heading: col("why_not"), wrap: true },
  ];
  const rows = board.excluded.map((row) => ({
    cells: {
      model: row.model_id,
      quant: row.quant,
      why: row.candidate.excluded_because || "",
    },
  }));
  fill(
    document.getElementById("board-excluded"),
    rows.length ? table(columns, rows) : el("p", { class: "muted", text: t("app.none") }),
  );
}

/** Expand a row into what produced it, or sort the board by a column heading. */
function onBoardClick(event) {
  const heading = event.target.closest("th[data-sort]");
  if (heading) {
    asked.sort = heading.dataset.sort;
    loadBoard();
    return;
  }
  const row = event.target.closest("tr.row");
  if (!row) return;
  const open = row.nextElementSibling;
  if (open && open.classList.contains("detail")) {
    open.remove();
    return;
  }
  const found = board.rows.find(
    (candidate) =>
      candidate.model_id === row.dataset.model && candidate.quant === row.dataset.quant,
  );
  if (!found) return;
  const cell = el("td", { colspan: BOARD_COLUMNS.length }, explanation(found));
  row.after(el("tr", { class: "detail" }, cell));
}

/** One row expanded: the score with its parts, the speed, the budget and the ladder. */
function explanation(row) {
  const candidate = row.candidate;
  const parts = [];
  if (candidate.score) parts.push(scoreTable(candidate.score));
  if (candidate.speed) parts.push(speedBlock(candidate.speed, board.working_context));
  if (candidate.placement) {
    parts.push(budgetBlock(candidate.placement.budget));
    parts.push(tiersBlock(candidate.placement));
  }
  const grid = el("div", { class: "detail-grid" }, parts);
  const act = el("button", { type: "button", text: t("board.plan_this") });
  act.addEventListener("click", () => planFor(row.model_id, row.quant));
  const extras = [grid, el("div", { class: "actions" }, act)];
  if (candidate.placement && candidate.placement.notes.length) {
    extras.splice(
      1,
      0,
      el(
        "div",
        {},
        [el("h3", { text: t("plan.notes") })].concat(
          candidate.placement.notes.map((note) => el("p", { class: "muted", text: note })),
        ),
      ),
    );
  }
  if (candidate.excluded_because) {
    extras.push(el("p", { text: candidate.excluded_because }));
  }
  return extras;
}

/** The composite score expanded into its four parts, their weights and what each added. */
function scoreTable(score) {
  const columns = [
    { key: "part", heading: col("part") },
    { key: "score", heading: col("score"), number: true },
    { key: "weight", heading: col("weight"), number: true },
    { key: "adds", heading: col("adds"), number: true },
  ];
  const rows = ["quality", "speed", "fit", "context"].map((part) => {
    const weight = score.weights[part] || 0;
    return {
      cells: {
        part: label("score_part", part),
        score: number(score[part]),
        weight: number(weight, 2),
        adds: number(score[part] * weight),
      },
    };
  });
  return el("div", { class: "scroll" }, [
    table(columns, rows, { caption: format(t("board.score_total"), { total: number(score.total) }) }),
  ]);
}

/** The two speeds and where a token's time goes. */
function speedBlock(speed, at) {
  const headline = format(t("plan.speed_headline"), {
    gen: number(speed.gen_tps),
    prompt: number(speed.pp_tps, 0),
    context: context(at),
    confidence: label("confidence", speed.confidence),
  });
  const keys = [
    "vram_seconds_per_token",
    "ram_seconds_per_token",
    "overhead_seconds_per_token",
  ];
  const total = keys.reduce((sum, key) => sum + (speed[key] || 0), 0);
  const parts = [el("p", { text: headline })];
  if (total > 0) {
    const columns = [
      { key: "part", heading: col("token_time") },
      { key: "seconds", heading: col("seconds"), number: true },
      { key: "share", heading: col("share"), number: true },
    ];
    const rows = keys.map((key) => ({
      cells: {
        part: label("token_time", key),
        seconds: number(speed[key] || 0, 4),
        share: percent((speed[key] || 0) / total),
      },
    }));
    parts.push(el("div", { class: "scroll" }, table(columns, rows)));
  }
  for (const note of speed.notes || []) parts.push(el("p", { class: "muted", text: note }));
  return el("div", {}, parts);
}

/** The memory budget, line by line, with where each line's figure came from. */
function budgetBlock(budget) {
  const columns = [
    { key: "component", heading: col("component"), wrap: true },
    { key: "where", heading: col("where") },
    { key: "size", heading: col("size"), number: true },
    { key: "from", heading: col("from") },
  ];
  const rows = budget.lines.map((line) => ({
    cells: {
      component: label("component", line.component),
      where: label("pool", line.pool),
      size: bytes(line.bytes),
      from: label("source", line.exact ? "exact" : "formula"),
    },
  }));
  const totals = [
    el("p", {
      text: format(t("plan.card_totals"), {
        required: bytes(budget.vram_required),
        available: bytes(budget.vram_available),
        share: percent(budget.vram_utilisation),
      }),
    }),
    el("p", {
      text: format(t("plan.system_totals"), {
        required: bytes(budget.ram_required),
        available: bytes(budget.ram_available),
        share: percent(budget.ram_utilisation),
      }),
    }),
    el("p", {
      class: `verdict-${budget.verdict}`,
      text: label("verdict_sentence", budget.verdict),
    }),
  ];
  const notes = budget.lines
    .filter((line) => line.note)
    .map((line) =>
      el("p", { class: "muted", text: `${label("component", line.component)}: ${line.note}` }),
    );
  return el("div", {}, [
    el("h3", { text: t("plan.budget") }),
    el("div", { class: "scroll" }, table(columns, rows)),
    ...totals,
    ...notes,
  ]);
}

/** The context ladder, and what happens at each rung. */
function tiersBlock(placement) {
  if (!placement.tiers.length) return null;
  const columns = [
    { key: "tokens", heading: col("context_full"), number: true },
    { key: "vram", heading: col("card_needs"), number: true },
    { key: "state", heading: col("on_this_machine") },
  ];
  const rows = placement.tiers.map((tier) => ({
    cells: {
      tokens: context(tier.tokens),
      vram: bytes(tier.vram_required),
      state: label("verdict", tier.verdict),
    },
    classes: { state: `verdict-${tier.verdict}` },
  }));
  const parts = [
    el("h3", { text: t("plan.tiers") }),
    el("div", { class: "scroll" }, table(columns, rows)),
    el("p", { class: "muted", text: t("plan.tiers_caption") }),
  ];
  if (placement.tiers.some((tier) => tier.verdict === "too-tight")) {
    parts.push(
      el("p", {
        class: "verdict-too-tight",
        text: label("verdict_sentence", "too-tight"),
      }),
    );
  }
  return el("div", {}, parts);
}

// ---------------------------------------------------------------- panels

/** The query string every board request carries: the needs, plus any simulation. */
function query() {
  const parameters = new URLSearchParams();
  parameters.set("use_case", asked.use_case);
  for (const capability of asked.require) parameters.append("require", capability);
  parameters.set("prefer", asked.prefer);
  if (asked.min_context) parameters.set("min_context", String(asked.min_context));
  if (asked.max_download) parameters.set("max_download", asked.max_download);
  parameters.set("all_quants", String(asked.all_quants));
  parameters.set("vision", String(asked.vision));
  parameters.set("limit", String(asked.limit));
  parameters.set("sort", asked.sort);
  for (const name of ["profile", "memory", "ram", "cpu_cores"]) {
    if (asked[name]) parameters.set(name, String(asked[name]));
  }
  return parameters.toString();
}

/** Ask for the board and draw it. */
async function loadBoard() {
  const answer = await working(document.getElementById("board-table"), () =>
    api(`${API}/models/top?${query()}`),
  );
  if (!answer) return;
  board = answer;
  renderBoard();
}

/** The scan: the machine in the header, and the same facts on the Host panel. */
async function loadSystem(refresh = false) {
  const report = await working(document.getElementById("host-facts"), () =>
    refresh ? api(`${API}/scan`, { method: "POST" }) : api(`${API}/system`),
  );
  if (!report) return;
  renderHost(report);
  loadDoctor();
}

/** The machine, as facts rather than as prose: identifiers, numbers and units only. */
function renderHost(report) {
  const host = report.host;
  const cpu = host.cpu;
  const memory = host.memory;
  document.getElementById("host-line").textContent = [
    `${host.os} ${host.os_version} (${host.arch})`,
    cpu.model,
    bytes(memory.total_bytes),
    ...host.gpus.map((gpu) => `${gpu.name} ${bytes(gpu.vram_total_bytes)}`),
  ]
    .filter(Boolean)
    .join(" · ");

  // Values are identifiers, numbers and unit symbols joined by a middle dot rather than
  // sentences: a sentence would need a message per shape, and the terminal already owns
  // those. What is a word here -- the row label -- is the terminal's own word.
  const rows = [
    [t("host.os"), `${host.os} ${host.os_version} (${host.arch})`],
    [t("host.cpu"), `${cpu.model} · ${cpu.physical_cores}/${cpu.logical_cores}`],
    [
      t("host.memory"),
      [
        `${bytes(memory.available_bytes)} / ${bytes(memory.total_bytes)}`,
        memory.type,
        memory.speed_mts ? `${memory.speed_mts} MT/s` : null,
        memory.bandwidth_gbps ? `${number(memory.bandwidth_gbps)} GB/s` : null,
      ]
        .filter(Boolean)
        .join(" · "),
    ],
  ];
  for (const gpu of host.gpus) {
    rows.push([
      t("host.gpu"),
      [gpu.name, bytes(gpu.vram_total_bytes), gpu.backend_hint, gpu.driver]
        .filter(Boolean)
        .join(" · "),
    ]);
  }
  for (const disk of host.disks) {
    rows.push([t("host.disk"), `${disk.path} · ${bytes(disk.free_bytes)} / ${bytes(disk.total_bytes)}`]);
  }
  fill(document.getElementById("host-facts"), facts(rows));
  fill(
    document.getElementById("host-llamacpp"),
    facts([
      [
        t("host.installed"),
        [report.llamacpp.path, report.llamacpp.build ? `b${report.llamacpp.build}` : null]
          .filter(Boolean)
          .join(" · "),
      ],
      [t("host.llamacpp"), (report.llamacpp.backends || []).join(", ")],
    ]),
  );
  document.getElementById("sim-badge").hidden = !host.simulated;
}

/** Probe by probe: what worked, what did not, and what would unlock more. */
async function loadDoctor() {
  const diagnosis = await working(null, () => api(`${API}/doctor`));
  if (!diagnosis) return;
  const nodes = diagnosis.findings.map((finding) =>
    el("div", { class: `notice ${finding.level === "ok" ? "" : finding.level}` }, [
      el("strong", { text: finding.title }),
      el("p", { text: finding.detail }),
      finding.hint ? el("p", { class: "muted", text: `${t("app.hint")}: ${finding.hint}` }) : null,
    ]),
  );
  fill(document.getElementById("host-findings"), nodes);
}

/** Say once, where a reader will see it, that a catalog file could not be read. */
async function loadCatalogProblems() {
  const answer = await api(`${API}/catalog/problems`).catch(() => null);
  const banner = document.getElementById("catalog-problems");
  if (!answer || !answer.count) return;
  banner.textContent = answer.message;
  banner.hidden = false;
}

/** The hardware profiles this machine has, for the Simulate panel's list. */
async function loadProfiles() {
  const profiles = await api(`${API}/profiles`).catch(() => []);
  const select = document.getElementById("simulate-profile");
  fill(select, [
    el("option", { value: "", text: t("simulate.live") }),
    ...profiles.map((profile) => el("option", { value: profile.name, text: profile.name })),
  ]);
}

/** Offer the board's models on the Plan panel, so nothing has to be typed. */
function fillModelChoices() {
  const select = document.getElementById("plan-model");
  const chosen = select.value;
  const ids = [...new Set(board.rows.concat(board.excluded).map((row) => row.model_id))];
  fill(select, ids.map((id) => el("option", { value: id, text: id })));
  if (ids.includes(chosen)) select.value = chosen;
}

/** Plan one model and show the budget, the ladder and the command line. */
async function planFor(modelId, quant) {
  show("plan");
  const form = document.getElementById("plan-form");
  form.elements.model.value = modelId;
  if (quant) form.elements.quant.value = quant;
  await submitPlan();
}

/** Send the Plan form and draw what came back. */
async function submitPlan() {
  const form = document.getElementById("plan-form");
  const body = {
    model: form.elements.model.value,
    quant: form.elements.quant.value || null,
    context: Number(form.elements.context.value) || null,
    ub: Number(form.elements.ub.value) || null,
    vision: form.elements.vision.checked,
  };
  for (const name of ["profile", "memory", "ram", "cpu_cores"]) {
    if (asked[name]) body[name] = name === "cpu_cores" ? Number(asked[name]) : asked[name];
  }
  if (!body.model) {
    fill(document.getElementById("plan-output"), el("p", { text: t("plan.pick_first") }));
    return;
  }
  const plan = await working(document.getElementById("plan-output"), () =>
    api(`${API}/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  if (!plan) return;
  renderPlan(plan);
}

/** Everything `llamafit plan` prints, laid out for a browser. */
function renderPlan(plan) {
  const fileLine = plan.model_present
    ? format(t("plan.file_here"), { path: plan.model_path })
    : format(t("plan.file_missing"), {
        path: plan.model_path,
        size: bytes(plan.download_bytes),
      });
  const parts = [
    el("h2", { text: `${plan.name} ${plan.quant}` }),
    el("p", { text: fileLine }),
  ];
  if (plan.speed) parts.push(speedBlock(plan.speed, plan.placement.context));
  parts.push(budgetBlock(plan.placement.budget));
  if (plan.requested_budget) {
    parts.push(
      el("h3", {
        text: format(t("plan.requested_cost"), { context: context(plan.requested_context) }),
      }),
    );
    parts.push(budgetBlock(plan.requested_budget));
  }
  const tiers = tiersBlock(plan.placement);
  if (tiers) parts.push(tiers);
  if (plan.placement.notes.length) {
    parts.push(el("h3", { text: t("plan.notes") }));
    for (const note of plan.placement.notes) parts.push(el("p", { class: "muted", text: note }));
  }
  parts.push(el("h3", { text: t("plan.command") }));
  parts.push(el("pre", { text: plan.command.join(" ") }));
  const copy = el("button", { type: "button", text: t("plan.copy") });
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(plan.command.join(" "));
      copy.textContent = t("plan.copied");
    } catch (error) {
      showFailure(error);
    }
  });
  parts.push(el("div", { class: "actions" }, copy));
  fill(document.getElementById("plan-output"), parts);
}

/** Show one panel and mark its tab as the current one. */
function show(name) {
  for (const panel of document.querySelectorAll(".panel")) {
    panel.hidden = panel.id !== `panel-${name}`;
  }
  for (const tab of document.querySelectorAll("#tabs button")) {
    tab.setAttribute("aria-current", String(tab.dataset.panel === name));
  }
}

// ---------------------------------------------------------------- start-up

/** Put every word the server sent into the element that asked for it. */
function applyStrings() {
  for (const node of document.querySelectorAll("[data-t]")) {
    node.textContent = t(node.dataset.t);
  }
  document.documentElement.lang = ui.language;
  document.documentElement.dir = ui.direction;
  document.getElementById("estimate-notice").textContent = ui.estimate_notice;
  document.getElementById("sim-badge").textContent = t("simulate.badge");
  document.getElementById("source-link").href = ui.source_url;
}

/** Fill the two choice lists whose options are values rather than sentences. */
function fillChoices() {
  const useCases = ["general", "coding", "reasoning", "chat", "multimodal", "embedding"];
  fill(
    document.getElementById("needs-use-case"),
    useCases.map((value) => el("option", { value, text: value })),
  );
  fill(
    document.getElementById("needs-prefer"),
    ["balanced", "quality", "speed"].map((value) => el("option", { value, text: value })),
  );
  fill(
    document.getElementById("needs-capabilities"),
    Object.entries(ui.labels.capability).map(([value, word]) =>
      el("label", {}, [
        el("input", { type: "checkbox", name: "require", value }),
        el("span", { text: word }),
      ]),
    ),
  );
}

/** Wire every control up once, then ask the server for the first answer. */
function listen() {
  document.getElementById("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest("button[data-panel]");
    if (tab) show(tab.dataset.panel);
  });

  document.getElementById("needs-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    asked.use_case = form.elements.use_case.value;
    asked.require = [...form.querySelectorAll('input[name="require"]:checked')].map(
      (input) => input.value,
    );
    asked.prefer = form.elements.prefer.value;
    asked.min_context = Number(form.elements.min_context.value) || 0;
    asked.max_download = form.elements.max_download.value.trim();
    asked.all_quants = form.elements.all_quants.checked;
    asked.vision = form.elements.vision.checked;
    asked.limit = Number(form.elements.limit.value) || 10;
    show("board");
    loadBoard();
  });

  document.getElementById("simulate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    asked.profile = form.elements.profile.value;
    asked.memory = form.elements.memory.value.trim();
    asked.ram = form.elements.ram.value.trim();
    asked.cpu_cores = form.elements.cpu_cores.value.trim();
    document.getElementById("sim-badge").hidden = !(
      asked.profile ||
      asked.memory ||
      asked.ram ||
      asked.cpu_cores
    );
    show("board");
    loadBoard();
  });

  document.getElementById("simulate-clear").addEventListener("click", () => {
    document.getElementById("simulate-form").reset();
    Object.assign(asked, { profile: "", memory: "", ram: "", cpu_cores: "" });
    document.getElementById("sim-badge").hidden = true;
    loadBoard();
  });

  document.getElementById("plan-form").addEventListener("submit", (event) => {
    event.preventDefault();
    submitPlan();
  });

  document.getElementById("rescan").addEventListener("click", () => loadSystem(true));
}

/** Fetch the words, wire the page up, then ask for the machine and the board. */
async function boot() {
  ui = await api(`${API}/ui`);
  applyStrings();
  fillChoices();
  listen();
  show("board");
  loadCatalogProblems();
  loadProfiles();
  await loadSystem();
  await loadBoard();
}

boot().catch((error) => {
  const banner = document.getElementById("failure");
  banner.textContent = String(error && error.message ? error.message : error);
  banner.hidden = false;
});
