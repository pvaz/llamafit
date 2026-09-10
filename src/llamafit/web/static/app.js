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
  limit: 50,
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
        class:
          [column.number ? "number" : column.wrap ? "wrap" : "", column.cell || ""]
            .filter(Boolean)
            .join(" ") || null,
        scope: "col",
        "data-sort": column.sort || null,
        "aria-sort": column.sort && column.sort === options.sorted ? "descending" : null,
      }),
    ),
  );
  const body = rows.map((row) => {
    // A row may stop early and give the rest of its width to one cell. That is how a
    // candidate that was never ranked says why in the same list as the ones that were,
    // on the same line, without a column of its own that every other row would pay for.
    const upto = row.span ? columns.slice(0, row.span.at) : columns;
    const cells = upto.map((column) =>
      el("td", {
        text: row.cells[column.key] === undefined ? "" : row.cells[column.key],
        class:
          [
            column.number ? "number" : column.wrap ? "wrap" : "",
            column.cell || "",
            row.classes?.[column.key] || "",
          ]
            .filter(Boolean)
            .join(" ") || null,
      }),
    );
    if (row.span) {
      cells.push(
        el("td", {
          text: row.span.text,
          title: row.span.text,
          class: row.span.class || null,
          colspan: columns.length - row.span.at,
        }),
      );
    }
    return el("tr", row.attrs || {}, cells);
  });
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

// Seven columns, which are the first seven of the priority the terminal's board already
// worked out in llamafit/cli/render_board.py and llamafit/tui/board_view.py: the four a
// row cannot be told apart or acted on without, then how fast it runs, then whether it
// fits, then how much context it holds. The six the terminal admits after those --
// `Runs`, `Qual`, `Card`, `Prompt tok/s`, `Size` and `RAM` -- are all still on this page,
// inside the row, where the figure that produced them is. A browser has room for all
// fourteen and that is exactly the trap: fourteen columns of one weight is a table with
// no answer in it, and the answer is the reason somebody opened this.
const BOARD_COLUMNS = [
  { key: "rank", name: "rank", number: true, cell: "rank" },
  { key: "model", name: "model", cell: "name" },
  { key: "quant", name: "quant", cell: "quant" },
  { key: "score", name: "score", number: true, sort: "score", cell: "score" },
  { key: "gen", name: "gen", number: true, sort: "speed", cell: "tps" },
  { key: "confidence", name: "confidence", cell: "how" },
  { key: "verdict", name: "verdict", cell: "fit" },
  { key: "context", name: "context", number: true, sort: "context", cell: "ctx" },
];

/** Where an unranked row stops and its reason begins: after the three that identify it. */
const IDENTITY_COLUMNS = 3;

/** The columns actually drawn, kept so an expanded row knows how wide to be. */
let drawnColumns = BOARD_COLUMNS;

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
    context: placement ? context(placement.max_context_fit) : "",
  };
}

/** Whether every row's speed was arrived at the same way, so one sentence covers them.
 *
 * The terminal's rule, and for the terminal's reason: when the labels differ, no sentence
 * above the table is true of all of them and the word has to sit beside the figure. When
 * they agree, the sentence under the header says it once, in full, and the column would
 * be the same word repeated down the page.
 */
function oneConfidence() {
  const labels = board.rows.filter((row) => row.candidate.speed);
  return new Set(labels.map((row) => row.candidate.speed.confidence)).size <= 1;
}

/** Every candidate this request produced: the ranked ones, then the ones that were not.
 *
 * One list rather than two tables. A second table of failures is a second page to find
 * and lose your place in, and the reason a candidate was refused is the most useful
 * sentence on this page -- it is the one that says what to change about the request. An
 * unranked row keeps the three columns that identify it, aligned with every other row,
 * and spends the rest of its width on that sentence instead of on figures it has none
 * of. Ranked first and unranked after, because a candidate with no score has no place
 * in an order made of scores.
 */
function boardRows() {
  const ranked = board.rows.map((row) => ({
    cells: boardCells(row),
    attrs: { class: "row", "data-model": row.model_id, "data-quant": row.quant },
    classes: {
      verdict:
        row.candidate.placement && `verdict-${row.candidate.placement.budget.verdict}`,
    },
  }));
  const rest = board.excluded.map((row) => ({
    cells: boardCells(row),
    attrs: { class: "row out", "data-model": row.model_id, "data-quant": row.quant },
    span: { at: IDENTITY_COLUMNS, text: row.candidate.excluded_because || "", class: "why" },
  }));
  return ranked.concat(rest);
}

/** Draw the one list, and the captions saying what the figures in it are. */
function renderBoard() {
  const target = document.getElementById("board-table");
  const empty = document.getElementById("board-empty");
  empty.textContent = t("board.empty");
  empty.hidden = board.rows.length > 0;
  const rows = boardRows();
  if (!rows.length) {
    fill(target, el("p", { class: "muted", text: t("app.none") }));
  } else {
    drawnColumns = BOARD_COLUMNS.filter(
      (column) => column.name !== "confidence" || !oneConfidence(),
    );
    const columns = drawnColumns.map((column) => ({ ...column, heading: col(column.name) }));
    const drawn = table(columns, rows, { sorted: asked.sort });
    drawn.addEventListener("click", onBoardClick);
    fill(target, drawn);
  }
  renderCaptions();
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
  const found = board.rows
    .concat(board.excluded)
    .find(
      (candidate) =>
        candidate.model_id === row.dataset.model && candidate.quant === row.dataset.quant,
    );
  if (!found) return;
  const cell = el("td", { colspan: drawnColumns.length }, explanation(found));
  row.after(el("tr", { class: "detail" }, cell));
}

/** What the board no longer spends a column on, said once for the row it belongs to.
 *
 * How it runs and what it would cost to fetch were two of fourteen columns. They are
 * facts about one row, not figures to scan a board by, and this is where a reader is
 * already looking when they want them.
 */
function aboutBlock(row) {
  const placement = row.candidate.placement;
  const pairs = [
    [col("mode"), placement ? label("mode", placement.mode) : ""],
    [col("size"), bytes(row.download_bytes)],
  ];
  const parts = [facts(pairs)];
  if (row.local_path) {
    parts.push(el("p", { class: "muted", text: t("board.installed") }));
    parts.push(el("p", { class: "muted", text: row.local_path }));
  }
  return el("div", {}, parts);
}

/** One row expanded: what it is, the score with its parts, the speed, budget and ladder. */
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
  const extras = [aboutBlock(row), grid, planBlock(row)];
  if (candidate.placement && candidate.placement.notes.length) {
    extras.splice(
      2,
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

/** Planning one model, where the model is: inside its own row rather than on a screen
 * of its own that asks a person to pick it out of a list a second time.
 *
 * The two fields are the ones a plan takes that the request above the list does not.
 * Everything else -- the machine, or the machine being pretended at -- comes from what
 * the page is already showing, so the command that comes back is a command for the
 * board a person is looking at.
 */
function planBlock(row) {
  const output = el("div", {});
  const size = el("input", { type: "number", min: 1, step: 1024 });
  const batch = el("input", { type: "number", min: 1 });
  const field = (word, input) => el("label", {}, [el("span", { text: word }), input]);
  const act = el("button", { type: "button", text: t("board.plan_this") });
  act.addEventListener("click", async () => {
    const body = {
      model: row.model_id,
      quant: row.quant,
      vision: asked.vision,
      context: Number(size.value) || null,
      ub: Number(batch.value) || null,
    };
    for (const name of ["profile", "memory", "ram", "cpu_cores"]) {
      if (asked[name]) body[name] = name === "cpu_cores" ? Number(asked[name]) : asked[name];
    }
    const plan = await working(output, () =>
      api(`${API}/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
    if (plan) renderPlan(plan, output, row);
  });
  const controls = [field(t("plan.context"), size), field(t("plan.micro_batch"), batch), act];
  return el("div", { class: "plan" }, [el("div", { class: "actions" }, controls), output]);
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
  return el("div", { class: "wide" }, [
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
    el("div", { class: `finding ${finding.level === "ok" ? "" : finding.level}` }, [
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

/** The hardware profiles this machine has, for the machine-to-pretend list. */
async function loadProfiles() {
  const profiles = await api(`${API}/profiles`).catch(() => []);
  const select = document.getElementById("simulate-profile");
  fill(select, [
    el("option", { value: "", text: t("simulate.live") }),
    ...profiles.map((profile) => el("option", { value: profile.name, text: profile.name })),
  ]);
}

/** Everything `llamafit plan` prints that the row above does not already say.

 * A plan asked for at the board's own context repeats the row's speed, budget and
 * ladder exactly, so those are drawn only when the two fields above changed the
 * question and the answer is a different one. What is always new is the file, the
 * notes and the command line.
 */
function renderPlan(plan, target, row) {
  const fileLine = plan.model_present
    ? format(t("plan.file_here"), { path: plan.model_path })
    : format(t("plan.file_missing"), {
        path: plan.model_path,
        size: bytes(plan.download_bytes),
      });
  const planned = row.candidate.placement;
  const same = planned && planned.context === plan.placement.context;
  const parts = [el("h3", { text: t("panel.plan") }), el("p", { text: fileLine })];
  if (!same) {
    if (plan.speed) parts.push(speedBlock(plan.speed, plan.placement.context));
    parts.push(budgetBlock(plan.placement.budget));
  }
  if (plan.requested_budget) {
    parts.push(
      el("h3", {
        text: format(t("plan.requested_cost"), { context: context(plan.requested_context) }),
      }),
    );
    parts.push(budgetBlock(plan.requested_budget));
  }
  const tiers = same ? null : tiersBlock(plan.placement);
  if (tiers) parts.push(tiers);
  if (!same && plan.placement.notes.length) {
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
  fill(target, parts);
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
  // How a size may be written, kept off the controls row but still on the field.
  document.querySelector('[name="max_download"]').title = t("simulate.sizes");
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

/** Read the request off its controls and ask the server again.
 *
 * There is no Apply button. Every control re-asks the moment it changes, because the
 * question and its answer are on one screen now, and a person who can see both should
 * not have to tell the page to put them together.
 */
function readNeeds() {
  const form = document.getElementById("needs-form");
  asked.use_case = form.elements.use_case.value;
  asked.require = [...form.querySelectorAll('input[name="require"]:checked')].map(
    (input) => input.value,
  );
  asked.prefer = form.elements.prefer.value;
  asked.min_context = Number(form.elements.min_context.value) || 0;
  asked.max_download = form.elements.max_download.value.trim();
  asked.all_quants = form.elements.all_quants.checked;
  asked.vision = form.elements.vision.checked;
  asked.limit = Number(form.elements.limit.value) || 50;
  return loadBoard();
}

/** Wire every control up once, then ask the server for the first answer. */
function listen() {
  const needs = document.getElementById("needs-form");
  needs.addEventListener("change", readNeeds);
  needs.addEventListener("submit", (event) => {
    event.preventDefault();
    readNeeds();
  });
  // A reset restores the fields after the event, so the request is read on the next turn.
  needs.addEventListener("reset", () => setTimeout(readNeeds, 0));

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
    loadBoard();
  });

  document.getElementById("simulate-clear").addEventListener("click", () => {
    document.getElementById("simulate-form").reset();
    Object.assign(asked, { profile: "", memory: "", ram: "", cpu_cores: "" });
    document.getElementById("sim-badge").hidden = true;
    loadBoard();
  });

  document.getElementById("rescan").addEventListener("click", () => loadSystem(true));
}

/** Fetch the words, wire the page up, then ask for the machine and the board. */
async function boot() {
  ui = await api(`${API}/ui`);
  applyStrings();
  fillChoices();
  listen();
  loadCatalogProblems();
  loadProfiles();
  await loadSystem();
  // The first board is read off the controls rather than off the defaults beside them: a
  // browser may restore a form on reload, and a request the controls do not show is a
  // request nobody made.
  await readNeeds();
}

boot().catch((error) => {
  const banner = document.getElementById("failure");
  banner.textContent = String(error && error.message ? error.message : error);
  banner.hidden = false;
});
