// StructureMASST main page.
"use strict";

const { api, runJob, h, msg, md, spinner, jobProgress, jobOutcome, makeTable, tabs } = SMA;
const { StructureInput, LibraryOptions, RawOptions } = SearchUI;

// UI-only state kept per tab (Sankey choices and the like)
const UI_KEY = "structuremasst_ui";
const ui = Object.assign(
  { appendMode: false, sankeyOptions: {}, sankeyCols: {}, defaultSankeyVals: null },
  (() => { try { return JSON.parse(sessionStorage.getItem(UI_KEY) || "{}"); } catch (e) { return {}; } })(),
);
function saveUi() { try { sessionStorage.setItem(UI_KEY, JSON.stringify(ui)); } catch (e) { /* ignore */ } }

let state = null;
let input, libOpts, rawOpts;
const $ = (id) => document.getElementById(id);

function plot(el, fig, filename) {
  Plotly.newPlot(el, fig.data, fig.layout, { responsive: true, displaylogo: false, toImageButtonOptions: { format: "svg", filename } });
}

// ---------------------------------------------------------------------------
// Library search ("Get Available Spectra")
// ---------------------------------------------------------------------------
function renderAppendToggle() {
  const box = $("append-box");
  box.replaceChildren();
  if (state && state.groups.length) {
    box.append(h("label", { class: "check", title: "Molecule will be added to previous ones as additional query, rather than replacing previously requested molecules." },
      h("input", { type: "checkbox", checked: ui.appendMode, onchange: (e) => { ui.appendMode = e.target.checked; saveUi(); } }), "Add to existing"));
  }
}

async function getSpectra() {
  const box = $("library-status");
  if (!input.hasInput()) {
    box.replaceChildren(msg("warning", "Please enter a USI, SMILES/SMARTS, upload a CSV, or select a class label."));
    return;
  }
  const fd = new FormData();
  fd.append("search", JSON.stringify(Object.assign(input.input(), libOpts.values(), { append: ui.appendMode && !!(state && state.groups.length) })));
  if (input.file) fd.append("file", input.file);
  $("get-spectra").disabled = true;
  box.replaceChildren(spinner("Finding spectra..."));
  try {
    const v = await runJob("/api/search/library", fd, jobProgress(box, "Finding spectra..."));
    jobOutcome(box, v);
    if (v.status === "done") { ui.sankeyOptions = {}; saveUi(); }
    await loadState();
  } catch (e) {
    box.replaceChildren(msg("error", e.message));
  } finally {
    $("get-spectra").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Available library entries
// ---------------------------------------------------------------------------
async function tableAction(path, body, statusBox) {
  try {
    await api(path, { body });
    await loadState();
  } catch (e) {
    statusBox.replaceChildren(msg("warning", e.message));
  }
}

function renderLibrary() {
  const sec = $("library-section");
  sec.replaceChildren();
  if (!state || !state.groups.length) return;
  const body = h("div", { class: "body" }, h("h3", {}, "Available Library Entries"));
  sec.append(h("details", { class: "expander", open: true }, h("summary", {}, "Available Library Entries"), body));

  tabs(body, "library", state.groups.map((g, figId) => ({
    key: g.name, label: g.name,
    render: (tab) => {
      tab.append(h("h5", { html: md(`Retrieved **${g.total_spectra}** spectra for molecule (${g.name}).`) }));
      tab.append(h("div", { class: "small" }, "Below you see the public MS/MS spectra available for your search. The higher the spectral diversity, the less blind-spots you will have in public metabolomics raw data."));
      if (g.total_spectra === 0) { tab.append(msg("info", "No spectra available for this structure.")); return; }
      if (g.sankey) {
        const el = h("div", { class: "plot" });
        tab.append(el);
        plot(el, g.sankey, `plot_${figId}`);
      }

      // molecule overview: Compound_Name, Molecular_Formula, structure, others
      const ov = g.overview;
      const si = ov.columns.indexOf("Smiles");
      const front = ["Compound_Name", "Molecular_Formula"].filter((c) => ov.columns.includes(c));
      const rest = ov.columns.filter((c) => !front.includes(c));
      const cols = [...front, "structure", ...rest];
      const rows = ov.rows.map((r) => {
        const smi = si >= 0 ? r[si] : "";
        const img = smi ? `/api/render?smiles=${encodeURIComponent(smi)}&size=500` : null;
        return [...front.map((c) => r[ov.columns.indexOf(c)]), img, ...rest.map((c) => r[ov.columns.indexOf(c)])];
      });
      const status = h("div");
      const molTable = makeTable(tab, { columns: cols, rows }, {
        selectable: true, images: { structure: { width: 240, height: 240 } }, rowHeight: 250,
        height: rows.length > 2 ? 620 : false, name: `${g.name}_molecules`,
      });
      const molAction = (action) => tableAction("/api/library/molecules", { name: g.name, action, rows: molTable.selected() }, status);
      const btns = h("div", { class: "btn-row" },
        h("button", { onclick: () => molAction("remove") }, "Remove selected molecule(s)"),
        h("button", { onclick: () => molAction("keep") }, "Keep only selected molecule(s)"));
      if (ov.rows.length > 1) btns.append(h("button", { style: "margin-left:auto", onclick: () => molAction("split") }, "Split each molecule into separate query"));
      tab.append(btns, status);

      const ikBody = h("div", { class: "body" });
      tab.append(h("details", { class: "expander", open: true }, h("summary", {}, "Molecules by InChIKey"), ikBody));
      let spectra = g.spectra;
      if (spectra.length > 100) {
        ikBody.append(msg("warning", `⚠️ Too many InChIKeys (${spectra.length}). Showing only the first 100.`));
        spectra = spectra.slice(0, 100);
      }
      tabs(ikBody, `ik_${g.name}`, spectra.map((sp) => ({
        key: sp.ik, label: sp.ik,
        render: (ikTab) => {
          const st = h("div");
          const t = makeTable(ikTab, sp.table, {
            selectable: true, links: { spectrum_link: "Open Spectrum" }, titles: { spectrum_link: "USI Viewer" }, name: `${g.name}_${sp.ik}`,
          });
          const act = (action) => tableAction("/api/library/spectra", { name: g.name, ik: sp.ik, action, rows: t.selected() }, st);
          ikTab.append(h("div", { class: "btn-row" },
            h("button", { onclick: () => act("remove") }, "Remove selected spectra"),
            h("button", { onclick: () => act("keep") }, "Keep only selected spectra")), st);
        },
      })));
    },
  })));
}

// ---------------------------------------------------------------------------
// Raw data search panel
// ---------------------------------------------------------------------------
// Created once: the status box and running flag outlive re-renders of the panel,
// so the outcome of a search stays visible after the page reloads its state.
const rawPanel = h("div");
const rawStatus = h("div");
let rawRunning = false;

function renderRawPanel() {
  const sec = $("raw-section");
  if (!state || !state.groups.length) { sec.replaceChildren(); return; }
  const nSpectra = state.groups.reduce((a, g) => a + g.spectra.reduce((b, s) => b + s.table.rows.length, 0), 0);
  if (!rawOpts) rawOpts = new RawOptions(rawPanel, "main", { cards: true, analogues: true });
  rawOpts.setForceFasst(!!state.direct_spectra);
  const btn = h("button", { type: "button", disabled: rawRunning, onclick: () => searchRaw() }, "🔍 Search Raw Data");
  sec.replaceChildren(
    h("h5", {}, `Next, search all ${nSpectra} spectra of ${state.groups.length} molecule(s) across public metabolomics raw data to retrieve matching samples:`),
    rawPanel, h("div", { class: "btn-row" }, btn), rawStatus);
}

async function searchRaw() {
  const setRunning = (v) => {
    rawRunning = v;
    const b = document.querySelector("#raw-section > .btn-row button");
    if (b) b.disabled = v;
  };
  setRunning(true);
  rawStatus.replaceChildren(spinner("Searching raw data…"));
  try {
    const v = await runJob("/api/search/raw", rawOpts.params(), jobProgress(rawStatus, "Searching raw data…"));
    jobOutcome(rawStatus, v);
    if (v.status === "done") await loadState();
  } catch (e) {
    rawStatus.replaceChildren(msg("error", e.message));
  } finally {
    setRunning(false);
  }
}

// ---------------------------------------------------------------------------
// Raw data results
// ---------------------------------------------------------------------------
const REMOVE_COLS = ["best_spectral_match", "lib_usi", "USI", "scan_id", "spectrum_id_int", "representative_spectrum_int",
  "spectrum_difference", "unique_spectra_in_mri", "Check LC peak", "mri", "mri_id_int"];

const PRESETS = [
  [["ATTRIBUTE_DatasetAccession", "UBERONBodyPartName", "NCBIDivision", "NCBITaxonomy"],
    "Understand biological distributions at a glance while not losing sight of the number of datasets confirming an observed pattern."],
  [["similar_library_spectra", "query_spectrum_id", "NCBIDivision", "NCBITaxonomy"],
    "Explore if matches to suspicious sample types are driven by different <em>query_spectrum_ids</em> than those for expected sample types."],
  [["inchikey_first_block", "Compound_Name", "UBERONBodyPartName", "NCBITaxonomy"],
    "After substructure or Tanimoto similarity searches, investigate if molecules differ by sample types."],
  [["Delta Mass", "Modified", "UBERONBodyPartName", "DOIDCommonName"],
    "If modifications were enabled, explore if specific modifications differ by body part and disease (especially interesting for drugs)."],
];

function downstreamButton(label, path, body, opts = {}) {
  const status = h("div");
  const btn = h("button", {
    type: "button", disabled: opts.disabled,
    onclick: async () => {
      btn.disabled = true;
      status.replaceChildren(spinner(opts.spin || "Running…"));
      try {
        const v = await runJob(path, body, jobProgress(status, opts.spin));
        if (jobOutcome(status, v) && opts.link) status.append(h("p", {}, h("a", { href: opts.link[0] }, opts.link[1])));
      } catch (e) {
        status.replaceChildren(msg("error", e.message));
      } finally {
        btn.disabled = false;
      }
    },
  }, label);
  return h("div", { style: "max-width:520px" }, h("div", { class: "btn-row" }, btn), status);
}

function renderResultTab(tab, name, table, resultFigId, multi) {
  const n = table.rows.length;
  tab.append(h("p", { html: md(`Retrieved **${n}** matching samples with ReDU metadata for **${name}**.`) }));
  if (!(table.columns.includes("mri_id_int") || table.columns.includes("mri")) || n === 0) {
    tab.append(msg("warning", "No ReDU metadata matches found."));
    return;
  }

  const newCols = table.columns.filter((c) => !REMOVE_COLS.includes(c));
  const options = [...new Set([...(ui.sankeyOptions[name] || []), ...newCols])];
  ui.sankeyOptions[name] = options;
  const defVal = (v) => (options.includes(v) ? v : options[0]);
  if (!ui.defaultSankeyVals) ui.defaultSankeyVals = PRESETS[0][0];
  ui.sankeyCols[name] = (ui.sankeyCols[name] || ui.defaultSankeyVals).map(defVal);
  saveUi();

  tab.append(h("h5", {}, "Suggested views"));
  const presetBox = h("div", { class: "presetbox" });
  const selects = [];
  const plotEl = h("div", { class: "plot" });
  const plotMsg = h("div");
  const drawSankey = async () => {
    const cols = ui.sankeyCols[name];
    plotMsg.replaceChildren();
    if (new Set(cols).size !== 4) {
      Plotly.purge(plotEl);
      plotMsg.append(msg("warning", "Warning: Some selected columns have the same value."));
      return;
    }
    try {
      plot(plotEl, await api("/api/results/sankey", { body: { name, cols } }), `plot_${resultFigId}`);
    } catch (e) { plotMsg.append(msg("error", e.message)); }
  };
  for (const [vals, sub] of PRESETS) {
    presetBox.append(h("div", { class: "preset-row" },
      h("div", { class: "preset-text" }, h("span", { class: "preset-title" }, vals.join(" → ")), h("span", { class: "preset-sub", html: sub })),
      h("button", { type: "button", onclick: () => {
        ui.sankeyCols[name] = vals.map(defVal); saveUi();
        selects.forEach((s, i) => { s.value = ui.sankeyCols[name][i]; });
        drawSankey();
      } }, "Apply")));
  }
  tab.append(presetBox);
  const selRow = h("div", { class: "row" });
  for (let i = 0; i < 4; i++) {
    const s = h("select", { onchange: () => { ui.sankeyCols[name][i] = s.value; saveUi(); drawSankey(); } },
      options.map((o) => h("option", { value: o, selected: ui.sankeyCols[name][i] === o }, o)));
    selects.push(s);
    selRow.append(h("div", { class: "col" }, h("label", { class: "field" }, `Column ${i + 1}`), s));
  }
  tab.append(selRow, plotMsg, plotEl);
  drawSankey();

  const t = makeTable(tab, table, {
    selectable: true, name: `${name}_raw_matches`,
    links: { best_spectral_match: "View MS/MS match", modification_site: "View Modification Site", "Check LC peak": "View LC peak" },
  });

  // selection by column value(s); the values come from the server so they match its rules
  const colSel = h("select", {}, ["None", ...table.columns].map((c) => h("option", { value: c }, c)));
  const valSel = h("select", { multiple: true, disabled: true });
  const applyAll = h("input", { type: "checkbox", disabled: true });
  const updateApplyAll = () => {
    const use = colSel.value !== "None" && valSel.selectedOptions.length > 0;
    applyAll.disabled = !use;
    if (!use) applyAll.checked = false;
  };
  colSel.addEventListener("change", async () => {
    valSel.replaceChildren();
    valSel.disabled = true;
    if (colSel.value !== "None") {
      const r = await api(`/api/results/values?name=${encodeURIComponent(name)}&col=${encodeURIComponent(colSel.value)}`);
      valSel.replaceChildren(...r.values.map((v) => h("option", { value: v }, v)));
      valSel.disabled = false;
    }
    updateApplyAll();
  });
  valSel.addEventListener("change", updateApplyAll);
  tab.append(h("div", { class: "row" },
    h("div", { class: "col", style: "flex:2" }, h("label", { class: "field" }, "Column"), colSel),
    h("div", { class: "col", style: "flex:3" }, h("label", { class: "field" }, "Value(s)"), valSel),
    h("div", { class: "col", style: "flex:5" })));
  const st = h("div");
  const act = (action) => {
    const vals = [...valSel.selectedOptions].map((o) => o.value);
    const byValue = colSel.value !== "None" && vals.length > 0;
    tableAction("/api/results/rows", {
      name, action, rows: byValue ? [] : t.selected(),
      column: byValue ? colSel.value : null, values: byValue ? vals : [], apply_all: byValue && applyAll.checked,
    }, st);
  };
  const actions = h("div", { class: "btn-row" },
    h("button", { onclick: () => act("remove") }, "Remove selected rows"),
    h("button", { onclick: () => act("keep") }, "Keep only selected rows"));
  // with a single molecule there is nothing else to apply the filter to
  if (multi) {
    actions.append(h("label", { class: "check", title: "Applies the column/value filter above to every molecule, not just this one. Only available when a column + value(s) are selected." },
      applyAll, "Apply to all molecules"));
  }
  tab.append(actions, st);

  tab.append(h("h5", { html: md(`Downstream tooling for **${name}**`) }));
  tab.append(downstreamButton(`Populate DomainMASST for ${name}`, "/api/domainmasst", { name }, {
    disabled: n === 0, spin: "Running DomainMASSTs…", link: ["/domainmasst", "➡️ Click for DomainMASST Results"],
  }));
  tab.append(downstreamButton("Setup LifeMASST", "/api/lifemasst/setup", { name }, {
    spin: "Setting up LifeMASSTs…", link: ["/lifemasst", "➡️ Click for LifeMASST Workspace"],
  }));
}

function renderResults() {
  const sec = $("results-section");
  sec.replaceChildren();
  if (!state || !state.raw_results || !state.groups.length) return;
  const results = state.raw_results;
  if (!results.some((r) => r.table.rows.length > 0)) {
    sec.append(msg("warning", "No raw data matches found. Please try a different query structure or adjust your search parameters."));
    return;
  }
  const body = h("div", { class: "body" }, h("h3", {}, "Metabolomics Raw Data Matches"));
  sec.append(h("details", { class: "expander", open: true }, h("summary", {}, "Metabolomics Raw Data Matches"), body));
  tabs(body, "results", results.map((r, i) => ({ key: r.name, label: r.name, render: (tab) => renderResultTab(tab, r.name, r.table, i, results.length > 1) })));

  if (results.length > 1 && results.some((r) => r.table.rows.length > 1)) {
    body.append(h("h5", {}, "Downstream tooling for all molecules"));
    if (results.every((r) => r.table.rows.length > 1)) {
      body.append(downstreamButton("Populate DomainMASST with Molecule Co-occurrence", "/api/domainmasst", {}, {
        spin: "Running DomainMASSTs…", link: ["/domainmasst", "➡️ Click for DomainMASST Results"],
      }));
    }
    body.append(downstreamButton("Setup LifeMASST", "/api/lifemasst/setup", {}, {
      spin: "Setting up LifeMASSTs…", link: ["/lifemasst", "➡️ Click for LifeMASST Workspace"],
    }));
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function loadState() {
  state = await api("/api/state");
  renderAppendToggle();
  renderLibrary();
  renderRawPanel();
  renderResults();
}

(async function init() {
  SMA.shell("/");
  await SMA.ensureSession();
  $("sid").textContent = SMA.sid;
  input = new StructureInput($("search-input"), "main", { mgf: true, batchBelowName: true });
  libOpts = new LibraryOptions($("search-options"), "main", input);
  $("get-spectra").addEventListener("click", getSpectra);
  try { await loadState(); } catch (e) { $("library-status").replaceChildren(msg("error", e.message)); }
})();
