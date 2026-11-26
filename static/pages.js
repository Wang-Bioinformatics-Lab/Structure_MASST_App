// DomainMASST, LifeMASST and GeoMASST pages.
"use strict";

const { api, runJob, h, msg, spinner, jobProgress, jobOutcome, makeTable, fileUrl } = SMA;
const { StructureInput, LibraryOptions, RawOptions, logProgress, showLog, prepPanel } = SearchUI;

// Reports are built from user input: run them without access to this app (the
// server also serves them with a sandbox CSP). Full screen is a separate
// permission (allowfullscreen), not a sandbox flag; the GeoMASST map uses it.
const SANDBOX = "allow-scripts allow-popups allow-popups-to-escape-sandbox allow-downloads";

function header(main, title, text) {
  main.append(h("h1", {}, title));
  if (text) main.append(h("p", {}, text));
}

// An iframe that follows the height its document asks for (the GeoMASST map and
// the LifeMASST tree heatmap post Streamlit's setFrameHeight message).
function autoFrame(src, height, title) {
  const frame = h("iframe", { class: "report-frame", src, title, style: `height:${height}px`, sandbox: SANDBOX, allowfullscreen: true });
  window.addEventListener("message", (e) => {
    const d = e.data;
    if (e.source === frame.contentWindow && d && d.type === "streamlit:setFrameHeight" && d.height) frame.style.height = `${Math.ceil(d.height)}px`;
  });
  return frame;
}

// The shared search controls plus their "Advanced search options".
function searchControls(container, prefix) {
  const input = new StructureInput(container, prefix, { mgf: false });
  const advBody = h("div", { class: "body" });
  container.append(h("details", { class: "expander" }, h("summary", {}, "Advanced search options"), advBody));
  const lib = new LibraryOptions(advBody, prefix, input, { label: "Library search type" });
  const raw = new RawOptions(advBody, prefix, {});
  return {
    input,
    hasInput: () => input.hasInput(),
    search: () => ({ input: Object.assign(input.input(), lib.values()), raw: raw.params() }),
    file: () => input.file,
  };
}

// ---------------------------------------------------------------------------
// DomainMASST
// ---------------------------------------------------------------------------
async function domainPage(main) {
  header(main, "Domain MASST (preview)",
    "This page lets you further explore your StructureMASST results in the context of different DomainMASST analyses. " +
    "You can compare and contextualize findings across plants, microbes, human and mouse tissues, foods, and other domains of interest.");
  const { files } = await api("/api/domainmasst/files");
  if (!files.length) {
    main.append(msg("warning", "This is a downstream tool. Run StructureMASST first and press the populate DomainMASST button to generate DomainMASST results."));
    return;
  }
  const sel = h("select", {}, files.map((f) => h("option", { value: f }, f)));
  const chosen = h("p");
  const frame = h("iframe", { class: "report-frame", title: "DomainMASST result", sandbox: SANDBOX, allowfullscreen: true });
  const show = () => { chosen.textContent = `You selected: ${sel.value}`; frame.src = fileUrl(sel.value); };
  sel.addEventListener("change", show);
  main.append(h("div", { style: "max-width:520px" }, h("label", { class: "field" }, "Choose a file"), sel), chosen, frame);
  show();
}

// ---------------------------------------------------------------------------
// LifeMASST
// ---------------------------------------------------------------------------
const TREES = [
  ["trees/OTL_prepared/labelled_supertree_subset_prepped", "Open Tree of Life",
    "Open Tree of Life (subset with available metabolomics data)<br/>→ Subsetted comprehensive tree of life from Open Tree of Life, based on a synthesis of published phylogenies and taxonomies.<br/>Source: OpenTree et al. Open Tree of Life Synthetic Tree https://doi.org/10.5281/zenodo.3937741"],
  ["trees/timetree_prepared/TimeTree_subset_prepped", "TimeTree",
    "TimeTree<br/>→ Time-calibrated tree of life from TimeTree database, based on published divergence time estimates.<br/>Source: Kumar et al. (2022), <i>Molecular Biology and Evolution</i> 39(8). https://doi.org/10.1093/molbev/msac174"],
  ["trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint", "Bird Tree (time tree)",
    "Rooted timetree from Kimball et al. (2019)<br/>→ Branch lengths represent absolute divergence times estimated with fossil calibrations.<br/>Source: Kimball et al. (2019), <i>Diversity</i> 11, 109. https://doi.org/10.3390/d11070109"],
  ["trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa", "Bird Tree (molecular tree)",
    "CYB+ND2 supertree from Kimball et al. (2019)<br/>→ Branch lengths represent molecular substitution distances based on mitochondrial gene data.<br/>Source: Kimball et al. (2019), <i>Diversity</i> 11, 109. https://doi.org/10.3390/d11070109"],
  ["trees/mammalian_prepared/Foley2022_Concatenation_HRA_neutral_241_10miss_rooted", "Mammal Tree (molecular tree)",
    "Concatenated phylogeny from Foley et al. (2023)<br/>→ Branch lengths represent molecular substitution distances based on genome-wide nearly neutral sites.<br/>Source: Foley et al. (2023), <i>Science</i> 380, eabl8189. https://doi.org/10.1126/science.abl8189"],
  ["trees/fish_prepared/fish_raxml_12k", "Fish Tree (molecular tree)",
    "Molecular phylogram (Actinopterygii 12k; RAxML)<br/>→ Branch lengths represent molecular substitution distances (substitutions per site) from maximum-likelihood inference.<br/>Source: Rabosky et al. (2018), <i>Nature</i> 559, 392–395. https://doi.org/10.1038/s41586-018-0273-1"],
  ["trees/fish_prepared/fish_treepl_12k", "Fish Tree (time tree)",
    "Time-calibrated tree (Actinopterygii 12k; treePL)<br/>→ Branch lengths represent absolute divergence times (millions of years) estimated via fossil-calibrated treePL.<br/>Source: Rabosky et al. (2018), <i>Nature</i> 559, 392–395. https://doi.org/10.1038/s41586-018-0273-1"],
  ["trees/plants_prepared/1kp_astral_alltaxa_FAA", "Plant Tree (molecular tree)",
    "Molecular phylogeny from One Thousand Plant Transcriptomes Initiative (1KP, 2019)<br/>→ Branch lengths represent molecular substitution distances inferred from large-scale transcriptome data.<br/>Source: One Thousand Plant Transcriptomes Initiative (2019), <i>Nature</i> 574, 679–685. https://doi.org/10.1038/s41586-019-1693-2"],
  ["upload", "Custom Uploaded Tree", ""],
];

const MIN_SPECIFICITY = [
  ["", "No minimum (all flexible matches)"], ["species", "Species or finer"], ["genus", "Genus or finer"],
  ["family", "Family or finer"], ["order", "Order or finer"], ["class", "Class or finer"], ["phylum", "Phylum or finer"],
];

async function lifePage(main) {
  header(main, "LifeMASST", null);
  const status = await api("/api/lifemasst/status");

  let controls;
  let fresh = null; // the "start fresh" expander when input came from StructureMASST
  if (status.prepared_from_structuremasst) {
    main.append(msg("info", "LifeMASST input has already been prepared from StructureMASST for this session. Simply select a tree from the dropdown and hit Run LifeMASST!"));
    const body = h("div", { class: "body" });
    fresh = h("details", { class: "expander" }, h("summary", {}, "Discard StructureMASST results and start fresh."), body);
    main.append(fresh);
    controls = searchControls(body, "lm");
  } else {
    controls = searchControls(main, "lm");
  }

  const treeSel = h("select", {}, TREES.map(([k, l]) => h("option", { value: k }, l)));
  const specSel = h("select", {}, MIN_SPECIFICITY.map(([k, l]) => h("option", { value: k }, l)));
  const message = h("div");
  const rest = h("div");
  main.append(
    h("div", { class: "row", style: "margin-top:1rem" },
      h("div", { class: "col" }, h("label", { class: "field", title: "Select the phylogenetic or taxonomic tree to use for LifeMASST analysis." }, "Select a tree for LifeMASST"), treeSel),
      h("div", { class: "col" }), h("div", { class: "col" }), h("div", { class: "col" })),
    message, rest);

  const runBtn = h("button", { type: "button" }, "Run LifeMASST");
  const runStatus = h("div");
  const results = h("div");
  const restBody = h("div");
  rest.append(
    h("div", { class: "row" },
      h("div", { class: "col" }, h("label", { class: "field", title: "Only show tree nodes where the StructureMASST flexible match is at least this specific. 'No minimum' includes all matches, even kingdom-level propagated hits." }, "Minimum StructureMASST match specificity"), specSel),
      h("div", { class: "col" }), h("div", { class: "col" }), h("div", { class: "col" })),
    restBody, h("div", { class: "btn-row" }, runBtn), runStatus, results);
  await prepPanel(restBody, "lifemasst");

  const update = () => {
    const t = TREES.find((x) => x[0] === treeSel.value);
    message.replaceChildren(t[2] ? h("div", { class: "lifebox", html: t[2] }) : "");
    if (treeSel.value === "upload") {
      message.append(msg("warning", "Custom uploaded tree is still not implemented."));
      rest.classList.add("hidden");
    } else {
      rest.classList.remove("hidden");
    }
  };
  treeSel.addEventListener("change", update);
  update();

  const showResults = (st) => {
    results.replaceChildren();
    if (st.tree_heatmap) results.append(autoFrame(fileUrl("lifemasst/tree_heatmap.html") + `&t=${Date.now()}`, 900, "LifeMASST tree heatmap"));
    if (st.metadata) {
      results.append(h("h3", {}, "Merged Metadata"));
      makeTable(results, st.metadata, { name: "merged_metadata" });
      results.append(msg("success", "Download an interactive LifeMASST plot below."));
      results.append(st.empress_zip
        ? h("a", { class: "btn", href: fileUrl("lifemasst/empress_results.zip"), download: "empress_results.zip" }, "📥 Download Empress Results")
        : h("button", { disabled: true }, "📥 Download Empress Results"));
    }
  };
  showResults(status);

  runBtn.addEventListener("click", async () => {
    // use the controls on this page when they hold an input (when the input came
    // from StructureMASST, only if "start fresh" is open)
    const useSearch = controls.hasInput() && (!fresh || fresh.open);
    const fd = new FormData();
    fd.append("params", JSON.stringify({ tree: treeSel.value, min_specificity: specSel.value, search: useSearch ? controls.search() : null }));
    if (useSearch && controls.file()) fd.append("file", controls.file());
    runBtn.disabled = true;
    try {
      const v = await runJob("/api/lifemasst/run", fd, logProgress(runStatus, "Running LifeMASST…"));
      if (v.status === "error") { jobOutcome(runStatus, v); showLog(runStatus, v); } else {
        runStatus.replaceChildren(...(v.warnings || []).map((w) => msg("warning", w)));
        showLog(runStatus, v);
      }
      showResults(await api("/api/lifemasst/status"));
    } catch (e) {
      runStatus.replaceChildren(msg("error", e.message));
    } finally {
      runBtn.disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------
// GeoMASST
// ---------------------------------------------------------------------------
const ALL = "All molecules";

async function geoPage(main) {
  header(main, "GeoMASST (preview)",
    "This page maps where a molecule was found. Use the StructureMASST results from this session, or search here directly.");
  const state = await api("/api/state");
  const names = (state.raw_results || []).map((r) => r.name);
  const hasResults = names.length > 0;
  const SRC_RESULTS = "This session's StructureMASST results";
  const SRC_SEARCH = "Search from here";
  const sources = hasResults ? [SRC_RESULTS, SRC_SEARCH] : [SRC_SEARCH];
  let source = sources[0];

  const srcBox = h("div", { class: "radio-group" });
  main.append(h("div", {}, h("label", { class: "field" }, "Matches to map"), srcBox));
  const resultsBox = h("div");
  const searchBox = h("div", { class: "hidden" });
  main.append(resultsBox, searchBox);

  const slider = (label, help) => {
    const out = h("span", { class: "muted" }, "5");
    const s = h("input", { type: "range", min: 1, max: 20, value: 5, style: "width:100%", oninput: () => { out.textContent = s.value; } });
    const wrap = h("div", { style: "max-width:520px;margin-top:0.5rem" }, h("label", { class: "field", title: help }, label, " ", out), s);
    return { wrap, value: () => parseInt(s.value, 10) };
  };

  // results source
  const options = names.length > 1 ? [ALL, ...names] : names;
  const nameSel = h("select", {}, options.map((n) => h("option", { value: n }, n)));
  const resSlider = slider("Maximum separate molecule maps",
    "One map shows every molecule at once, sized by how many were matched at each site. Pick molecules in the side list to split them out; this caps how many get their own map.");
  if (hasResults) {
    resultsBox.append(h("div", { style: "max-width:520px" }, h("label", { class: "field" }, "Select chemical"), nameSel), resSlider.wrap);
    const syncSlider = () => resSlider.wrap.classList.toggle("hidden", nameSel.value !== ALL);
    nameSel.addEventListener("change", syncSlider);
    syncSlider();
  }

  // search source
  if (!hasResults) searchBox.append(h("p", { class: "caption" }, "No StructureMASST results in this session - search here instead."));
  const controls = searchControls(searchBox, "geo");
  const searchSlider = slider("Maximum separate molecule maps",
    "Only matters for a batch file. One map shows every molecule at once; this caps how many also get their own.");
  searchBox.append(searchSlider.wrap);

  const setSource = (s) => {
    source = s;
    resultsBox.classList.toggle("hidden", s !== SRC_RESULTS);
    searchBox.classList.toggle("hidden", s !== SRC_SEARCH);
  };
  srcBox.append(...sources.map((s) => h("label", {}, h("input", { type: "radio", name: "geo_source", value: s, checked: s === source, onchange: () => setSource(s) }), s)));
  setSource(source);

  await prepPanel(main, "geomasst");
  const btn = h("button", { type: "button" }, "Run GeoMASST");
  const status = h("div");
  const mapBox = h("div");
  main.append(h("div", { class: "btn-row" }, btn), status, mapBox);

  btn.addEventListener("click", async () => {
    let params;
    const fd = new FormData();
    if (source === SRC_SEARCH) {
      if (!controls.hasInput()) { status.replaceChildren(msg("warning", "Enter a name, SMILES/SMARTS, class or USI to search for.")); return; }
      params = { source: "search", max_maps: searchSlider.value(), search: controls.search() };
      if (controls.file()) fd.append("file", controls.file());
    } else {
      params = { source: "results", name: nameSel.value, max_maps: resSlider.value() };
    }
    fd.append("params", JSON.stringify(params));
    btn.disabled = true;
    mapBox.replaceChildren();
    try {
      const v = await runJob("/api/geomasst", fd, logProgress(status, "Running GeoMASST…"));
      if (jobOutcome(status, v) && v.result) {
        if (v.result.caption) status.append(h("p", { class: "caption" }, v.result.caption));
        mapBox.append(autoFrame(fileUrl(v.result.file) + `&t=${Date.now()}`, 1100, "GeoMASST map"));
      }
      showLog(status, v);
    } catch (e) {
      status.replaceChildren(msg("error", e.message));
    } finally {
      btn.disabled = false;
    }
  });
}

(async function init() {
  const page = document.body.dataset.page;
  SMA.shell("/" + page);
  await SMA.ensureSession();
  const main = document.getElementById("main");
  try {
    if (page === "domainmasst") await domainPage(main);
    else if (page === "lifemasst") await lifePage(main);
    else if (page === "geomasst") await geoPage(main);
  } catch (e) {
    main.append(msg("error", e.message));
  }
})();
