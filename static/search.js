// The StructureMASST search controls, shared by the StructureMASST, LifeMASST
// and GeoMASST pages (formerly structureMASST.py + bin/streamlit_search_ui.py).
// Every control keeps its state under a page prefix, so pages do not collide.
"use strict";

const SearchUI = (() => {
  const { api, h, msg, spinner, runJob, jobOutcome, makeTable } = SMA;

  function loadState(key, defaults) {
    let saved = {};
    try { saved = JSON.parse(sessionStorage.getItem(key) || "{}"); } catch (e) { /* ignore */ }
    return Object.assign({}, defaults, saved);
  }
  function saveState(key, st) {
    try { sessionStorage.setItem(key, JSON.stringify(st)); } catch (e) { /* ignore */ }
    SMA.syncUrl(); // the address bar mirrors the search settings
  }

  // Whether a link's settings are what this tab already has. Then the tab keeps
  // its state as it is (name suggestions, an open editor), so a refresh, whose
  // address mirrors the form, changes nothing.
  function sameSettings(a, b) {
    const norm = (o) => JSON.stringify(Object.entries(o || {})
      .filter(([, v]) => v !== null && v !== undefined && String(v) !== "")
      .map(([k, v]) => [k, String(v)]).sort());
    return norm(a) === norm(b);
  }

  let classesPromise = null;
  function classes() {
    if (!classesPromise) classesPromise = api("/api/classes").then((r) => r.classes || []).catch(() => []);
    return classesPromise;
  }

  // -------------------------------------------------------------------------
  // Name | SMILES/SMARTS | Class label | USI, batch file, structure preview/editor
  // -------------------------------------------------------------------------
  class StructureInput {
    constructor(container, prefix, opts = {}) {
      this.prefix = prefix;
      this.opts = Object.assign({ mgf: false, batchBelowName: false }, opts);
      this.key = `sm_input_${prefix}`;
      this.st = loadState(this.key, {
        nameQuery: "", lastFetched: null, suggestions: [], nameChoice: null, nameWarning: null,
        smiles: "", newSmiles: "", editorOpen: false, classLabel: "", usi: "",
      });
      this.file = null;
      this.type = null; // smiles | smarts | Invalid | class_label | usi | null
      this.listeners = [];
      const link = SMA.linkParams("input");
      if (link && !sameSettings(link, this.urlParams())) this.fromLink(link);
      this.render(container);
      SMA.trackUrl("input", () => this.urlParams());
      if (this.lookUpName) this.onName(); // a link with only a name: look it up as if typed
    }

    save() { saveState(this.key, this.st); }

    // A link's ?usi= / ?smiles= (with ?name= when it came from a name search) /
    // ?class= / ?name= replaces this tab's input.
    fromLink(p) {
      if (!p) return;
      const v = (k) => String(p[k] || "").trim();
      Object.assign(this.st, { nameQuery: "", lastFetched: null, suggestions: [], nameChoice: null, nameWarning: null,
        smiles: "", newSmiles: "", editorOpen: false, classLabel: "", usi: "" });
      if (v("usi")) this.st.usi = v("usi");
      else if (v("smiles")) Object.assign(this.st, { smiles: v("smiles"), nameQuery: v("name") });
      else if (v("class")) this.st.classLabel = v("class");
      else if (v("name")) { this.st.nameQuery = v("name"); this.lookUpName = true; }
      this.save();
    }

    urlParams() {
      const s = this.st;
      if (s.usi) return { usi: s.usi };
      if (s.classLabel) return { class: s.classLabel };
      return { name: s.nameQuery, smiles: s.smiles.trim() ? this.effective() : "" };
    }
    onChange(fn) { this.listeners.push(fn); }
    emit() { this.listeners.forEach((fn) => fn(this.type)); }
    effective() { return (this.st.newSmiles || this.st.smiles || "").trim(); }

    // The search input sent to the server (see workflows::SearchInput).
    input() {
      return {
        usi: this.st.usi.trim(),
        smiles: this.st.smiles.trim() ? this.effective() : "",
        class_label: this.st.classLabel,
        name_query: this.st.nameQuery,
      };
    }
    hasInput() {
      const i = this.input();
      return !!(i.usi || i.smiles || i.class_label || i.name_query.trim() || this.file);
    }

    render(container) {
      const p = this.prefix;
      const id = (s) => `${p}${s}`;
      const nameInput = h("input", { type: "text", id: id("name"), placeholder: "e.g., diazepam, caffeine, etc.", autocomplete: "off" });
      const nameSuggest = h("div", { class: "hidden", style: "margin-top:0.5rem" },
        h("label", { class: "field" }, "Suggestions"), h("select", { id: id("choice") }));
      const nameStatus = h("div");
      const smilesInput = h("input", { type: "text", id: id("smiles"), placeholder: "Enter SMILES or SMARTS", autocomplete: "off", spellcheck: "false" });
      const classSel = h("select", { id: id("class") }, h("option", { value: "" }, ""));
      const usiInput = h("input", { type: "text", id: id("usi"), placeholder: "mzspec:...", autocomplete: "off", spellcheck: "false" });
      const fileInput = h("input", { type: "file", accept: this.opts.mgf ? ".csv,.mgf,text/csv" : ".csv,text/csv" });
      const fileName = h("div", { class: "small muted", style: "margin-top:0.4rem" });
      const pop = h("div", { class: this.opts.batchBelowName ? "popover-body left hidden" : "popover-body hidden" },
        h("p", { class: "small", style: "margin-top:0" }, this.opts.mgf
          ? "Drop a CSV (name + query/smiles columns) or MGF file for batch search. We recommend staying below 100 molecules/spectra or reach out to us to run locally."
          : "Drop CSV file for batch search (query/name or smiles/name)."),
        fileInput,
        h("div", { class: "btn-row" }, h("button", { type: "button", onclick: () => { this.file = null; fileInput.value = ""; fileName.textContent = ""; } }, "Remove file")));
      const batch = h("div", { style: this.opts.batchBelowName ? "margin-top:0.8rem" : "" },
        h("div", { class: "popover" }, h("button", { type: "button", onclick: () => pop.classList.toggle("hidden") }, "⬆ Add batch file"), pop),
        fileName);
      document.addEventListener("click", (e) => { if (!e.target.closest(".popover")) pop.classList.add("hidden"); });
      fileInput.addEventListener("change", () => {
        this.file = fileInput.files[0] || null;
        fileName.textContent = this.file ? `Batch file: ${this.file.name}` : "";
        if (this.file) pop.classList.add("hidden"); // do not leave it covering the search button
      });

      const col = (...c) => h("div", {}, ...c);
      const or = () => h("div", { class: "or" }, "or");
      const help = (t) => h("span", { class: "help", title: t }, " ⓘ");
      container.append(h("div", { class: "input-row input-row-4" },
        col(h("label", { class: "field", for: id("name") }, "Type a chemical name to search PubChem"), nameInput, nameSuggest, nameStatus,
            this.opts.batchBelowName ? batch : ""),
        or(),
        col(h("label", { class: "field", for: id("smiles") }, "SMILES/SMARTS",
              help("Enter a valid SMILES or SMARTS string. For SMARTS string creation, you can use third-party tools like SMARTSPlus https://smarts.plus/create")),
            smilesInput),
        or(),
        col(h("label", { class: "field", for: id("class") }, "Class Label",
              help("Select a (Classyfire) molecule class to find spectra for all molecules in this class. This is under development and not comprehensive yet.")),
            classSel),
        or(),
        col(h("label", { class: "field", for: id("usi") }, "USI", help("Provide a USI to directly search this spectrum. No structure is required.")), usiInput)));
      if (!this.opts.batchBelowName) container.append(batch);
      this.preview = h("div");
      container.append(this.preview);

      this.el = { nameInput, nameSuggest, nameStatus, smilesInput, classSel, usiInput };
      nameInput.addEventListener("change", () => this.onName());
      nameSuggest.querySelector("select").addEventListener("change", (e) => this.onChoice(e.target.value));
      smilesInput.addEventListener("change", () => this.onSmiles());
      classSel.addEventListener("change", () => this.onClass());
      usiInput.addEventListener("change", () => this.onUsi());
      SMA.searchable(classSel, "Type to search class labels…");

      classes().then((labels) => {
        classSel.append(...labels.map((l) => h("option", { value: l, selected: l === this.st.classLabel }, l)));
        SMA.refreshSelect(classSel);
      });
      this.sync();
      this.updatePreview();
    }

    sync() {
      const { nameInput, nameSuggest, nameStatus, smilesInput, classSel, usiInput } = this.el;
      nameInput.value = this.st.nameQuery;
      smilesInput.value = this.st.smiles;
      usiInput.value = this.st.usi;
      classSel.value = this.st.classLabel;
      SMA.refreshSelect(classSel);
      const sel = nameSuggest.querySelector("select");
      if (this.st.suggestions.length) {
        nameSuggest.classList.remove("hidden");
        sel.replaceChildren(...this.st.suggestions.map((s) => h("option", { value: s, selected: s === this.st.nameChoice }, s)));
      } else {
        nameSuggest.classList.add("hidden");
      }
      nameStatus.replaceChildren(this.st.nameWarning ? msg("warning", this.st.nameWarning) : "");
    }

    async onName() {
      const q = this.el.nameInput.value;
      Object.assign(this.st, { nameQuery: q, editorOpen: false, newSmiles: "", smiles: "", classLabel: "", usi: "" });
      if (q && q !== this.st.lastFetched) {
        this.el.nameStatus.replaceChildren(spinner("Searching PubChem…"));
        try {
          const r = await api(`/api/pubchem/suggest?q=${encodeURIComponent(q)}`);
          Object.assign(this.st, { suggestions: r.suggestions || [], lastFetched: q, nameWarning: null });
          if (this.st.suggestions.length) Object.assign(this.st, { nameChoice: this.st.suggestions[0], smiles: r.smiles || "", nameWarning: r.warning });
        } catch (e) { this.st.nameWarning = e.message; }
      } else if (!q) {
        Object.assign(this.st, { suggestions: [], lastFetched: null, nameWarning: null });
      }
      this.save(); this.sync(); this.updatePreview();
    }

    async onChoice(choice) {
      this.st.nameChoice = choice;
      this.el.nameStatus.replaceChildren(spinner("Resolving structure…"));
      try {
        const r = await api(`/api/pubchem/resolve?name=${encodeURIComponent(choice)}`);
        Object.assign(this.st, { smiles: r.smiles || "", nameWarning: r.warning });
      } catch (e) { Object.assign(this.st, { smiles: "", nameWarning: e.message }); }
      Object.assign(this.st, { editorOpen: false, newSmiles: this.st.smiles });
      this.save(); this.sync(); this.updatePreview();
    }

    onSmiles() {
      Object.assign(this.st, { smiles: this.el.smilesInput.value.trim(), editorOpen: false, newSmiles: "", nameQuery: "",
        suggestions: [], lastFetched: null, nameWarning: null, classLabel: "", usi: "" });
      this.save(); this.sync(); this.updatePreview();
    }

    onClass() {
      Object.assign(this.st, { classLabel: this.el.classSel.value, editorOpen: false, newSmiles: "", smiles: "", usi: "" });
      this.save(); this.sync(); this.updatePreview();
    }

    onUsi() {
      Object.assign(this.st, { usi: this.el.usiInput.value.trim(), smiles: "", newSmiles: "", editorOpen: false, classLabel: "",
        nameQuery: "", suggestions: [], lastFetched: null, nameWarning: null });
      this.save(); this.sync(); this.updatePreview();
    }

    async updatePreview() {
      const area = this.preview;
      const seq = (this.seq = (this.seq || 0) + 1);
      if (this.st.usi) { this.type = "usi"; area.replaceChildren(); this.emit(); return; }
      if (!this.st.smiles) {
        if (this.st.classLabel) {
          this.type = "class_label";
          area.replaceChildren(msg("info", `Class Label selected: ${this.st.classLabel}. Proceed to find spectra for molecules in this class.`));
        } else {
          this.type = null;
          area.replaceChildren();
        }
        this.emit();
        return;
      }
      area.replaceChildren(spinner("Reading structure…"));
      let r;
      try { r = await api("/api/preview", { body: { text: this.effective() } }); } catch (e) {
        if (seq === this.seq) area.replaceChildren(msg("error", e.message));
        return;
      }
      if (seq !== this.seq) return;
      this.type = r.type;
      area.replaceChildren();
      if (r.type === "smiles") {
        area.append(h("div", { class: "btn-row" },
          h("button", { type: "button", onclick: () => { this.st.editorOpen = true; this.save(); this.updatePreview(); } }, "✎ Edit Structure")));
        if (this.st.editorOpen) area.append(this.editorBlock(this.effective()));
        if (r.image) area.append(h("img", { class: "structure-img", src: r.image, alt: "structure", width: 300, height: 300 }));
        else if (r.warning) area.append(msg("warning", r.warning));
      } else if (r.type === "smarts") {
        if (r.image) area.append(h("img", { class: "structure-img", src: r.image, alt: "SMARTS", style: "max-width:500px" }));
        else area.append(msg("warning", r.warning || "Failed to retrieve SMARTS image from the API. You can still go ahead."));
      } else {
        area.append(msg("warning", "Not a valid SMILES/SMARTS"));
      }
      this.emit();
    }

    editorBlock(smiles) {
      const frame = h("iframe", { class: "editor-frame", src: "/static/ketcher/index.html", title: "Structure editor" });
      const status = h("div");
      const ketcher = async () => {
        for (let i = 0; i < 100; i++) {
          const k = frame.contentWindow && frame.contentWindow.ketcher;
          if (k) return k;
          await new Promise((r) => setTimeout(r, 150));
        }
        throw new Error("The structure editor did not load.");
      };
      frame.addEventListener("load", async () => {
        try { (await ketcher()).setMolecule(smiles); } catch (e) { status.replaceChildren(msg("error", e.message)); }
      });
      const apply = h("button", {
        type: "button", class: "primary",
        onclick: async () => {
          try {
            this.st.newSmiles = await (await ketcher()).getSmiles();
            this.save(); this.updatePreview();
          } catch (e) { status.replaceChildren(msg("error", e.message)); }
        },
      }, "Apply");
      const close = h("button", { type: "button", title: "Warning: You will lose your changes",
        onclick: () => { this.st.editorOpen = false; this.save(); this.updatePreview(); } }, "✕ Close Structure Editor");
      return h("details", { class: "expander", open: true }, h("summary", {}, "Structure Editor"),
        h("div", { class: "body" }, msg("info", "You can edit the structure below and click Apply to update it."),
          h("div", { class: "btn-row" }, apply, close), frame, status));
    }
  }

  // -------------------------------------------------------------------------
  // Library search type + optional formula / element filters / Tanimoto
  // -------------------------------------------------------------------------
  const SEARCH_OPTIONS = [["exact", "Exact structure match"], ["substructure", "Substructure match"], ["tanimoto", "Tanimoto similarity"]];

  class LibraryOptions {
    constructor(container, prefix, input, opts = {}) {
      this.key = `sm_lib_${prefix}`;
      const defaults = { searchtype: "exact", tanimoto: "0.8", formula: "", elements: "" };
      this.st = loadState(this.key, defaults);
      this.input = input;
      const p = SMA.linkParams("lib");
      if (p && !sameSettings(p, this.urlParams())) {
        this.st = Object.assign({}, defaults, {
          searchtype: SEARCH_OPTIONS.some((o) => o[0] === p.searchtype) ? p.searchtype : defaults.searchtype,
          tanimoto: p.tanimoto !== undefined && Number(p.tanimoto) >= 0 && Number(p.tanimoto) <= 1 ? p.tanimoto : defaults.tanimoto,
          formula: p.formula || "", elements: p.elements || "",
        });
        saveState(this.key, this.st);
      }
      this.label = opts.label || "Find available MS/MS spectra";
      this.root = h("div");
      container.append(this.root);
      input.onChange(() => this.render());
      this.render();
      SMA.trackUrl("lib", () => this.urlParams());
    }

    urlParams() {
      if (!this.input.hasInput()) return {}; // nothing entered yet: keep the address short
      const { searchtype: t, tanimoto, formula, elements } = this.st;
      return { searchtype: t, tanimoto: t === "tanimoto" ? tanimoto : "",
        formula: t !== "exact" ? formula : "", elements: t !== "exact" ? elements : "" };
    }

    render() {
      const type = this.input.type;
      let opts = SEARCH_OPTIONS;
      if (type === "class_label") opts = SEARCH_OPTIONS.filter((o) => o[0] === "exact");
      else if (type === "smarts") opts = SEARCH_OPTIONS.filter((o) => o[0] === "substructure");
      if (!opts.some((o) => o[0] === this.st.searchtype)) this.st.searchtype = opts[0][0];
      const set = (k) => (e) => { this.st[k] = e.target.value; saveState(this.key, this.st); };
      const text = (label, k, helpText) => h("div", { class: "col" },
        h("label", { class: "field", title: helpText || "" }, label),
        h("input", { type: "text", value: this.st[k], onchange: set(k) }));
      const row = h("div", { class: "row" },
        h("div", { class: "col", style: "flex:2" },
          h("label", { class: "field" }, this.label),
          h("div", { class: "radio-group" }, opts.map(([v, label]) => h("label", {},
            h("input", { type: "radio", name: `${this.key}_st`, value: v, checked: v === this.st.searchtype,
              onchange: () => { this.st.searchtype = v; saveState(this.key, this.st); this.render(); } }), label)))));
      if (this.st.searchtype !== "exact") {
        row.append(text("Fixed molecular formula (optional)", "formula", "Specify a fixed molecular formula that must be matched by returned molecules."));
      }
      if (this.st.searchtype === "tanimoto") row.append(text("Tanimoto threshold", "tanimoto"));
      if (this.st.searchtype !== "exact") {
        row.append(text("Allowed element difference (optional)", "elements",
          "Specify what elements can differ between input structure (if specied as SMILES) and library structures (e.g. 'NO' allows for difference in nitrogen and oxygen content)"));
      }
      row.append(h("div", { class: "col", style: "flex:2" }));
      this.root.replaceChildren(row);
      SMA.syncUrl(); // the search type may have been narrowed to fit the input
    }

    values() {
      return { searchtype: this.st.searchtype, tanimoto: this.st.tanimoto, allowed_formula: this.st.formula, allowed_elements: this.st.elements };
    }
  }

  // -------------------------------------------------------------------------
  // Raw data search options (mode, cosine, peaks, rank, FASST + modifications)
  // -------------------------------------------------------------------------
  const LAST_ITERATION = "09/2025";
  const MOD_CONDITIONS = ["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"];

  class RawOptions {
    constructor(container, prefix, opts = {}) {
      this.key = `sm_raw_${prefix}`;
      this.opts = Object.assign({ cards: false, analogues: false, forceFasst: false }, opts);
      const defaults = {
        mode: "FASSTrecords", minCos: 0.7, minPeaks: 5, minRank: 0, precTol: "0.02", fragTol: "0.02",
        modSearch: false, elimination: true, addition: true, analoguesSameDataset: false,
        modFormula: "", modMass: "", subsetMod: false, modCondition: "Raw file",
      };
      this.st = loadState(this.key, defaults);
      const p = SMA.linkParams("raw");
      if (p && !sameSettings(p, this.urlParams())) {
        // numbers are checked here and clamped to the allowed range in render()
        const num = (k, d) => (p[k] !== undefined && p[k] !== "" && Number.isFinite(Number(p[k])) ? Number(p[k]) : d);
        const flag = (k, d) => (p[k] === undefined ? d : p[k] === "1" || p[k] === "true");
        this.st = Object.assign({}, defaults, {
          mode: ["FASSTrecords", "FASST"].includes(p.mode) ? p.mode : defaults.mode,
          minCos: num("cosine", defaults.minCos), minPeaks: Math.round(num("peaks", defaults.minPeaks)),
          minRank: Math.min(10, Math.max(0, Math.round(num("rank", 0)))),
          precTol: p.prectol || defaults.precTol, fragTol: p.fragtol || defaults.fragTol,
          modSearch: flag("modsearch", false), elimination: flag("elimination", true), addition: flag("addition", true),
          analoguesSameDataset: flag("analogues", false), modFormula: p.modformula || "", modMass: p.modmass || "",
          subsetMod: MOD_CONDITIONS.includes(p.modcondition),
          modCondition: MOD_CONDITIONS.includes(p.modcondition) ? p.modcondition : defaults.modCondition,
        });
        saveState(this.key, this.st);
      }
      this.root = h("div");
      container.append(this.root);
      this.render();
      SMA.trackUrl("raw", () => this.urlParams());
    }

    urlParams() {
      const s = this.st;
      const mode = this.mode();
      const out = { mode, cosine: s.minCos, peaks: s.minPeaks, rank: Number(s.minRank) ? s.minRank : "" };
      if (mode === "FASST") {
        Object.assign(out, { prectol: s.precTol, fragtol: s.fragTol });
        if (s.modSearch) {
          Object.assign(out, { modsearch: "1", elimination: s.elimination ? "1" : "0", addition: s.addition ? "1" : "0",
            analogues: this.opts.analogues && s.analoguesSameDataset ? "1" : "",
            modformula: s.modFormula, modmass: s.modMass, modcondition: s.subsetMod ? s.modCondition : "" });
        }
      }
      return out;
    }

    setForceFasst(v) { this.opts.forceFasst = v; this.render(); SMA.syncUrl(); }
    mode() { return this.opts.forceFasst ? "FASST" : this.st.mode; }

    render() {
      const mode = this.mode();
      const [minCosAllowed, minPeaksAllowed] = mode === "FASST" ? [0.3, 1] : [0.7, 3];
      this.st.minCos = Math.min(1, Math.max(minCosAllowed, Number(this.st.minCos) || 0.7));
      this.st.minPeaks = Math.max(minPeaksAllowed, parseInt(this.st.minPeaks, 10) || 5);
      const save = () => saveState(this.key, this.st);
      const set = (k) => (e) => { this.st[k] = e.target.type === "checkbox" ? e.target.checked : e.target.value; save(); };
      const rerender = (k) => (e) => { set(k)(e); this.render(); };
      const root = h("div");

      root.append(h("div", {}, h("label", { class: "field" }, "Mode"),
        h("div", { class: "radio-group" }, ["FASSTrecords", "FASST"].map((m) => h("label", {},
          h("input", { type: "radio", name: `${this.key}_mode`, value: m, checked: mode === m, disabled: this.opts.forceFasst, onchange: rerender("mode") }), m)))));
      if (this.opts.forceFasst) root.append(h("div", { class: "small muted" }, "USI and MGF queries have no library spectrum ids, so only FASST can search them."));

      if (this.opts.cards) {
        root.append(h("div", { class: "row" },
          h("div", { class: "col card", html: `<h4><strong>FASSTrecords</strong></h4><p>This generally takes less than 1 minute even for hundreds of spectra as it relies on precomputed annotations, and is therefore especially recommended for large numbers of spectra or molecules expected in large numbers of samples.<br/>Last iteration: <strong>${LAST_ITERATION}</strong>.</p>` }),
          h("div", { class: "col card orange", html: `<h4><strong>FASST</strong></h4><p>Can be slow especially for molecules expected to be present in many datasets or large numbers of spectra. Searches can take several minutes and even fail depending on current traffic.<br/>Allows the discovery of unknown chemical analogues through modification search.<br/>Always up to date with the latest raw data indexed at <a href="https://fasst.gnps2.org/" target="_blank" style="color:#e76f51;">fasst.gnps2.org</a>.</p>` })));
      }

      const num = (label, key, attrs) => h("div", { class: "col" }, h("label", { class: "field" }, label),
        h("input", Object.assign({ type: "number", value: this.st[key], onchange: set(key) }, attrs)));
      root.append(h("div", { class: "row" },
        num("Minimum Cosine", "minCos", { min: minCosAllowed, max: 1, step: 0.01 }),
        num("Minimum Matching Peaks", "minPeaks", { min: minPeaksAllowed, step: 1 }),
        num("Minimum Annotation Rank (0 disables)", "minRank", { min: 0, max: 10, step: 1 }),
        h("div", { class: "col" })));

      if (mode === "FASST") {
        const text = (label, key, ph) => h("div", {}, h("label", { class: "field" }, label),
          h("input", { type: "text", value: this.st[key], placeholder: ph || "", onchange: set(key) }));
        const check = (label, key, onchange, title) => h("label", { class: "check", title: title || "" },
          h("input", { type: "checkbox", checked: this.st[key], onchange: onchange || set(key) }), label);
        const left = h("div", { class: "col" }, text("Fragment Tolerance (Da)", "fragTol"));
        const right = h("div", { class: "col" }, text("Precursor Tolerance (Da)", "precTol"),
          h("div", { style: "margin-top:0.5rem" }, check("Modification search", "modSearch", rerender("modSearch"))));
        if (this.st.modSearch) {
          left.append(h("div", { class: "row", style: "margin-top:0.5rem" },
            h("div", { class: "col" }, check("Elimination search", "elimination")),
            h("div", { class: "col" }, check("Addition search", "addition"))));
          if (this.opts.analogues) {
            left.append(h("div", { style: "margin-top:0.5rem" }, check("Only analogues found in a dataset that also contains the unmodified compound",
              "analoguesSameDataset", null, "A modification is more credible when the parent compound is present in the same study. Unmodified matches are always kept.")));
          }
          left.append(h("div", { style: "display:grid;grid-template-columns:2fr 0.5fr 2fr;gap:0.5rem" },
            text("Modification formula", "modFormula", "O for hydroxylation"), h("div", { class: "or" }, "or"),
            text("Modification mass (Da)", "modMass", "15.9949 for O")));
          left.append(h("div", { style: "margin-top:0.5rem" }, check("Only report modified molecules if <condition>", "subsetMod", rerender("subsetMod"))));
          if (this.st.subsetMod) {
            left.append(h("div", { style: "margin-top:0.5rem" }, h("label", { class: "field" }, "Unmodified found in same"),
              h("select", { onchange: set("modCondition") },
                MOD_CONDITIONS.map((o) => h("option", { value: o, selected: this.st.modCondition === o }, o)))));
          }
        }
        root.append(h("div", { class: "row", style: "margin-top:0.5rem" }, left, right, h("div", { class: "col" }), h("div", { class: "col" })));
      }
      this.root.replaceChildren(root);
    }

    params() {
      const mode = this.mode();
      const mod = mode === "FASST" && this.st.modSearch;
      return {
        mode, min_cosine: Number(this.st.minCos), min_peaks: parseInt(this.st.minPeaks, 10), min_rank: parseInt(this.st.minRank, 10) || 0,
        prec_tol: this.st.precTol, frag_tol: this.st.fragTol,
        modification_search: mod, elimination: this.st.elimination, addition: this.st.addition,
        modification_formula: this.st.modFormula, modification_mass: this.st.modMass,
        modification_condition: mod && this.st.subsetMod ? this.st.modCondition : null,
        analogues_same_dataset: mod && this.opts.analogues && this.st.analoguesSameDataset,
      };
    }
  }

  // -------------------------------------------------------------------------
  // Live job log (data preparation, Nextflow)
  // -------------------------------------------------------------------------
  function logProgress(box, label) {
    const head = h("div");
    const pre = h("pre", { class: "joblog hidden" });
    box.replaceChildren(head, pre);
    return (v) => {
      head.replaceChildren(spinner(v.message || label));
      if (v.log && v.log.length) {
        pre.classList.remove("hidden");
        pre.textContent = v.log.slice(-40).join("\n");
        pre.scrollTop = pre.scrollHeight;
      }
    };
  }

  function showLog(box, v) {
    if (v.log && v.log.length) {
      box.append(h("details", {}, h("summary", {}, "Show log"), h("pre", { class: "joblog" }, v.log.slice(-200).join("\n"))));
    }
  }

  // -------------------------------------------------------------------------
  // Reference data panel (bin/dependency_prep.py)
  // -------------------------------------------------------------------------
  async function prepPanel(container, tool) {
    const box = h("div");
    container.append(box);
    let st;
    try { st = await api(`/api/prep/${tool}`); } catch (e) { box.append(msg("error", e.message)); return; }
    const body = h("div", { class: "body" });
    box.append(h("details", { class: "expander" }, h("summary", {}, `${st.ready ? "✅" : "⚠️"} ${st.label} reference data`), body));
    if (st.note) body.append(h("p", { class: "small muted" }, st.note));
    makeTable(body, { columns: ["Data", "Status", "Delivered by", "Size (MB)"], rows: st.entries.map((e) => [e.Data, e.Status, e["Delivered by"], e["Size (MB)"]]) }, { name: `${tool}_reference_data` });
    if (st.download_mb) body.append(h("p", { class: "small muted" }, `About ${Math.round(st.download_mb)} MB will be fetched the first time you run ${st.label}.`));
    const status = h("div");
    const btn = h("button", {
      type: "button",
      onclick: async () => {
        btn.disabled = true;
        try {
          const v = await runJob(`/api/prep/${tool}`, {}, logProgress(status, `Preparing ${st.label} reference data…`));
          jobOutcome(status, v, v.status === "done" ? `${st.label} data is ready.` : null);
          showLog(status, v);
        } catch (e) { status.replaceChildren(msg("error", e.message)); }
        btn.disabled = false;
      },
    }, `Prepare ${st.label} data now`);
    body.append(h("div", { class: "btn-row" }, btn), status);
  }

  return { StructureInput, LibraryOptions, RawOptions, logProgress, showLog, prepPanel };
})();
