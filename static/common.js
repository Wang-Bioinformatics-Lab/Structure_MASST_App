// Shared helpers for all StructureMASST pages.
"use strict";

const SMA = (() => {
  // ---------- session (per browser tab, like a Streamlit session) ----------
  const SID_KEY = "structuremasst_sid";

  function storageGet(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
  function storageSet(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* ignore */ } }

  let sid = storageGet(SID_KEY);

  async function ensureSession() {
    if (sid && /^([0-9a-f]{12}|[0-9a-f]{32})$/.test(sid)) return sid;
    const r = await fetch("/api/session", { method: "POST" });
    sid = (await r.json()).sid;
    storageSet(SID_KEY, sid);
    return sid;
  }

  // ---------- API ----------
  async function api(path, opts = {}) {
    await ensureSession();
    const headers = Object.assign({ "X-Session-Id": sid }, opts.headers || {});
    let body = opts.body;
    if (body && !(body instanceof FormData) && typeof body !== "string") {
      body = JSON.stringify(body);
      headers["Content-Type"] = "application/json";
    }
    const r = await fetch(path, { method: opts.method || (body ? "POST" : "GET"), headers, body });
    let data = null;
    try { data = await r.json(); } catch (e) { /* non-JSON */ }
    if (!r.ok) throw new Error((data && data.error) || `Request failed (${r.status})`);
    return data;
  }

  // Start a job and poll it; onProgress(jobView) is called while it runs.
  async function runJob(path, body, onProgress) {
    const { job } = await api(path, { body: body || {} });
    for (;;) {
      const v = await api(`/api/jobs/${job}`);
      if (onProgress) onProgress(v);
      if (v.status !== "running") return v;
      await new Promise((res) => setTimeout(res, 1000));
    }
  }

  function fileUrl(name) {
    return `/api/files/${name.split("/").map(encodeURIComponent).join("/")}?sid=${sid}`;
  }

  // ---------- DOM helpers ----------
  function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // Minimal **bold** support for messages (as st.markdown did); everything else is escaped.
  function md(text) {
    return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  }

  function msg(kind, text, extra) {
    const el = h("div", { class: `msg ${kind}`, html: md(text) });
    if (extra) el.append(extra);
    return el;
  }

  function spinner(text) {
    return h("div", { class: "spinner" }, text || "Running…");
  }

  // Renders a job's progress into `box`; returns a callback for runJob.
  function jobProgress(box, label) {
    return (v) => {
      box.replaceChildren(spinner(v.message || label));
    };
  }

  // Show the outcome of a finished job: warnings, error (with logs) or success.
  function jobOutcome(box, v, successText) {
    box.replaceChildren();
    for (const w of v.warnings || []) box.append(msg("warning", w));
    if (v.status === "error") {
      const err = v.error || "Failed.";
      const nl = err.indexOf("\n");
      if (nl > 0) {
        const det = h("details", {}, h("summary", {}, "Show logs"), h("pre", {}, err.slice(nl + 1)));
        box.append(msg("error", err.slice(0, nl), det));
      } else {
        box.append(msg("error", err));
      }
      return false;
    }
    const text = successText || (v.result && v.result.message);
    if (text) box.append(msg("success", text));
    return true;
  }

  // ---------- page shell (sidebar like the Streamlit app) ----------
  const PAGES = [
    ["/", "StructureMASST"],
    ["/domainmasst", "domainMASST (under construction)"],
    ["/geomasst", "geoMASST (under construction)"],
    ["/lifemasst", "lifeMASST"],
  ];

  function shell(active) {
    const main = document.getElementById("main");
    const sidebar = h("aside", { class: "sidebar", id: "sidebar" },
      h("a", { href: "/" }, h("img", { class: "logo", src: "/logo.png", alt: "StructureMASST" })),
      h("nav", {}, PAGES.map(([href, label]) => h("a", { href, class: href === active ? "active" : null }, label))),
      h("div", { class: "contributors", html: "<strong>Contributors</strong><br>Yasin El Abiead (BOKU University)<br>Wilhan Nunes (UCSD)<br>Mingxun Wang (UCR)<br>" }),
    );
    const toggle = h("button", { class: "sidebar-toggle", onclick: () => sidebar.classList.toggle("open") }, "☰ Menu");
    const wrapper = h("div", { class: "shell" }, sidebar, h("div", { class: "main-col", style: "flex:1;min-width:0" }, toggle, main));
    main.classList.add("main");
    document.body.prepend(wrapper);
  }

  // ---------- tables (Tabulator) ----------
  // table = {columns:[...], rows:[[...]]}
  // opts: selectable, titles: {col: header}, links: {col: displayText}, images: {col: {width, height}}, height, name
  function makeTable(container, table, opts = {}) {
    const wrap = h("div", { class: "table-wrap" });
    const tools = h("div", { class: "table-tools" });
    const target = h("div");
    wrap.append(tools, target);
    container.append(wrap);

    const cols = table.columns || [];
    const data = (table.rows || []).map((r, i) => {
      const o = { __i: i };
      cols.forEach((c, j) => { o["c" + j] = r[j]; });
      return o;
    });
    const links = opts.links || {};
    const images = opts.images || {};
    const columns = cols.map((c, j) => {
      // column names can come from uploaded files: escape them
      const def = { title: escapeHtml((opts.titles || {})[c] || c), field: "c" + j, headerTooltip: true, maxWidth: 420 };
      if (links[c]) {
        // only real web links become clickable (never javascript: or data: URLs)
        def.formatter = (cell) => {
          const v = cell.getValue();
          if (typeof v !== "string" || !/^https?:\/\//i.test(v)) return "";
          const a = document.createElement("a");
          a.href = v; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = links[c];
          return a;
        };
        def.headerSort = false;
      } else if (images[c]) {
        def.formatter = (cell) => {
          const v = cell.getValue();
          if (!v) return "";
          const img = document.createElement("img");
          img.src = v; img.loading = "lazy";
          img.width = images[c].width; img.height = images[c].height;
          return img;
        };
        def.headerSort = false;
        def.width = images[c].width + 16;
        def.maxWidth = undefined;
      } else {
        def.formatter = (cell) => {
          const v = cell.getValue();
          if (v === null || v === undefined) return '<span style="color:#aaa">None</span>';
          const s = String(v);
          return s.replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
        };
      }
      return def;
    });
    if (opts.selectable) {
      columns.unshift({
        formatter: "rowSelection", titleFormatter: "rowSelection", hozAlign: "center",
        headerSort: false, width: 40, frozen: true, cellClick: (e, cell) => cell.getRow().toggleSelect(),
      });
    }
    const tab = new Tabulator(target, {
      data,
      columns,
      layout: "fitData",
      height: opts.height || (data.length > 12 ? 440 : false),
      selectableRows: opts.selectable ? true : false,
      rowHeight: opts.rowHeight,
      placeholder: "No rows",
      columnDefaults: { resizable: true },
    });
    const csvBtn = h("button", {
      onclick: () => tab.download("csv", (opts.name || "table") + ".csv"),
    }, "Download CSV");
    const count = h("span", { class: "muted", style: "margin-right:auto" }, `${data.length} rows`);
    tools.append(count, csvBtn);
    return {
      tabulator: tab,
      selected: () => tab.getSelectedData().map((d) => d.__i).sort((a, b) => a - b),
    };
  }

  // ---------- tabs ----------
  // items: [{key, label, render(container)}]; remembers the active tab by `id`.
  const activeTabs = {};
  function tabs(container, id, items) {
    const bar = h("div", { class: "tabs" });
    const body = h("div");
    container.append(bar, body);
    if (!items.length) return;
    let active = activeTabs[id];
    if (!items.some((it) => it.key === active)) active = items[0].key;
    const show = (key) => {
      activeTabs[id] = key;
      for (const b of bar.children) b.classList.toggle("active", b.dataset.key === key);
      body.replaceChildren();
      const it = items.find((x) => x.key === key);
      if (it) it.render(body);
    };
    for (const it of items) {
      const b = h("button", { onclick: () => show(it.key) }, it.label);
      b.dataset.key = it.key;
      bar.append(b);
    }
    show(active);
  }

  return { ensureSession, api, runJob, fileUrl, h, md, msg, spinner, jobProgress, jobOutcome, shell, makeTable, tabs,
           get sid() { return sid; } };
})();
