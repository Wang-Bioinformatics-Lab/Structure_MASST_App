//! The StructureMASST workflow logic that used to live in structureMASST.py and
//! bin/streamlit_search_ui.py (between the widgets): building the query table,
//! library search and grouping, dereplication, raw data search
//! post-processing, the table actions and the downstream tools.

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use crate::jobs::JobCtx;
use crate::links;
use crate::session::{Group, MgfSpectrum, SessionRef};
use crate::table::{value_as_f64, value_as_i64, value_to_string, Table};
use crate::validate;
use crate::AppState;

// ---------------------------------------------------------------------------
// Search input → query table (build_query_table / the main page's input block)
// ---------------------------------------------------------------------------

/// What the search controls hold (shared by the StructureMASST, LifeMASST and GeoMASST pages).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct SearchInput {
    #[serde(default)]
    pub usi: String,
    /// effective SMILES/SMARTS (structure editor result if applied)
    #[serde(default)]
    pub smiles: String,
    #[serde(default)]
    pub class_label: String,
    #[serde(default)]
    pub name_query: String,
    /// exact | substructure | tanimoto
    #[serde(default = "default_searchtype")]
    pub searchtype: String,
    #[serde(default)]
    pub allowed_formula: String,
    #[serde(default)]
    pub allowed_elements: String,
    #[serde(default = "default_tanimoto")]
    pub tanimoto: String,
    #[serde(default)]
    pub append: bool,
}

fn default_searchtype() -> String {
    "exact".into()
}
fn default_tanimoto() -> String {
    "0.8".into()
}

/// An uploaded batch file.
pub struct Upload {
    pub filename: String,
    pub bytes: Vec<u8>,
}

/// Which page builds the table: the StructureMASST page and the shared
/// search controls of LifeMASST/GeoMASST differ in small details.
#[derive(Clone, Copy, PartialEq)]
pub enum Flavor {
    Main,
    Shortcut,
}

const QCOLS: [&str; 7] = ["query", "name", "type", "searchtype", "formula", "allowed_elements", "original_smiles"];

fn qrow(query: &str, name: &str, typ: &str, searchtype: &str, formula: &str, elements: &str, original: &str) -> Vec<Value> {
    vec![json!(query), json!(name), json!(typ), json!(searchtype), json!(formula), json!(elements), json!(original)]
}

static NON_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\w]").unwrap());
static NON_WORD_RUNS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s\W]+").unwrap());
static EDGE_UNDERSCORES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^_+|_+$").unwrap());

pub fn sanitize_batch_name(name: &str, flavor: Flavor) -> String {
    let s = NON_WORD.replace_all(name, "_").to_string();
    if flavor == Flavor::Shortcut {
        let s = NON_WORD_RUNS.replace_all(&s, "_").to_string();
        EDGE_UNDERSCORES.replace_all(&s, "").to_string()
    } else {
        s
    }
}

fn parse_tanimoto(s: &str) -> Result<f64> {
    let t: f64 = s.trim().parse().map_err(|_| anyhow!("Tanimoto threshold must be a number between 0 and 1."))?;
    if !(0.0..=1.0).contains(&t) {
        bail!("Tanimoto threshold must be a number between 0 and 1.");
    }
    Ok(t)
}

#[derive(Deserialize)]
struct Prepared {
    query: String,
    #[serde(rename = "type")]
    typ: String,
    original_smiles: String,
}

/// Turn the controls into the canonical query table (priority: USI > SMILES/SMARTS >
/// batch file > class label > name). Returns the table and any MGF spectra.
pub async fn build_query_table(
    app: &AppState,
    input: &SearchInput,
    upload: Option<Upload>,
    flavor: Flavor,
) -> Result<(Table, IndexMap<String, MgfSpectrum>)> {
    let st = input.searchtype.as_str();
    if !["exact", "substructure", "tanimoto"].contains(&st) {
        bail!("invalid search type");
    }
    let mut mgf = IndexMap::new();
    let mut table = Table::new(QCOLS.iter().map(|s| s.to_string()).collect());
    let formula = |selected: &str| if selected == "substructure" { input.allowed_formula.clone() } else { "any".into() };
    let elements = |selected: &str| if selected == "substructure" { input.allowed_elements.clone() } else { "any".into() };

    let smiles = input.smiles.trim();
    // check what was typed before it reaches RDKit or the database
    if !input.usi.trim().is_empty() {
        validate::usi(input.usi.trim())?;
    } else if !smiles.is_empty() {
        validate::structure(smiles)?;
    }
    validate::name(&input.name_query)?;
    for (v, what) in [(&input.allowed_formula, "Fixed molecular formula"), (&input.allowed_elements, "Allowed element difference")] {
        validate::formula_like(v.trim(), what)?;
    }
    if !input.usi.trim().is_empty() {
        table.rows.push(qrow(input.usi.trim(), "USI_Query", "usi", "usi", "any", "any", ""));
    } else if !smiles.is_empty() {
        let p: Vec<Prepared> = app.worker.call_as("prepare_queries", json!({"queries": [{"query": smiles}]})).await?;
        let p = p.into_iter().next().ok_or_else(|| anyhow!("could not read the structure"))?;
        if p.typ == "Invalid" {
            bail!("Not a valid SMILES/SMARTS");
        }
        let searchtype = if p.typ == "smarts" { "substructure" } else { st };
        table.rows.push(qrow(&p.query, "Input_query", &p.typ, searchtype, &formula(st), &elements(st), &p.original_smiles));
    } else if let Some(up) = upload {
        if up.filename.to_lowercase().ends_with(".mgf") {
            if flavor == Flavor::Shortcut {
                bail!("Upload a CSV file here (MGF batches are searched from the StructureMASST page).");
            }
            use base64::Engine;
            let spectra: Vec<Value> = app
                .worker
                .call_as("parse_mgf", json!({"data": base64::engine::general_purpose::STANDARD.encode(&up.bytes)}))
                .await?;
            if spectra.is_empty() {
                bail!("No valid spectra found in the uploaded MGF file.");
            }
            if spectra.len() > validate::max_batch() {
                bail!("The MGF file has {} spectra; the limit is {}.", spectra.len(), validate::max_batch());
            }
            for s in spectra {
                let sid = s["spectrum_id"].as_str().unwrap_or_default().to_string();
                let name = sanitize_batch_name(s["name"].as_str().unwrap_or(&sid), Flavor::Main);
                mgf.insert(sid.clone(), MgfSpectrum { precursor_mz: s["precursor_mz"].clone(), peaks: s["peaks"].clone() });
                table.rows.push(qrow(&sid, &name, "mgf_spectrum", "mgf_spectrum", "any", "any", ""));
            }
        } else {
            table = csv_query_table(app, &up.bytes, st, flavor).await?;
        }
    } else if !input.class_label.trim().is_empty() {
        let c = input.class_label.trim();
        table.rows.push(qrow(c, c, "class_label", "class_label", "any", "any", ""));
    } else if !input.name_query.trim().is_empty() {
        bail!("Name search did not resolve to a usable single-component SMILES.");
    } else {
        bail!("Please enter a USI, SMILES/SMARTS, upload a CSV, or select a class label.");
    }

    // empty names become "Input_query"
    if let Some(c) = table.col("name") {
        for r in table.rows.iter_mut() {
            let n = value_to_string(&r[c]).trim().to_string();
            r[c] = json!(if n.is_empty() || r[c].is_null() { "Input_query".to_string() } else { n });
        }
    }
    if table.is_empty() {
        bail!("No valid search input could be prepared.");
    }
    validate_query_table(app, &table).await?;
    Ok((table, mgf))
}

/// Every row of the query table, whatever produced it (typed input or a batch file).
async fn validate_query_table(app: &AppState, t: &Table) -> Result<()> {
    if t.len() > validate::max_batch() {
        bail!("Too many queries ({}); the limit is {}.", t.len(), validate::max_batch());
    }
    let mut classes: Option<HashSet<&str>> = None;
    for r in 0..t.len() {
        let get = |c: &str| t.cell(r, c).filter(|v| !v.is_null()).map(value_to_string).unwrap_or_default();
        let (q, name, typ, st) = (get("query"), get("name"), get("type"), get("searchtype"));
        let row = |e: anyhow::Error| anyhow!("{}: {e}", if t.len() > 1 { format!("Row {} ({name})", r + 1) } else { "Input".to_string() });
        validate::name(&name).map_err(row)?;
        validate::one_of(&st, &["exact", "substructure", "tanimoto", "usi", "class_label", "mgf_spectrum"], "search type").map_err(row)?;
        match typ.as_str() {
            "usi" => validate::usi(&q).map_err(row)?,
            "smiles" | "smarts" => validate::structure(&q).map_err(row)?,
            "mgf_spectrum" => validate::name(&q).map_err(row)?,
            "class_label" => {
                if classes.is_none() {
                    let all = app.classes.get_or_try_init(|| async { app.worker.call_as::<Vec<String>>("molecule_classes", json!({})).await }).await?;
                    classes = Some(all.iter().map(String::as_str).collect());
                }
                if !classes.as_ref().unwrap().contains(q.as_str()) {
                    bail!("{}", row(anyhow!("unknown class label '{}'", q.chars().take(80).collect::<String>())));
                }
            }
            _ => bail!("{}", row(anyhow!("not a valid SMILES/SMARTS, USI or class label"))),
        }
        for (c, what) in [("formula", "Fixed molecular formula"), ("allowed_elements", "Allowed element difference")] {
            let v = get(c);
            if v != "any" {
                validate::formula_like(v.trim(), what).map_err(row)?;
            }
        }
    }
    Ok(())
}

async fn csv_query_table(app: &AppState, bytes: &[u8], selected: &str, flavor: Flavor) -> Result<Table> {
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let mut columns: Vec<String> = rdr.headers().context("Could not read CSV")?.iter().map(|s| s.trim().to_string()).collect();
    if !columns.iter().any(|c| c == "query") {
        match columns.iter().position(|c| c == "smiles") {
            Some(i) => columns[i] = "query".into(),
            None => bail!("CSV must contain either ('query','name') or ('smiles','name')."),
        }
    }
    let mut t = Table::new(columns.clone());
    for rec in rdr.records() {
        let rec = rec.context("Could not read CSV")?;
        t.rows.push((0..columns.len()).map(|i| rec.get(i).map(|v| if v.is_empty() { Value::Null } else { json!(v) }).unwrap_or(Value::Null)).collect());
    }
    if !t.has("name") {
        let names = (0..t.len()).map(|i| json!(format!("Input_{}", i + 1))).collect();
        t.set_column("name", names);
    }
    // dropna(subset=["query", "name"]), strip, sanitize names
    let (qc, nc) = (t.col("query").unwrap(), t.col("name").unwrap());
    t.rows.retain(|r| !r[qc].is_null() && !r[nc].is_null());
    for r in t.rows.iter_mut() {
        r[qc] = json!(value_to_string(&r[qc]).trim());
        r[nc] = json!(sanitize_batch_name(value_to_string(&r[nc]).trim(), flavor));
    }
    if t.is_empty() {
        bail!("The CSV has no rows with both a query and a name.");
    }
    if t.len() > validate::max_batch() {
        bail!("The CSV has {} queries; the limit is {}.", t.len(), validate::max_batch());
    }
    for r in t.rows.iter() {
        let q = value_to_string(&r[qc]);
        if q.len() > validate::MAX_QUERY_LEN {
            bail!("A query in the CSV is too long (max {} characters).", validate::MAX_QUERY_LEN);
        }
    }
    let given_types: Option<Vec<Value>> = t.col("type").map(|c| t.rows.iter().map(|r| r[c].clone()).collect());
    let req: Vec<Value> = t
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let typ = given_types.as_ref().map(|g| g[i].clone()).filter(|v| !v.is_null());
            json!({"query": r[qc], "type": typ.map(|v| value_to_string(&v).trim().to_string())})
        })
        .collect();
    let prepared: Vec<Prepared> = app.worker.call_as("prepare_queries", json!({"queries": req})).await?;
    t.set_column("query", prepared.iter().map(|p| json!(p.query)).collect());
    t.set_column("type", prepared.iter().map(|p| json!(p.typ)).collect());
    t.set_column("original_smiles", prepared.iter().map(|p| json!(p.original_smiles)).collect());
    if !t.has("searchtype") {
        let sts = prepared
            .iter()
            .map(|p| {
                json!(match p.typ.as_str() {
                    "usi" => "usi",
                    "smarts" => "substructure",
                    "class_label" => "class_label",
                    _ if flavor == Flavor::Main => "exact",
                    _ => selected,
                })
            })
            .collect();
        t.set_column("searchtype", sts);
    }
    for col in ["formula", "allowed_elements"] {
        let vals = t.column_values(col).into_iter().map(|v| if v.is_null() { json!("any") } else { v }).collect();
        t.set_column(col, vals);
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// Library search ("Get Available Spectra")
// ---------------------------------------------------------------------------

/// Add the "USI Viewer" link column and move it (plus name / precursor m/z) to the front.
fn with_spectrum_links(t: &Table) -> Table {
    let mut t = t.clone();
    if t.has("query_spectrum_id") {
        let links: Vec<Value> = t
            .column_values("query_spectrum_id")
            .iter()
            .map(|v| Value::String(links::library_spectrum_link(&value_to_string(v))))
            .collect();
        t.set_column("spectrum_link", links);
    }
    t.reorder_first(&["spectrum_link", "Compound_Name", "Precursor_MZ"])
}

fn first_non_null_string(t: &Table, col: &str) -> String {
    t.column_values(col).iter().find(|v| !v.is_null()).map(value_to_string).unwrap_or_default()
}

/// `pd.concat` + drop_duplicates on query_spectrum_id (or full rows).
fn concat_dedup(a: &Table, b: &Table) -> Table {
    if a.is_empty() && a.columns.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    let mut out = Table::concat(&[a, b]);
    let mut seen = HashSet::new();
    match out.col("query_spectrum_id") {
        Some(c) => out.rows.retain(|r| seen.insert(value_to_string(&r[c]))),
        None => out.rows.retain(|r| seen.insert(serde_json::to_string(r).unwrap_or_default())),
    }
    out
}

fn sha1_12(s: &str) -> String {
    sha1_smol::Sha1::from(s).digest().to_string()[..12].to_string()
}

/// Library entries for every row of a query table (grouped by name, then InChIKey).
pub async fn library_groups(
    app: &AppState,
    ctx: &JobCtx,
    query_table: &Table,
    tanimoto: Option<f64>,
    mgf: &IndexMap<String, MgfSpectrum>,
) -> Result<IndexMap<String, Group>> {
    let mut groups: IndexMap<String, Group> = IndexMap::new();
    let mut overviews: IndexMap<String, Vec<Table>> = IndexMap::new();
    let n = query_table.len();
    for r in 0..n {
        let get = |c: &str| query_table.cell(r, c).map(value_to_string).unwrap_or_default();
        let (q, name, typ, searchtype) = (get("query"), get("name"), get("type"), get("searchtype"));
        ctx.progress(format!("Finding spectra for {name} ({}/{n})…", r + 1));

        if ["usi", "mgf_spectrum"].contains(&typ.as_str()) || ["usi", "mgf_spectrum"].contains(&searchtype.as_str()) {
            let ik = sha1_12(&q);
            let pmz = if typ == "mgf_spectrum" { mgf.get(&q).map(|s| s.precursor_mz.clone()).unwrap_or(Value::Null) } else { Value::Null };
            let usi = if typ == "usi" { q.clone() } else { String::new() };
            let st = Table {
                columns: ["inchikey_first_block", "Compound_Name", "Smiles", "Precursor_MZ", "query_spectrum_id", "USI"].iter().map(|s| s.to_string()).collect(),
                rows: vec![vec![json!(ik), json!(name), json!(""), pmz, json!(q), json!(usi)]],
            };
            let g = groups.entry(name.clone()).or_default();
            let merged = match g.spectra.get(&ik) {
                Some(prev) => concat_dedup(prev, &with_spectrum_links(&st)),
                None => with_spectrum_links(&st),
            };
            g.spectra.insert(ik.clone(), merged);
            overviews.entry(name.clone()).or_default().push(Table {
                columns: ["Compound_Name", "inchikey_first_block", "Smiles", "USI"].iter().map(|s| s.to_string()).collect(),
                rows: vec![vec![json!(name), json!(ik), json!(""), json!(q)]],
            });
            continue;
        }

        let lib: Result<Table> = app
            .worker
            .call_as(
                "library_table",
                json!({
                    "smiles": q, "searchtype": searchtype,
                    "tanimoto_threshold": if searchtype == "tanimoto" { tanimoto } else { None },
                    "allowed_formula": get("formula"), "allowed_elements": get("allowed_elements"),
                }),
            )
            .await;
        let lib = match lib {
            Ok(t) => t,
            Err(e) => {
                ctx.warn(format!("Error for {name}: {e:#}"));
                continue;
            }
        };
        if lib.is_empty() {
            continue;
        }
        let g = groups.entry(name.clone()).or_default();
        let mut ov = Table::new(vec!["Compound_Name".into(), "inchikey_first_block".into(), "Smiles".into()]);
        for ik in lib.unique("inchikey_first_block") {
            let ik = value_to_string(&ik);
            let sub = with_spectrum_links(&lib.filter_eq("inchikey_first_block", &ik));
            let merged = match g.spectra.get(&ik) {
                Some(prev) => concat_dedup(prev, &sub),
                None => sub.clone(),
            };
            g.spectra.insert(ik.clone(), merged);
            ov.rows.push(vec![json!(first_non_null_string(&sub, "Compound_Name")), json!(ik), json!(first_non_null_string(&sub, "Smiles"))]);
        }
        overviews.entry(name).or_default().push(ov);
    }
    for (name, tables) in overviews {
        if let Some(g) = groups.get_mut(&name) {
            g.overview = add_formulas(app, &Table::concat(&tables.iter().collect::<Vec<_>>())).await;
        }
    }
    groups.retain(|_, g| !g.spectra.is_empty());
    Ok(groups)
}

/// Molecular_Formula column (after Compound_Name), as the overview table shows it.
async fn add_formulas(app: &AppState, ov: &Table) -> Table {
    let smiles: Vec<String> = ov.column_values("Smiles").iter().map(|v| if v.is_null() { String::new() } else { value_to_string(v) }).collect();
    let formulas: Vec<String> = app.worker.call_as("formulas", json!({"items": smiles})).await.unwrap_or_else(|_| vec![String::new(); ov.len()]);
    let mut out = ov.clone();
    out.set_column("Molecular_Formula", formulas.into_iter().map(Value::String).collect());
    out.reorder_first(&["Compound_Name", "Molecular_Formula"])
}

/// "Get Available Spectra" on the StructureMASST page (with optional append mode).
pub async fn library_search(
    app: Arc<AppState>,
    session: SessionRef,
    ctx: JobCtx,
    input: SearchInput,
    upload: Option<Upload>,
) -> Result<Value> {
    ctx.progress("Preparing structures…");
    let tanimoto = if input.searchtype == "tanimoto" { Some(parse_tanimoto(&input.tanimoto)?) } else { None };
    let (query_table, mgf) = build_query_table(&app, &input, upload, Flavor::Main).await?;
    let new_groups = library_groups(&app, &ctx, &query_table, tanimoto, &mgf).await?;

    let mut s = session.lock().await;
    let append = input.append && !s.data.groups.is_empty();
    s.data.raw_results = None;
    s.data.raw_mode = None;
    s.data.mgf_store.extend(mgf);
    for r in 0..query_table.len() {
        let name = query_table.cell(r, "name").map(value_to_string).unwrap_or_default();
        let q = query_table.cell(r, "query").map(value_to_string).unwrap_or_default();
        s.data.query_by_name.insert(name, q);
    }
    if !append {
        s.data.groups.clear();
        s.data.query_table = Some(query_table.clone());
    } else {
        let merged = match &s.data.query_table {
            Some(prev) if !prev.is_empty() => {
                let mut t = Table::concat(&[prev, &query_table]);
                let mut seen = HashSet::new();
                t.rows.retain(|r| seen.insert(serde_json::to_string(r).unwrap_or_default()));
                t
            }
            _ => query_table.clone(),
        };
        s.data.query_table = Some(merged);
    }
    if new_groups.is_empty() {
        s.save()?;
        bail!(if append { "No spectra available for this input. Existing results were kept." } else { "No spectra available for this input." });
    }
    for (name, g) in new_groups {
        match s.data.groups.get_mut(&name) {
            None => {
                s.data.groups.insert(name, g);
            }
            Some(existing) => {
                for (ik, t) in g.spectra {
                    let merged = match existing.spectra.get(&ik) {
                        Some(prev) => concat_dedup(prev, &t),
                        None => t,
                    };
                    existing.spectra.insert(ik, merged);
                }
                let mut ov = Table::concat(&[&existing.overview, &g.overview]);
                let mut seen = HashSet::new();
                if let Some(c) = ov.col("inchikey_first_block") {
                    ov.rows.retain(|r| seen.insert(value_to_string(&r[c])));
                }
                existing.overview = ov;
            }
        }
    }
    s.save()?;
    Ok(json!({"queries": s.data.groups.len()}))
}

// ---------------------------------------------------------------------------
// Library table actions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RowAction {
    Remove,
    Keep,
    Split,
}

fn sorted_unique(rows: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = rows.iter().copied().collect::<HashSet<_>>().into_iter().collect();
    v.sort_unstable();
    v
}

/// Take whole queries out of the session: their library entries, raw data
/// results and query rows.
fn remove_queries(data: &mut crate::session::SessionData, names: &HashSet<String>) {
    if names.is_empty() {
        return;
    }
    data.groups.retain(|n, _| !names.contains(n));
    if let Some(r) = data.raw_results.as_mut() {
        r.retain(|n, _| !names.contains(n));
    }
    data.query_by_name.retain(|n, _| !names.contains(n));
    if let Some(t) = data.query_table.as_mut() {
        if let Some(c) = t.col("name") {
            t.rows.retain(|r| !names.contains(value_to_string(&r[c]).trim()));
        }
    }
}

/// Molecules the last raw data search found nothing for: whole queries without
/// matches, and - in a query holding several molecules (an unsplit substructure
/// or Tanimoto search) - the molecules none of its matches belong to.
#[derive(Debug, Default)]
pub struct Unmatched {
    pub queries: Vec<String>,
    /// query -> rows of its molecule overview without matches
    pub within: Vec<(String, Vec<usize>)>,
}

impl Unmatched {
    /// Number of molecules (rows of the molecule overviews) this takes away.
    pub fn molecules(&self, data: &crate::session::SessionData) -> usize {
        let whole: usize = self.queries.iter().map(|q| data.groups.get(q).map_or(1, |g| g.overview.len().max(1))).sum();
        whole + self.within.iter().map(|(_, rows)| rows.len()).sum::<usize>()
    }
}

pub fn unmatched(data: &crate::session::SessionData) -> Unmatched {
    let mut u = Unmatched::default();
    let Some(results) = data.raw_results.as_ref() else { return u };
    for (name, t) in results {
        if t.is_empty() || !(t.has("mri_id_int") || t.has("mri")) {
            u.queries.push(name.clone());
            continue;
        }
        let Some(g) = data.groups.get(name) else { continue };
        let (Some(oc), true) = (g.overview.col("inchikey_first_block"), t.has("inchikey_first_block")) else { continue };
        if g.overview.len() < 2 {
            continue;
        }
        let hit: HashSet<String> = t.column_values("inchikey_first_block").iter().map(|v| value_to_string(v)).collect();
        let rows: Vec<usize> = (0..g.overview.len()).filter(|&r| !hit.contains(&value_to_string(&g.overview.rows[r][oc]))).collect();
        // a query with matches never loses all of its molecules here (keys that do
        // not line up, as for USI queries, must not empty it)
        if !rows.is_empty() && rows.len() < g.overview.len() {
            u.within.push((name.clone(), rows));
        }
    }
    u
}

/// "Remove molecules without raw data matches": from the library entries and
/// from the raw data results. Returns how many molecules were removed.
pub fn remove_unmatched(data: &mut crate::session::SessionData) -> usize {
    let u = unmatched(data);
    let n = u.molecules(data);
    for (name, rows) in &u.within {
        if let Some(g) = data.groups.get_mut(name) {
            let overview = g.overview.drop_rows(rows);
            let keep: HashSet<String> = overview.unique("inchikey_first_block").iter().map(value_to_string).collect();
            g.spectra.retain(|ik, _| keep.contains(ik));
            g.overview = overview;
        }
    }
    remove_queries(data, &u.queries.into_iter().collect());
    n
}

/// "Remove / Keep only selected molecule(s)" and "Split each molecule into separate query".
pub fn molecule_action(data: &mut crate::session::SessionData, name: &str, action: RowAction, rows: &[usize]) -> Result<()> {
    let rows = sorted_unique(rows);
    let group = data.groups.get(name).ok_or_else(|| anyhow!("unknown query {name}"))?.clone();
    match action {
        RowAction::Remove | RowAction::Keep => {
            if rows.is_empty() {
                bail!("No rows selected!");
            }
            let overview = if matches!(action, RowAction::Remove) { group.overview.drop_rows(&rows) } else { group.overview.take_rows(&rows) };
            if overview.is_empty() {
                // no molecule left: the query goes, with its raw data results
                remove_queries(data, &HashSet::from([name.to_string()]));
                return Ok(());
            }
            let keep: HashSet<String> = overview.unique("inchikey_first_block").iter().map(value_to_string).collect();
            let g = data.groups.get_mut(name).unwrap();
            g.spectra.retain(|ik, _| keep.contains(ik));
            g.overview = overview;
        }
        RowAction::Split => {
            let groups = &mut data.groups;
            if group.overview.len() < 2 {
                bail!("Splitting needs more than one molecule.");
            }
            let to_split = if rows.is_empty() { group.overview.clone() } else { group.overview.take_rows(&rows) };
            let mut new_rows = Vec::new();
            for (idx, row) in to_split.rows.iter().enumerate() {
                let cname = to_split.col("Compound_Name").map(|c| &row[c]).cloned().unwrap_or(Value::Null);
                let cname = if cname.is_null() { "None".to_string() } else { value_to_string(&cname) };
                let new_name = format!("{}_{}", cname, idx + 1);
                let ik = to_split.col("inchikey_first_block").map(|c| value_to_string(&row[c])).unwrap_or_default();
                let smiles = to_split.col("Smiles").map(|c| value_to_string(&row[c])).unwrap_or_default();
                let mut spectra = IndexMap::new();
                if let Some(t) = group.spectra.get(&ik) {
                    spectra.insert(ik.clone(), t.clone());
                }
                groups.insert(new_name.clone(), Group { overview: Table { columns: to_split.columns.clone(), rows: vec![row.clone()] }, spectra });
                // each split molecule is its own query now (downstream pages look results up by name)
                new_rows.push(qrow(&smiles, &new_name, "smiles", "exact", "any", "any", &smiles));
                data.query_by_name.insert(new_name, smiles);
            }
            groups.shift_remove(name);
            data.query_by_name.shift_remove(name);
            let add = Table { columns: QCOLS.iter().map(|s| s.to_string()).collect(), rows: new_rows };
            data.query_table = Some(match &data.query_table {
                Some(t) => Table::concat(&[t, &add]),
                None => add,
            });
        }
    }
    Ok(())
}

/// "Remove / Keep only selected spectra" for one InChIKey tab.
pub fn spectra_action(groups: &mut IndexMap<String, Group>, name: &str, ik: &str, action: RowAction, rows: &[usize]) -> Result<()> {
    let rows = sorted_unique(rows);
    if rows.is_empty() {
        bail!("No rows selected!");
    }
    let t = groups.get_mut(name).and_then(|g| g.spectra.get_mut(ik)).ok_or_else(|| anyhow!("unknown InChIKey {ik}"))?;
    *t = match action {
        RowAction::Remove => t.drop_rows(&rows),
        RowAction::Keep => t.take_rows(&rows),
        RowAction::Split => bail!("invalid action"),
    };
    Ok(())
}

/// String a value is matched by in the "Column / Value(s)" selection.
pub fn select_value(v: &Value) -> String {
    if v.is_null() { "<NA>".to_string() } else { value_to_string(v) }
}

/// Distinct selection values of a column, sorted.
pub fn column_values(t: &Table, col: &str) -> Vec<String> {
    let mut v: Vec<String> = t.column_values(col).iter().map(select_value).collect::<HashSet<_>>().into_iter().collect();
    v.sort();
    v
}

/// "Remove / Keep only selected rows" of a raw data result table. With a
/// column/value selection and `apply_all`, the same selection is applied to
/// every other molecule too.
pub fn result_action(
    results: &mut IndexMap<String, Table>,
    name: &str,
    action: RowAction,
    rows: &[usize],
    by_value: Option<(&str, &[String])>,
    apply_all: bool,
) -> Result<()> {
    let pick = |t: &Table, col: &str, vals: &[String]| -> Vec<usize> {
        t.column_values(col).iter().enumerate().filter(|(_, v)| vals.contains(&select_value(v))).map(|(i, _)| i).collect()
    };
    let t = results.get(name).ok_or_else(|| anyhow!("unknown query {name}"))?;
    let selected = match by_value {
        Some((col, vals)) if !vals.is_empty() => pick(t, col, vals),
        _ => sorted_unique(rows),
    };
    if selected.is_empty() {
        bail!("No rows selected!");
    }
    let apply = |t: &Table, sel: &[usize]| match action {
        RowAction::Remove => Ok(t.drop_rows(sel)),
        RowAction::Keep => Ok(t.take_rows(sel)),
        RowAction::Split => Err(anyhow!("invalid action")),
    };
    let new_t = apply(t, &selected)?;
    results.insert(name.to_string(), new_t);
    if let (true, Some((col, vals))) = (apply_all, by_value) {
        for (other, t) in results.iter_mut() {
            if other != name && t.has(col) {
                let sel = pick(t, col, vals);
                *t = apply(t, &sel)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw data search ("Search Raw Data")
// ---------------------------------------------------------------------------

/// Keep one spectrum per falcon cluster (the one closest to the representative).
/// Rows without library integer ids (USI / MGF queries) are kept as they are.
pub fn dereplicate(t: &Table) -> Table {
    if !t.has("spectrum_id_int") || !t.has("representative_spectrum_int") {
        let mut t = t.clone();
        for c in ["similar_library_spectra", "unique_spectra_in_mri"] {
            if !t.has(c) {
                let n = t.len();
                t.set_column(c, vec![json!(1); n]);
            }
        }
        return t;
    }
    if t.is_empty() {
        return t.clone();
    }
    let mut t = t.clone();
    let sid: Vec<Option<i64>> = t.column_values("spectrum_id_int").iter().map(value_as_i64).collect();
    let rep: Vec<Option<i64>> = t.column_values("representative_spectrum_int").iter().map(value_as_i64).collect();
    t.set_column("spectrum_id_int", sid.iter().map(|v| json!(v)).collect());
    t.set_column("representative_spectrum_int", rep.iter().map(|v| json!(v)).collect());
    let mut sizes: HashMap<i64, i64> = HashMap::new();
    for r in rep.iter().flatten() {
        *sizes.entry(*r).or_default() += 1;
    }
    t.set_column("similar_library_spectra", rep.iter().map(|r| json!(r.map(|r| sizes[&r]))).collect());
    t.set_column("spectrum_difference", sid.iter().zip(&rep).map(|(s, r)| json!(s.zip(*r).map(|(s, r)| s - r))).collect());

    let rep_c = t.col("representative_spectrum_int").unwrap();
    let diff_c = t.col("spectrum_difference").unwrap();
    let mut order: Vec<usize> = (0..t.len()).filter(|&i| rep[i].is_some()).collect();
    order.sort_by_key(|&i| (value_as_i64(&t.rows[i][rep_c]), value_as_i64(&t.rows[i][diff_c]).unwrap_or(i64::MAX)));
    let mut grouped: BTreeMap<i64, Vec<Value>> = BTreeMap::new();
    for i in order {
        let key = rep[i].unwrap();
        let row = &t.rows[i];
        match grouped.get_mut(&key) {
            None => {
                grouped.insert(key, row.clone());
            }
            Some(acc) => {
                for (c, v) in row.iter().enumerate() {
                    if acc[c].is_null() && !v.is_null() {
                        acc[c] = v.clone(); // pandas first(): first non-null per column
                    }
                }
            }
        }
    }
    let mut out = Table { columns: t.columns.clone(), rows: grouped.into_values().collect() };
    out = out.reorder_first(&["representative_spectrum_int"]);
    let reps = out.column_values("representative_spectrum_int");
    out.set_column("spectrum_id_int", reps);
    out
}

fn frames_for_query(group: &Group) -> Option<Table> {
    let frames: Vec<Table> = group.spectra.values().map(dereplicate).filter(|t| !t.is_empty()).collect();
    if frames.is_empty() {
        return None;
    }
    Some(Table::concat(&frames.iter().collect::<Vec<_>>()))
}

fn merge_compound_name(t: &Table, overview: &Table) -> Table {
    let mut ov = Table::new(vec!["inchikey_first_block".into(), "Compound_Name".into()]);
    for r in 0..overview.len() {
        ov.rows.push(vec![
            json!(overview.cell(r, "inchikey_first_block").map(value_to_string).unwrap_or_default()),
            overview.cell(r, "Compound_Name").cloned().unwrap_or(Value::Null),
        ]);
    }
    let mut t = t.clone();
    if t.has("inchikey_first_block") {
        let iks = t.column_values("inchikey_first_block").iter().map(|v| json!(value_to_string(v))).collect();
        t.set_column("inchikey_first_block", iks);
    }
    t.merge_left(&ov, "inchikey_first_block")
}

/// Column order and "Modified" sorting applied before results are shown.
pub fn arrange_results(t: &Table) -> Table {
    if !(t.has("mri_id_int") || t.has("mri")) || t.is_empty() {
        return t.clone();
    }
    let mut t = t.reorder_first(&["best_spectral_match", "modification_site", "Check LC peak"]);
    if t.has("Modified") {
        let order = ["addition", "elimination", "no"];
        let vals: Vec<Value> = t
            .column_values("Modified")
            .iter()
            .map(|v| {
                let s = value_to_string(v).to_lowercase();
                if order.contains(&s.as_str()) { json!(s) } else { Value::Null }
            })
            .collect();
        t.set_column("Modified", vals);
        let c = t.col("Modified").unwrap();
        t.rows.sort_by_key(|r| r[c].as_str().and_then(|s| order.iter().position(|o| *o == s)).unwrap_or(order.len()));
    }
    t
}

/// Port of `filter_analogues_to_parent_datasets`: keep an analogue only where the
/// unmodified compound was found in the same dataset.
pub fn filter_analogues_to_parent_datasets(t: &Table) -> Table {
    let delta_col = "Unit Delta Mass";
    if t.is_empty() || !t.has(delta_col) {
        return t.clone();
    }
    let Some(ds_col) = ["ATTRIBUTE_DatasetAccession", "Dataset"].into_iter().find(|c| t.has(c)) else {
        return t.clone();
    };
    let delta: Vec<Option<f64>> = t.column_values(delta_col).iter().map(value_as_f64).collect();
    let ds = t.column_values(ds_col);
    let parents: HashSet<String> = ds.iter().zip(&delta).filter(|(d, x)| **x == Some(0.0) && !d.is_null()).map(|(d, _)| value_to_string(d)).collect();
    let keep: Vec<usize> = (0..t.len())
        .filter(|&i| delta[i] == Some(0.0) || delta[i].is_none() || (!ds[i].is_null() && parents.contains(&value_to_string(&ds[i]))))
        .collect();
    t.take_rows(&keep)
}

#[derive(Clone, Debug, Deserialize)]
pub struct RawSearchParams {
    pub mode: String,
    pub min_cosine: f64,
    pub min_peaks: i64,
    #[serde(default)]
    pub min_rank: i64,
    #[serde(default)]
    pub prec_tol: Option<String>,
    #[serde(default)]
    pub frag_tol: Option<String>,
    #[serde(default)]
    pub modification_search: bool,
    #[serde(default)]
    pub elimination: bool,
    #[serde(default)]
    pub addition: bool,
    #[serde(default)]
    pub modification_formula: Option<String>,
    #[serde(default)]
    pub modification_mass: Option<String>,
    #[serde(default)]
    pub modification_condition: Option<String>,
    #[serde(default)]
    pub analogues_same_dataset: bool,
}

/// Do the groups contain direct USI or MGF queries (which FASSTrecords cannot search)?
pub fn has_direct_spectra(groups: &IndexMap<String, Group>, mgf: &IndexMap<String, MgfSpectrum>) -> bool {
    groups.values().flat_map(|g| g.spectra.values()).any(|t| {
        t.column_values("query_spectrum_id").iter().any(|v| {
            let s = value_to_string(v);
            !v.is_null() && (s.starts_with("mzspec:") || mgf.contains_key(&s))
        })
    })
}

/// The raw data search for every query group; returns name → ReDU-annotated matches.
pub async fn raw_results_for(
    app: &AppState,
    ctx: &JobCtx,
    groups: &IndexMap<String, Group>,
    mgf: &IndexMap<String, MgfSpectrum>,
    p: &RawSearchParams,
    output_folder: &Path,
) -> Result<IndexMap<String, Table>> {
    let (min_cos_allowed, min_peaks_allowed) = if p.mode == "FASST" { (0.3, 1) } else { (0.7, 3) };
    if p.min_cosine < min_cos_allowed - 1e-9 || p.min_cosine > 1.0 {
        bail!("Minimum Cosine must be between {min_cos_allowed} and 1.0 for {}.", p.mode);
    }
    if p.min_peaks < min_peaks_allowed {
        bail!("Minimum Matching Peaks must be at least {min_peaks_allowed} for {}.", p.mode);
    }
    if let Some(c) = &p.modification_condition {
        validate::one_of(c, &["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"], "modification condition")?;
    }
    if let Some(f) = &p.modification_formula {
        validate::formula_like(f.trim(), "Modification formula")?;
    }
    if p.min_rank < 0 || p.min_rank > 10 {
        bail!("Minimum Annotation Rank must be between 0 and 10.");
    }
    if p.mode == "FASSTrecords" && has_direct_spectra(groups, mgf) {
        bail!("FASSTrecords cannot run for USI input (no library spectrum_id_int available). Use FASST.");
    }

    let mut new_results: IndexMap<String, Table> = IndexMap::new();
    match p.mode.as_str() {
        "FASSTrecords" => {
            for (i, (name, group)) in groups.iter().enumerate() {
                let Some(df_for_name) = frames_for_query(group) else {
                    ctx.warn(format!("No entries left for **{name}**, skipping."));
                    continue;
                };
                ctx.progress(format!("Searching raw data for {name} ({}/{})…", i + 1, groups.len()));
                let redu: Table = app
                    .worker
                    .call_as(
                        "masst_and_redu",
                        json!({"table": df_for_name, "cosine_threshold": p.min_cosine, "matching_peaks": p.min_peaks,
                               "min_annotation_rank": p.min_rank, "chunk_size": 200}),
                    )
                    .await?;
                if !redu.has("Cosine") || !redu.has("Matching Peaks") {
                    new_results.insert(name.clone(), Table::default());
                    continue;
                }
                let mut redu = redu;
                if redu.has("mri_id_int") {
                    redu = merge_compound_name(&redu, &group.overview);
                    let n = redu.len();
                    redu.set_column("query_name", vec![json!(name); n]);
                }
                new_results.insert(name.clone(), arrange_results(&redu));
            }
        }
        "FASST" => {
            let prec_tol = validate::number(p.prec_tol.as_deref().unwrap_or("0.02"), "Precursor Tolerance")?;
            let frag_tol = validate::number(p.frag_tol.as_deref().unwrap_or("0.02"), "Fragment Tolerance")?;
            if !(0.0..=10.0).contains(&prec_tol) || !(0.0..=10.0).contains(&frag_tol) {
                bail!("Tolerances must be between 0 and 10 Da.");
            }

            ctx.progress("Checking the FASST API…");
            let mut fasst_works = false;
            for attempt in 1..=3 {
                if let Ok(v) = app.worker.call("fasst_health", json!({})).await {
                    if value_as_f64(&v).map(|n| n > 5.0).unwrap_or(false) {
                        fasst_works = true;
                        break;
                    }
                }
                if attempt < 3 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
            if !fasst_works {
                bail!(
                    "The FASST API is not responding after multiple attempts. Please try again later or contact support. \
                     FASSTrecords mode remains available."
                );
            }

            let mut modimass: Option<f64> = None;
            if p.modification_search {
                let formula = p.modification_formula.as_deref().unwrap_or("").trim();
                if !formula.is_empty() {
                    let v = app
                        .worker
                        .call("formula_mass", json!({"formula": formula}))
                        .await
                        .with_context(|| format!("Could not parse modification formula '{formula}'"))?;
                    modimass = value_as_f64(&v);
                } else if let Some(m) = p.modification_mass.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
                    modimass = Some(validate::number(m, "Modification mass")?);
                }
            }
            let condition = if p.modification_search { p.modification_condition.clone() } else { None };
            let common = json!({
                "database": "metabolomicspanrepo_index",
                "precursor_mz_tol": prec_tol, "fragment_mz_tol": frag_tol,
                "min_cos": p.min_cosine, "matching_peaks": p.min_peaks,
                "analog": p.modification_search, "modimass": modimass,
                "elimination": p.modification_search && p.elimination,
                "addition": p.modification_search && p.addition,
                "modification_condition": condition,
                "output_folder": output_folder,
            });

            for (i, (name, group)) in groups.iter().enumerate() {
                let Some(df_for_name) = frames_for_query(group) else {
                    ctx.warn(format!("No entries left for **{name}**, skipping."));
                    continue;
                };
                ctx.progress(format!("Searching raw data with FASST for {name} ({}/{})… this can take several minutes", i + 1, groups.len()));
                let ids: Vec<String> = df_for_name.column_values("query_spectrum_id").iter().map(value_to_string).collect();
                let is_mgf = ids.iter().any(|s| mgf.contains_key(s));
                let mut args = common.clone();
                let mut redu: Table = if is_mgf {
                    let spectra: Vec<Value> = ids
                        .iter()
                        .filter_map(|s| mgf.get(s).map(|m| json!({"spectrum_id": s, "precursor_mz": m.precursor_mz, "peaks": m.peaks})))
                        .collect();
                    args["spectra"] = json!(spectra);
                    app.worker.call_as("raw_data_matches_from_peaks", args).await?
                } else {
                    args["table"] = json!(df_for_name);
                    app.worker.call_as("raw_data_matches", args).await?
                };
                if !redu.is_empty() {
                    if !redu.has("lib_usi") {
                        let lib_usi = redu.column_values("query_spectrum_id").iter().map(|v| json!(links::library_usi(&value_to_string(v)))).collect();
                        redu.set_column("lib_usi", lib_usi);
                    }
                    let usis = redu.column_values("USI");
                    let libs = redu.column_values("lib_usi");
                    let best: Vec<Value> = usis
                        .iter()
                        .zip(&libs)
                        .map(|(u, l)| {
                            let l = if l.is_null() { String::new() } else { value_to_string(l) };
                            json!(if l.is_empty() { String::new() } else { links::spectraresolver_link(&value_to_string(u), &l) })
                        })
                        .collect();
                    redu.set_column("best_spectral_match", best);
                    if !redu.has("Precursor_MZ") && is_mgf && redu.has("query_spectrum_id") {
                        let pmz = redu.column_values("query_spectrum_id").iter().map(|v| mgf.get(&value_to_string(v)).map(|m| m.precursor_mz.clone()).unwrap_or(Value::Null)).collect();
                        redu.set_column("Precursor_MZ", pmz);
                    }
                    let lc = redu.column_values("Check LC peak");
                    if redu.has("Precursor_MZ") {
                        let mz = redu.column_values("Precursor_MZ");
                        let filled = lc
                            .iter()
                            .enumerate()
                            .map(|(r, v)| {
                                if v.is_null() || value_to_string(v).trim().is_empty() {
                                    json!(links::dashboard_eic_url(&value_to_string(&usis[r]), &mz[r], 0.05))
                                } else {
                                    v.clone()
                                }
                            })
                            .collect();
                        redu.set_column("Check LC peak", filled);
                    } else {
                        redu.set_column("Check LC peak", lc);
                    }
                    if !is_mgf && redu.has("inchikey_first_block") {
                        redu = merge_compound_name(&redu, &group.overview);
                    }
                    let n = redu.len();
                    redu.set_column("query_name", vec![json!(name); n]);
                }
                new_results.insert(name.clone(), arrange_results(&redu));
            }
            if p.analogues_same_dataset {
                for t in new_results.values_mut() {
                    *t = filter_analogues_to_parent_datasets(t);
                }
            }
        }
        other => bail!("unknown mode {other}"),
    }
    Ok(new_results)
}

pub async fn raw_search(app: Arc<AppState>, session: SessionRef, ctx: JobCtx, p: RawSearchParams) -> Result<Value> {
    let (groups, mgf, dir) = {
        let s = session.lock().await;
        (s.data.groups.clone(), s.data.mgf_store.clone(), s.dir.clone())
    };
    if groups.is_empty() {
        bail!("Run 'Get Available Spectra' first.");
    }
    let results = raw_results_for(&app, &ctx, &groups, &mgf, &p, &dir).await?;
    let mut s = session.lock().await;
    s.data.raw_results = Some(results);
    s.data.raw_mode = Some(p.mode.clone());
    s.save()?;
    Ok(json!({"ok": true}))
}

/// Library + raw data search from the shared controls on the LifeMASST/GeoMASST
/// pages (run_structuremasst_search). Does not touch the StructureMASST results.
pub async fn shortcut_search(
    app: &AppState,
    ctx: &JobCtx,
    input: &SearchInput,
    upload: Option<Upload>,
    p: &RawSearchParams,
    output_folder: &Path,
) -> Result<(Table, IndexMap<String, Table>)> {
    ctx.progress("Preparing structures…");
    let tanimoto = if input.searchtype == "tanimoto" { Some(parse_tanimoto(&input.tanimoto)?) } else { None };
    let (query_table, mgf) = build_query_table(app, input, upload, Flavor::Shortcut).await?;
    let groups = library_groups(app, ctx, &query_table, tanimoto, &mgf).await?;
    if groups.is_empty() {
        bail!("No spectra available for this input.");
    }
    let results = raw_results_for(app, ctx, &groups, &mgf, p, output_folder).await?;
    Ok((query_table, results))
}

// ---------------------------------------------------------------------------
// Downstream: DomainMASST
// ---------------------------------------------------------------------------

/// File-name safe version of a query name.
pub fn safe_name(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect();
    if s.is_empty() || s.starts_with('.') { format!("q{s}") } else { s }
}

fn write_tsv(path: &Path, t: &Table) -> Result<()> {
    let mut w = csv::WriterBuilder::new().delimiter(b'\t').from_path(path)?;
    w.write_record(&t.columns)?;
    for r in &t.rows {
        w.write_record(r.iter().map(|v| if v.is_null() { String::new() } else { value_to_string(v) }))?;
    }
    w.flush()?;
    Ok(())
}

fn topic_masst_input(t: &Table) -> Table {
    let keep: Vec<&str> = ["USI", "Cosine", "Matching Peaks", "Delta Mass"].into_iter().filter(|c| t.has(c)).collect();
    let mut out = Table::new(keep.iter().map(|s| s.to_string()).collect());
    for r in 0..t.len() {
        out.rows.push(keep.iter().map(|c| t.cell(r, c).cloned().unwrap_or(Value::Null)).collect());
    }
    let n = out.len();
    out.set_column("Status", vec![json!(1); n]);
    out
}

async fn run_topic_massts(app: &AppState, input_tsv: &Path, out_dir: &Path, query_indicator: &str) -> Result<()> {
    let out = tokio::process::Command::new(&app.cfg.python)
        .current_dir(app.cfg.root.join("external/microbe_masst/code"))
        .arg("masst_client.py")
        .args(["--mode", "draw"])
        .arg("--out_file")
        .arg(format!("{}/topic_masst_", out_dir.display()))
        .arg("--input_usi_results_file")
        .arg(input_tsv)
        .args(["--usi_or_lib_id", " "])
        .args(["--compound_name", query_indicator])
        .output()
        .await
        .context("failed to start masst_client.py")?;
    if !out.status.success() {
        bail!(
            "DomainMASSTs failed.\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

pub async fn domainmasst(app: Arc<AppState>, session: SessionRef, ctx: JobCtx, name: String) -> Result<Value> {
    let (table, dir) = {
        let s = session.lock().await;
        (s.data.raw_results.as_ref().and_then(|r| r.get(&name)).cloned(), s.dir.clone())
    };
    let table = table.ok_or_else(|| anyhow!("No results for {name}"))?;
    if table.is_empty() {
        bail!("No results for {name}");
    }
    let input = topic_masst_input(&table);
    if input.columns.len() == 1 {
        bail!("Required columns not present in results (need at least USI).");
    }
    let input_path = dir.join(format!("topicMasst_input_{}.tsv", safe_name(&name)));
    write_tsv(&input_path, &input)?;
    ctx.progress("Running DomainMASSTs…");
    // masst_client puts the name into output file names, so pass a file-safe one
    run_topic_massts(&app, &input_path, &dir, &safe_name(&name)).await?;
    Ok(json!({"message": format!("DomainMASSTs for **{name}** completed.")}))
}

fn mri_key(t: &Table) -> Option<Vec<Option<String>>> {
    let col = if t.has("mri") { "mri" } else if t.has("mri_id_int") { "mri_id_int" } else { return None };
    Some(t.column_values(col).iter().map(|v| if v.is_null() { None } else { Some(value_to_string(v)) }).collect())
}

/// Intersect samples (MRIs) across all molecules, keep the best-cosine row per MRI.
pub fn cooccurrence_table(tables: &IndexMap<String, Table>) -> Result<Table> {
    let mut sets: Vec<HashSet<String>> = Vec::new();
    for t in tables.values() {
        sets.push(mri_key(t).map(|k| k.into_iter().flatten().collect()).unwrap_or_default());
    }
    let mut common: HashSet<String> = sets.first().cloned().unwrap_or_default();
    for s in &sets[1..] {
        common = common.intersection(s).cloned().collect();
    }
    if common.is_empty() {
        bail!("No shared samples across selected molecules.");
    }
    let mut frames = Vec::new();
    for t in tables.values() {
        let mut t = t.clone();
        if !t.has("Cosine") && t.has("cosine") {
            let c = t.col("cosine").unwrap();
            t.columns[c] = "Cosine".into();
        }
        let Some(keys) = mri_key(&t) else { continue };
        t.set_column("mri_key", keys.iter().map(|k| json!(k)).collect());
        let idx: Vec<usize> = keys.iter().enumerate().filter(|(_, k)| k.as_ref().map(|k| common.contains(k)).unwrap_or(false)).map(|(i, _)| i).collect();
        frames.push(t.take_rows(&idx));
    }
    if frames.is_empty() {
        bail!("No rows remained after intersecting by MRI.");
    }
    let all = Table::concat(&frames.iter().collect::<Vec<_>>());
    let cos: Vec<f64> = all.column_values("Cosine").iter().map(|v| value_as_f64(v).unwrap_or(-1.0)).collect();
    let keys: Vec<String> = all.column_values("mri_key").iter().map(value_to_string).collect();
    let mut best: BTreeMap<&String, usize> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        match best.get(k) {
            Some(&b) if cos[b] >= cos[i] => {}
            _ => {
                best.insert(k, i);
            }
        }
    }
    let mut picked = all.take_rows(&best.values().copied().collect::<Vec<_>>());
    picked.drop_column("mri_key");
    if picked.has("Cosine") {
        let c = picked.col("Cosine").unwrap();
        picked.rows.sort_by(|a, b| match (value_as_f64(&a[c]), value_as_f64(&b[c])) {
            (Some(x), Some(y)) => y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
    }
    if !picked.has("USI") {
        bail!("Cannot run DomainMASSTs: required column 'USI' not found.");
    }
    Ok(topic_masst_input(&picked))
}

pub async fn domainmasst_intersection(app: Arc<AppState>, session: SessionRef, ctx: JobCtx) -> Result<Value> {
    let job_name = "moleculeIntersection";
    let (tables, dir) = {
        let s = session.lock().await;
        (s.data.raw_results.clone().unwrap_or_default(), s.dir.clone())
    };
    if tables.len() < 2 || tables.values().any(|t| t.len() <= 1) {
        bail!("Needs results for at least two molecules.");
    }
    let input = cooccurrence_table(&tables)?;
    let out_tsv = dir.join(format!("domainMasst_input_{job_name}.tsv"));
    write_tsv(&out_tsv, &input)?;
    ctx.progress("Running DomainMASSTs…");
    run_topic_massts(&app, &out_tsv, &dir, job_name).await?;
    Ok(json!({"message": "DomainMASSTs for Molecule Intersection completed."}))
}

// ---------------------------------------------------------------------------
// Downstream: LifeMASST
// ---------------------------------------------------------------------------

/// Write the LifeMASST inputs (structuremasst_input.tsv + lifemasst_input_summary.tsv).
/// Returns false when no raw-data summaries could be made.
async fn setup_lifemasst_files(app: &AppState, session_dir: &Path, query_table: &Table, tables: &IndexMap<String, Table>) -> Result<bool> {
    let res = app
        .worker
        .call("lifemasst_setup", json!({"session_dir": session_dir, "query_table": query_table, "redu_tables": tables}))
        .await?;
    Ok(!res["out_path"].is_null())
}

/// "Setup LifeMASST" on the results page, for one query (`name`) or all of them.
pub async fn lifemasst_setup(app: Arc<AppState>, session: SessionRef, ctx: JobCtx, name: Option<String>) -> Result<Value> {
    let (query_table, results, dir) = {
        let s = session.lock().await;
        (s.data.query_table.clone().unwrap_or_default(), s.data.raw_results.clone().unwrap_or_default(), s.dir.clone())
    };
    let (input, tables): (Table, IndexMap<String, Table>) = match &name {
        Some(n) => {
            let c = query_table.col("name");
            let idx: Vec<usize> = (0..query_table.len()).filter(|&r| c.map(|c| value_to_string(&query_table.rows[r][c]).trim() == n).unwrap_or(false)).collect();
            let mut m = IndexMap::new();
            if let Some(t) = results.get(n) {
                m.insert(n.clone(), t.clone());
            }
            (query_table.take_rows(&idx), m)
        }
        None => (query_table, results),
    };
    ctx.progress("Setting up LifeMASSTs…");
    if !setup_lifemasst_files(&app, &dir, &input, &tables).await? {
        bail!("No valid tables to process for LifeMASST.");
    }
    let mut s = session.lock().await;
    s.data.lifemasst_prepared_source = Some("structuremasst".into());
    s.save()?;
    Ok(json!({"message": "LifeMASST setup completed."}))
}

/// Built-in trees and how their tips are matched.
pub struct TreeInfo {
    pub key: &'static str,
    pub match_id: &'static str,
    pub id_prefix: &'static str,
}

pub const TREES: &[TreeInfo] = &[
    TreeInfo { key: "trees/OTL_prepared/labelled_supertree_subset_prepped", match_id: "OpenTreeOfLifeTaxonomyID", id_prefix: "ott" },
    TreeInfo { key: "trees/timetree_prepared/TimeTree_subset_prepped", match_id: "Native_tree_label", id_prefix: "" },
    TreeInfo { key: "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint", match_id: "Native_tree_label", id_prefix: "" },
    TreeInfo { key: "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa", match_id: "Native_tree_label", id_prefix: "" },
    TreeInfo { key: "trees/mammalian_prepared/Foley2022_Concatenation_HRA_neutral_241_10miss_rooted", match_id: "Native_tree_label", id_prefix: "" },
    TreeInfo { key: "trees/fish_prepared/fish_raxml_12k", match_id: "Native_tree_label", id_prefix: "" },
    TreeInfo { key: "trees/fish_prepared/fish_treepl_12k", match_id: "Native_tree_label", id_prefix: "" },
    TreeInfo { key: "trees/plants_prepared/1kp_astral_alltaxa_FAA", match_id: "Native_tree_label", id_prefix: "" },
];

const MIN_SPECIFICITY: &[&str] = &["", "species", "genus", "family", "order", "class", "phylum"];

/// Run a command, streaming its output lines into the job log.
async fn run_logged(ctx: &JobCtx, mut cmd: tokio::process::Command, what: &str) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().with_context(|| format!("failed to start {what}"))?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let c1 = ctx.clone();
    let c2 = ctx.clone();
    let t1 = tokio::spawn(async move {
        let mut l = BufReader::new(out).lines();
        while let Ok(Some(line)) = l.next_line().await {
            c1.log(line);
        }
    });
    let t2 = tokio::spawn(async move {
        let mut l = BufReader::new(err).lines();
        while let Ok(Some(line)) = l.next_line().await {
            c2.log(line);
        }
    });
    let status = child.wait().await?;
    let _ = tokio::join!(t1, t2);
    if !status.success() {
        bail!("{what} failed (exit {}). See the log.", status.code().unwrap_or(-1));
    }
    Ok(())
}

/// First-run reference data preparation (bin/dependency_prep.py), with live log.
pub async fn ensure_ready(app: &AppState, ctx: &JobCtx, tool: &str, force: bool) -> Result<()> {
    let st = app.worker.call("prep_status", json!({"tool": tool})).await?;
    if st["ready"].as_bool() == Some(true) && !force {
        return Ok(());
    }
    let label = st["label"].as_str().unwrap_or(tool).to_string();
    if st["available"].as_bool() == Some(false) || st["missing_shipped"].as_bool() == Some(true) {
        bail!("{}", st["note"].as_str().filter(|s| !s.is_empty()).map(String::from).unwrap_or(format!("{label} is missing files that ship with its repository.")));
    }
    let mb = st["download_mb"].as_f64().unwrap_or(0.0);
    ctx.progress(format!(
        "Preparing {label} reference data{} — this happens once, then every later run reuses it.",
        if mb > 0.0 { format!(" (about {mb:.0} MB)") } else { String::new() }
    ));
    let mut cmd = tokio::process::Command::new(&app.cfg.python);
    cmd.current_dir(&app.cfg.root).arg("-u").arg("bin/dependency_prep.py").arg(tool);
    run_logged(ctx, cmd, &format!("{label} preparation")).await
}

#[derive(Clone, Deserialize)]
pub struct ShortcutSearch {
    pub input: SearchInput,
    pub raw: RawSearchParams,
}

#[derive(Deserialize)]
pub struct LifeRunParams {
    pub tree: String,
    #[serde(default)]
    pub min_specificity: String,
    /// search from the LifeMASST page (instead of the prepared StructureMASST input)
    #[serde(default)]
    pub search: Option<ShortcutSearch>,
}

pub async fn lifemasst_run(app: Arc<AppState>, session: SessionRef, ctx: JobCtx, p: LifeRunParams, upload: Option<Upload>) -> Result<Value> {
    let tree = TREES.iter().find(|t| t.key == p.tree).ok_or_else(|| anyhow!("Unknown tree {}", p.tree))?;
    if !MIN_SPECIFICITY.contains(&p.min_specificity.as_str()) {
        bail!("Unknown minimum specificity {}", p.min_specificity);
    }
    ensure_ready(&app, &ctx, "lifemasst", false).await?;

    let session_dir = session.lock().await.dir.clone();
    let dir = session_dir.join("lifemasst");
    std::fs::create_dir_all(&dir)?;
    let (in_path, out_path) = (dir.join("structuremasst_input.tsv"), dir.join("lifemasst_input_summary.tsv"));

    if let Some(search) = &p.search {
        let (query_table, results) = shortcut_search(&app, &ctx, &search.input, upload, &search.raw, &session_dir).await?;
        ctx.progress("Setting up LifeMASST input…");
        let ok = setup_lifemasst_files(&app, &session_dir, &query_table, &results).await?;
        let mut s = session.lock().await;
        s.data.lifemasst_prepared_source = Some("lifemasst_shortcut".into());
        s.save()?;
        if !ok {
            bail!("LifeMASST setup failed because no valid raw-data summaries could be created.");
        }
    } else if !in_path.exists() || !out_path.exists() {
        bail!("No search input was provided, and no prepared StructureMASST LifeMASST input files were found. Either enter input above or run StructureMASST first.");
    }

    let unique = dir.join("structuremasst_input_unique.tsv");
    app.worker.call("dedup_molecules", json!({"in_path": in_path, "out_path": unique})).await?;

    let abs = |p: PathBuf| std::path::absolute(&p).unwrap_or(p);
    let nf_work = std::env::var("NF_WORK_DIR").map(PathBuf::from).unwrap_or_else(|_| app.cfg.root.join("work/work"));
    let nf_env = std::env::var("NF_CONDA_CACHE").map(PathBuf::from).unwrap_or_else(|_| app.cfg.root.join("work/work_env"));
    std::fs::create_dir_all(&nf_work)?;
    std::fs::create_dir_all(&nf_env)?;
    let override_cfg = dir.join("nf_override.config");
    std::fs::write(
        &override_cfg,
        format!("workDir = \"{}\"\nconda {{\n    enabled = true\n    cacheDir = \"{}\"\n}}\n", abs(nf_work).display(), abs(nf_env).display()),
    )?;
    let level = if tree.key.starts_with("trees/OTL_prepared/") { "NCBIFamily" } else { "NCBISpecies" };
    let data = app.cfg.root.join("external/LifeMASST/data");
    let mut cmd = tokio::process::Command::new(&app.cfg.nextflow);
    cmd.current_dir(&app.cfg.root)
        .arg("run")
        .arg(app.cfg.root.join("external/LifeMASST/nf_workflow.nf"))
        .arg("-c").arg(abs(override_cfg))
        .arg("--input_molecules").arg(abs(unique))
        .arg("--structureMASST_input_file").arg(abs(out_path))
        .arg("--tree_path").arg(abs(data.join(format!("{}.nwk", tree.key))))
        .arg("--tree_features").arg(abs(data.join(format!("{}.tsv", tree.key))))
        .args(["--tax_id", tree.match_id])
        .args(["--tax_id_prefix", tree.id_prefix])
        .args(["--masst_matchLevel", level])
        .args(["--wikidata_matchLevel", level])
        .arg("--output_folder").arg(abs(dir.clone()));
    if !p.min_specificity.is_empty() {
        cmd.args(["--min_specificity", &p.min_specificity]);
    }
    ctx.progress("Running LifeMASST…");
    run_logged(&ctx, cmd, "LifeMASST").await?;
    Ok(json!({"message": "LifeMASST completed."}))
}

#[derive(Serialize)]
pub struct LifeStatus {
    pub has_results: bool,
    pub prepared_from_structuremasst: bool,
    pub tree_heatmap: bool,
    pub metadata: Option<Table>,
    pub empress_zip: bool,
}

pub fn read_tsv(path: &Path) -> Option<Table> {
    let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t').flexible(true).from_path(path).ok()?;
    let columns: Vec<String> = rdr.headers().ok()?.iter().map(String::from).collect();
    let rows = rdr
        .records()
        .filter_map(|r| r.ok())
        .map(|r| r.iter().map(|s| if s.is_empty() { Value::Null } else { json!(s) }).collect())
        .collect();
    Some(Table { columns, rows })
}

// ---------------------------------------------------------------------------
// Downstream: GeoMASST
// ---------------------------------------------------------------------------

pub const ALL_MOLECULES: &str = "All molecules";

#[derive(Deserialize)]
pub struct GeoParams {
    /// "results" (this session's StructureMASST results) or "search"
    pub source: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_max_maps")]
    pub max_maps: i64,
    #[serde(default)]
    pub search: Option<ShortcutSearch>,
}

fn default_max_maps() -> i64 {
    5
}

fn concat_results(results: &IndexMap<String, Table>) -> (Table, usize) {
    let mut frames = Vec::new();
    for (name, t) in results {
        if t.is_empty() {
            continue;
        }
        let mut t = t.clone();
        if !t.has("query_name") {
            let n = t.len();
            t.set_column("query_name", vec![json!(name); n]);
        }
        frames.push(t);
    }
    let n = frames.len();
    (Table::concat(&frames.iter().collect::<Vec<_>>()), n)
}

/// SMILES behind each result name (structures_for_results).
fn structures_for_results(query_table: Option<&Table>, query_by_name: &IndexMap<String, String>) -> IndexMap<String, String> {
    let mut out = IndexMap::new();
    if let Some(t) = query_table {
        for r in 0..t.len() {
            let get = |c: &str| t.cell(r, c).filter(|v| !v.is_null()).map(value_to_string).unwrap_or_default().trim().to_string();
            let mut smiles = get("original_smiles");
            if smiles.is_empty() && get("type") == "smiles" {
                smiles = get("query");
            }
            if !smiles.is_empty() {
                out.insert(get("name"), smiles);
            }
        }
    }
    for (n, s) in query_by_name {
        if !s.is_empty() {
            out.insert(n.clone(), s.clone());
        }
    }
    out
}

pub async fn geomasst(app: Arc<AppState>, session: SessionRef, ctx: JobCtx, p: GeoParams, upload: Option<Upload>) -> Result<Value> {
    let max_maps = p.max_maps.clamp(1, 20);
    ensure_ready(&app, &ctx, "geomasst", false).await?;
    let session_dir = session.lock().await.dir.clone();
    let mut caption = None;
    let (table, compound_name, structures) = if p.source == "search" {
        let search = p.search.as_ref().ok_or_else(|| anyhow!("Enter a name, SMILES/SMARTS, class or USI to search for."))?;
        let (query_table, results) = shortcut_search(&app, &ctx, &search.input, upload, &search.raw, &session_dir).await?;
        let (table, n) = concat_results(&results);
        if table.is_empty() {
            bail!("No raw-data matches for that search.");
        }
        caption = Some(format!("{} matched samples across {n} molecule(s).", table.len()));
        let mut structures = IndexMap::new();
        for r in 0..query_table.len() {
            let get = |c: &str| query_table.cell(r, c).filter(|v| !v.is_null()).map(value_to_string).unwrap_or_default();
            if get("type") == "smiles" {
                let s = get("original_smiles");
                structures.insert(get("name"), if s.is_empty() { get("query") } else { s });
            }
        }
        let name = if query_table.len() == 1 { query_table.cell(0, "name").map(value_to_string).unwrap_or_default() } else { ALL_MOLECULES.to_string() };
        (table, name, structures)
    } else {
        let s = session.lock().await;
        let results = s.data.raw_results.clone().unwrap_or_default();
        let name = p.name.clone().unwrap_or_else(|| ALL_MOLECULES.into());
        let table = if name == ALL_MOLECULES { concat_results(&results).0 } else { results.get(&name).cloned().ok_or_else(|| anyhow!("No results for {name}"))? };
        (table, name, structures_for_results(s.data.query_table.as_ref(), &s.data.query_by_name))
    };
    ctx.progress("Running GeoMASST…");
    let out = session_dir.join("geomasst.html");
    app.worker
        .call(
            "geomasst_html",
            json!({"table": table, "compound_name": compound_name, "max_molecule_maps": max_maps, "structures": structures, "out_path": out}),
        )
        .await?;
    Ok(json!({"file": "geomasst.html", "caption": caption}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(cols: &[&str], rows: Vec<Vec<Value>>) -> Table {
        Table { columns: cols.iter().map(|s| s.to_string()).collect(), rows }
    }

    #[test]
    fn dereplicate_keeps_closest_to_representative() {
        let tab = t(
            &["spectrum_id_int", "representative_spectrum_int", "x"],
            vec![vec![json!(12), json!(10), json!("b")], vec![json!(10), json!(10), Value::Null], vec![json!(20), json!(20), json!("c")]],
        );
        let d = dereplicate(&tab);
        assert_eq!(d.columns[0], "representative_spectrum_int");
        assert_eq!(d.len(), 2);
        assert_eq!(d.cell(0, "x"), Some(&json!("b")));
        assert_eq!(d.cell(0, "similar_library_spectra"), Some(&json!(2)));
        assert_eq!(d.cell(0, "spectrum_id_int"), Some(&json!(10)));
    }

    #[test]
    fn dereplicate_passes_usi_rows_through() {
        let tab = t(&["query_spectrum_id", "USI"], vec![vec![json!("mzspec:X:y:scan:1"), json!("mzspec:X:y:scan:1")]]);
        let d = dereplicate(&tab);
        assert_eq!(d.len(), 1);
        assert_eq!(d.cell(0, "similar_library_spectra"), Some(&json!(1)));
        assert_eq!(d.cell(0, "unique_spectra_in_mri"), Some(&json!(1)));
    }

    #[test]
    fn modified_sorting() {
        let tab = t(&["mri", "Modified"], vec![vec![json!("a"), json!("No")], vec![json!("b"), json!("addition")], vec![json!("c"), json!("weird")]]);
        let r = arrange_results(&tab);
        assert_eq!(r.column_values("Modified"), vec![json!("addition"), json!("no"), Value::Null]);
    }

    #[test]
    fn cooccurrence_picks_best_cosine() {
        let a = t(&["mri", "USI", "Cosine"], vec![vec![json!("m1"), json!("u1"), json!(0.8)], vec![json!("m2"), json!("u2"), json!(0.9)]]);
        let b = t(&["mri", "USI", "Cosine"], vec![vec![json!("m1"), json!("u3"), json!(0.95)]]);
        let mut m = IndexMap::new();
        m.insert("a".to_string(), a);
        m.insert("b".to_string(), b);
        let c = cooccurrence_table(&m).unwrap();
        assert_eq!(c.columns, vec!["USI", "Cosine", "Status"]);
        assert_eq!(c.rows, vec![vec![json!("u3"), json!(0.95), json!(1)]]);
    }

    #[test]
    fn analogue_filter_keeps_parents_and_their_datasets() {
        let tab = t(
            &["Unit Delta Mass", "ATTRIBUTE_DatasetAccession"],
            vec![
                vec![json!(0), json!("A")],
                vec![json!(16), json!("A")],
                vec![json!(16), json!("B")],
                vec![Value::Null, json!("C")],
            ],
        );
        let f = filter_analogues_to_parent_datasets(&tab);
        assert_eq!(f.len(), 3);
        assert!(!f.rows.iter().any(|r| r[1] == json!("B")));
    }

    #[test]
    fn batch_names_are_sanitized_like_pandas() {
        assert_eq!(sanitize_batch_name("caffeine (1)", Flavor::Main), "caffeine__1_");
        assert_eq!(sanitize_batch_name("caffeine (1)", Flavor::Shortcut), "caffeine__1");
    }

    #[test]
    fn apply_to_all_molecules() {
        let a = t(&["NCBIDivision"], vec![vec![json!("Primates")], vec![json!("Rodents")]]);
        let b = t(&["NCBIDivision"], vec![vec![json!("Rodents")], vec![Value::Null], vec![json!("Primates")]]);
        let mut m = IndexMap::new();
        m.insert("a".to_string(), a);
        m.insert("b".to_string(), b);
        let vals = vec!["Primates".to_string()];
        result_action(&mut m, "a", RowAction::Keep, &[], Some(("NCBIDivision", &vals)), true).unwrap();
        assert_eq!(m["a"].len(), 1);
        assert_eq!(m["b"].rows, vec![vec![json!("Primates")]]);
    }

    /// Queries with their molecules' InChIKeys, and raw data matches per query (as InChIKeys).
    fn session_with(groups: Vec<(&str, Vec<&str>)>, raw: Vec<(&str, Vec<&str>)>) -> crate::session::SessionData {
        let mut d = crate::session::SessionData::default();
        let mut qt = t(&["query", "name"], vec![]);
        for (name, iks) in &groups {
            let ov = t(&["Compound_Name", "inchikey_first_block"], iks.iter().map(|ik| vec![json!(format!("mol {ik}")), json!(ik)]).collect());
            let spectra = iks.iter().map(|ik| (ik.to_string(), t(&["spectrum_id"], vec![vec![json!(format!("s_{ik}"))]]))).collect();
            d.groups.insert(name.to_string(), Group { overview: ov, spectra });
            d.query_by_name.insert(name.to_string(), format!("q_{name}"));
            qt.rows.push(vec![json!(format!("q_{name}")), json!(name)]);
        }
        d.query_table = Some(qt);
        d.raw_results = Some(
            raw.into_iter()
                .map(|(name, iks)| (name.to_string(), t(&["mri_id_int", "inchikey_first_block"], iks.iter().enumerate().map(|(i, ik)| vec![json!(i), json!(ik)]).collect())))
                .collect(),
        );
        d
    }

    #[test]
    fn remove_molecules_without_raw_data_matches() {
        // a: matched; b: no matches; c: three molecules of which only X matched
        let mut d = session_with(
            vec![("a", vec!["A"]), ("b", vec!["B"]), ("c", vec!["X", "Y", "Z"])],
            vec![("a", vec!["A"]), ("b", vec![]), ("c", vec!["X", "X"])],
        );
        assert_eq!(unmatched(&d).molecules(&d), 3);
        assert_eq!(remove_unmatched(&mut d), 3);
        assert_eq!(d.groups.keys().collect::<Vec<_>>(), ["a", "c"]);
        assert_eq!(d.groups["c"].overview.len(), 1);
        assert_eq!(d.groups["c"].spectra.keys().collect::<Vec<_>>(), ["X"]);
        assert!(!d.raw_results.as_ref().unwrap().contains_key("b"));
        assert!(!d.query_by_name.contains_key("b"));
        assert_eq!(d.query_table.as_ref().unwrap().len(), 2);
        assert_eq!(unmatched(&d).molecules(&d), 0);
    }

    #[test]
    fn a_query_with_matches_keeps_its_molecules_when_keys_do_not_line_up() {
        // like a USI query: the matches carry no key the molecule overview knows
        let d = session_with(vec![("u", vec!["K1", "K2"])], vec![("u", vec!["other"])]);
        assert_eq!(unmatched(&d).molecules(&d), 0);
    }

    #[test]
    fn removing_every_molecule_of_a_query_removes_the_query() {
        let mut d = session_with(vec![("a", vec!["A"]), ("b", vec!["B1", "B2"])], vec![("a", vec!["A"]), ("b", vec!["B1"])]);
        molecule_action(&mut d, "b", RowAction::Remove, &[0, 1]).unwrap();
        assert!(!d.groups.contains_key("b"));
        assert!(!d.raw_results.as_ref().unwrap().contains_key("b"));
        molecule_action(&mut d, "a", RowAction::Keep, &[0]).unwrap();
        assert!(d.groups.contains_key("a"));
    }
}
