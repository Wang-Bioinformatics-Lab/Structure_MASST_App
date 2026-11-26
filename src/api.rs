//! JSON API used by the browser UI.

use axum::body::Body;
use axum::extract::{FromRequestParts, Multipart, Path as AxPath, Query, State};
use axum::http::{header, request::Parts, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::external;
use crate::plots;
use crate::session::SessionRef;
use crate::table::Table;
use crate::validate;
use crate::workflows::{self, RowAction, Upload};
use crate::AppState;

// ---------------------------------------------------------------------------
// Errors and the session extractor
// ---------------------------------------------------------------------------

pub struct ApiError(StatusCode, String);

impl ApiError {
    fn bad(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::BAD_REQUEST, msg.into())
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(StatusCode::BAD_REQUEST, format!("{:#}", e.into()))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

/// Session from the `X-Session-Id` header (or `?sid=` for links/iframes).
pub struct Sess(pub String, pub SessionRef);

impl FromRequestParts<Arc<AppState>> for Sess {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, app: &Arc<AppState>) -> Result<Self, Self::Rejection> {
        let from_header = parts.headers.get("x-session-id").and_then(|v| v.to_str().ok()).map(String::from);
        // ?sid= only for downloading session files (links, iframes); every other
        // call must send the header, which cross-site forms cannot set
        let file_get = parts.method == axum::http::Method::GET && parts.uri.path().starts_with("/api/files/");
        let from_query = if file_get {
            parts.uri.query().and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("sid=").map(String::from)))
        } else {
            None
        };
        let sid = from_header.or(from_query).ok_or_else(|| ApiError(StatusCode::UNAUTHORIZED, "missing session".into()))?;
        let s = app.sessions.get(&sid).ok_or_else(|| ApiError(StatusCode::UNAUTHORIZED, "invalid session".into()))?;
        Ok(Sess(sid, s))
    }
}

/// Largest batch file accepted (CSV / MGF). Other requests keep axum's 2 MB default.
const UPLOAD_LIMIT: usize = 25 * 1024 * 1024;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/health", get(|| async { Json(json!({"ok": true})) }))
        .route("/api/session", post(new_session))
        .route("/api/state", get(state))
        .route("/api/classes", get(classes))
        .route("/api/pubchem/suggest", get(pubchem_suggest))
        .route("/api/pubchem/resolve", get(pubchem_resolve))
        .route("/api/preview", post(preview))
        .route("/api/render", get(render))
        .route("/api/search/library", post(search_library).layer(axum::extract::DefaultBodyLimit::max(UPLOAD_LIMIT)))
        .route("/api/jobs/{id}", get(job))
        .route("/api/library/molecules", post(library_molecules))
        .route("/api/library/spectra", post(library_spectra))
        .route("/api/search/raw", post(search_raw))
        .route("/api/results/rows", post(result_rows))
        .route("/api/results/remove-unmatched", post(remove_unmatched))
        .route("/api/results/values", get(result_values))
        .route("/api/results/sankey", post(result_sankey))
        .route("/api/domainmasst", post(domainmasst))
        .route("/api/domainmasst/files", get(domainmasst_files))
        .route("/api/prep/{tool}", get(prep_status).post(prep_run))
        .route("/api/lifemasst/setup", post(lifemasst_setup))
        .route("/api/lifemasst/status", get(lifemasst_status))
        .route("/api/lifemasst/run", post(lifemasst_run).layer(axum::extract::DefaultBodyLimit::max(UPLOAD_LIMIT)))
        .route("/api/geomasst", post(geomasst).layer(axum::extract::DefaultBodyLimit::max(UPLOAD_LIMIT)))
        .route("/api/files/{*path}", get(session_file))
}

// ---------------------------------------------------------------------------
// Session + state
// ---------------------------------------------------------------------------

/// A new session, or the one asked for with `{"resume": "<id>"}` while the server
/// still has it. Links carry their session (?session=<id>); one that is gone or
/// malformed is ignored and a new session is started instead.
async fn new_session(State(app): State<Arc<AppState>>, body: axum::body::Bytes) -> ApiResult<Json<Value>> {
    let wanted = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("resume").and_then(Value::as_str).map(str::to_owned));
    if let Some(id) = wanted.filter(|id| app.sessions.exists(id)) {
        return Ok(Json(json!({"sid": id, "resumed": true})));
    }
    Ok(Json(json!({"sid": app.sessions.create()?, "resumed": false})))
}

async fn state(Sess(sid, s): Sess) -> ApiResult<Json<Value>> {
    let s = s.lock().await;
    let groups: Vec<Value> = s
        .data
        .groups
        .iter()
        .map(|(name, g)| {
            let all = Table::concat(&g.spectra.values().collect::<Vec<_>>());
            json!({
                "name": name,
                "overview": g.overview,
                "spectra": g.spectra.iter().map(|(ik, t)| json!({"ik": ik, "table": t})).collect::<Vec<_>>(),
                "total_spectra": all.len(),
                "sankey": if all.is_empty() { None } else { plots::library_sankey(&all) },
            })
        })
        .collect();
    let raw = s.data.raw_results.as_ref().map(|r| r.iter().map(|(name, t)| json!({"name": name, "table": t})).collect::<Vec<_>>());
    Ok(Json(json!({
        "sid": sid,
        "query_table": s.data.query_table,
        "groups": groups,
        "direct_spectra": workflows::has_direct_spectra(&s.data.groups, &s.data.mgf_store),
        "raw_mode": s.data.raw_mode,
        "raw_results": raw,
        "unmatched_molecules": workflows::unmatched(&s.data).molecules(&s.data),
    })))
}

/// ClassyFire class labels for the "Class Label" dropdown (loaded once).
async fn classes(State(app): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let labels = app
        .classes
        .get_or_try_init(|| async { app.worker.call_as::<Vec<String>>("molecule_classes", json!({})).await })
        .await?;
    Ok(Json(json!({"classes": labels})))
}

// ---------------------------------------------------------------------------
// Structure input helpers
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SuggestQ {
    q: String,
}

async fn pubchem_suggest(State(app): State<Arc<AppState>>, Query(q): Query<SuggestQ>) -> ApiResult<Json<Value>> {
    validate::name(&q.q)?;
    let suggestions = external::pubchem_autocomplete(&app.http, q.q.trim()).await;
    let (smiles, warning) = match suggestions.first() {
        Some(top) => external::resolve_name(&app.http, top).await,
        None => (String::new(), None),
    };
    Ok(Json(json!({"suggestions": suggestions, "smiles": smiles, "warning": warning})))
}

#[derive(Deserialize)]
struct ResolveQ {
    name: String,
}

async fn pubchem_resolve(State(app): State<Arc<AppState>>, Query(q): Query<ResolveQ>) -> ApiResult<Json<Value>> {
    validate::name(&q.name)?;
    let (smiles, warning) = external::resolve_name(&app.http, &q.name).await;
    Ok(Json(json!({"smiles": smiles, "warning": warning})))
}

#[derive(Deserialize)]
struct PreviewReq {
    text: String,
}

/// Detect SMILES vs SMARTS and return a structure image.
async fn preview(State(app): State<Arc<AppState>>, Json(req): Json<PreviewReq>) -> ApiResult<Json<Value>> {
    use base64::Engine;
    let text = req.text.trim().to_string();
    if text.is_empty() {
        return Ok(Json(json!({"type": null})));
    }
    validate::structure(&text)?;
    let types: Vec<String> = app.worker.call_as("detect", json!({"items": [text]})).await?;
    let typ = types.into_iter().next().unwrap_or_else(|| "Invalid".into());
    let b64 = |png: Vec<u8>| format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png));
    let (image, warning) = match typ.as_str() {
        "smiles" => match app.render_png(&text, 300).await {
            Some(png) => (Some(b64(png)), None),
            None => (None, Some("Could not parse SMILES for rendering.".to_string())),
        },
        "smarts" => match external::smarts_image(&app.http, &text, &app.cfg.smarts_api_key, &crate::session::new_sid()).await {
            Some(png) => (Some(b64(png)), None),
            None => (None, Some("Failed to retrieve SMARTS image from the API. You can still go ahead.".to_string())),
        },
        _ => (None, Some("Not a valid SMILES/SMARTS".to_string())),
    };
    Ok(Json(json!({"type": typ, "image": image, "warning": warning})))
}

#[derive(Deserialize)]
struct RenderQ {
    smiles: String,
    size: Option<u32>,
}

/// PNG of a SMILES (molecule overview table images).
async fn render(State(app): State<Arc<AppState>>, Query(q): Query<RenderQ>) -> Response {
    let size = q.size.unwrap_or(500).clamp(50, 1000);
    if validate::structure(&q.smiles).is_err() {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match app.render_png(&q.smiles, size).await {
        Some(png) => (
            [(header::CONTENT_TYPE, HeaderValue::from_static("image/png")), (header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=86400"))],
            png,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

// ---------------------------------------------------------------------------
// Multipart: a JSON field plus an optional batch file
// ---------------------------------------------------------------------------

fn multipart_error(e: axum::extract::multipart::MultipartError) -> ApiError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError(StatusCode::PAYLOAD_TOO_LARGE, format!("The file is too large (max {} MB).", UPLOAD_LIMIT / 1024 / 1024))
    } else {
        ApiError::bad(e.body_text())
    }
}

async fn read_multipart<T: DeserializeOwned>(mut mp: Multipart, field: &str) -> ApiResult<(T, Option<Upload>)> {
    let mut json_text = None;
    let mut upload = None;
    while let Some(f) = mp.next_field().await.map_err(multipart_error)? {
        let name = f.name().unwrap_or("").to_string();
        if name == "file" {
            let filename = f.file_name().unwrap_or("upload.csv").to_string();
            let bytes = f.bytes().await.map_err(multipart_error)?;
            if !bytes.is_empty() {
                upload = Some(Upload { filename, bytes: bytes.to_vec() });
            }
        } else if name == field {
            json_text = Some(f.text().await.map_err(multipart_error)?);
        }
    }
    let text = json_text.ok_or_else(|| ApiError::bad(format!("missing field {field}")))?;
    let value: T = serde_json::from_str(&text).map_err(|e| ApiError::bad(format!("invalid {field}: {e}")))?;
    Ok((value, upload))
}

// ---------------------------------------------------------------------------
// Library search
// ---------------------------------------------------------------------------

async fn search_library(State(app): State<Arc<AppState>>, Sess(sid, s): Sess, mp: Multipart) -> ApiResult<Json<Value>> {
    let (input, upload): (workflows::SearchInput, _) = read_multipart(mp, "search").await?;
    if let Some(id) = app.jobs.running(&sid, "library") {
        return Ok(Json(json!({"job": id})));
    }
    external::track_event(&app.http, "Get Available Spectra Clicked");
    let app2 = app.clone();
    let id = app.jobs.spawn(&sid, "library", move |ctx| workflows::library_search(app2, s, ctx, input, upload));
    Ok(Json(json!({"job": id})))
}

async fn job(State(app): State<Arc<AppState>>, Sess(sid, _): Sess, AxPath(id): AxPath<String>) -> ApiResult<Json<Value>> {
    let v = app.jobs.view(&sid, &id).ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "unknown job".into()))?;
    Ok(Json(serde_json::to_value(v)?))
}

#[derive(Deserialize)]
struct RowsReq {
    name: String,
    #[serde(default)]
    ik: Option<String>,
    action: RowAction,
    #[serde(default)]
    rows: Vec<usize>,
    /// select by column value(s) instead of rows
    #[serde(default)]
    column: Option<String>,
    #[serde(default)]
    values: Vec<String>,
    #[serde(default)]
    apply_all: bool,
}

async fn library_molecules(Sess(_, s): Sess, Json(req): Json<RowsReq>) -> ApiResult<Json<Value>> {
    let mut s = s.lock().await;
    workflows::molecule_action(&mut s.data, &req.name, req.action, &req.rows)?;
    s.save()?;
    Ok(Json(json!({"ok": true})))
}

/// "Remove molecules without raw data matches" (library entries and results).
async fn remove_unmatched(Sess(_, s): Sess) -> ApiResult<Json<Value>> {
    let mut s = s.lock().await;
    let removed = workflows::remove_unmatched(&mut s.data);
    s.save()?;
    Ok(Json(json!({"removed": removed})))
}

async fn library_spectra(Sess(_, s): Sess, Json(req): Json<RowsReq>) -> ApiResult<Json<Value>> {
    let ik = req.ik.clone().ok_or_else(|| ApiError::bad("missing ik"))?;
    let mut s = s.lock().await;
    workflows::spectra_action(&mut s.data.groups, &req.name, &ik, req.action, &req.rows)?;
    s.save()?;
    Ok(Json(json!({"ok": true})))
}

// ---------------------------------------------------------------------------
// Raw data search + results
// ---------------------------------------------------------------------------

async fn search_raw(State(app): State<Arc<AppState>>, Sess(sid, s): Sess, Json(p): Json<workflows::RawSearchParams>) -> ApiResult<Json<Value>> {
    if !["FASSTrecords", "FASST"].contains(&p.mode.as_str()) {
        return Err(ApiError::bad("invalid mode"));
    }
    if let Some(id) = app.jobs.running(&sid, "raw") {
        return Ok(Json(json!({"job": id})));
    }
    external::track_event(&app.http, "Do Search Button Clicked");
    let app2 = app.clone();
    let id = app.jobs.spawn(&sid, "raw", move |ctx| workflows::raw_search(app2, s, ctx, p));
    Ok(Json(json!({"job": id})))
}

async fn result_rows(Sess(_, s): Sess, Json(req): Json<RowsReq>) -> ApiResult<Json<Value>> {
    let mut s = s.lock().await;
    let results = s.data.raw_results.as_mut().ok_or_else(|| ApiError::bad("no results"))?;
    let by_value = req.column.as_deref().filter(|c| *c != "None" && !req.values.is_empty()).map(|c| (c, req.values.as_slice()));
    workflows::result_action(results, &req.name, req.action, &req.rows, by_value, req.apply_all)?;
    s.save()?;
    Ok(Json(json!({"ok": true})))
}

#[derive(Deserialize)]
struct ValuesQ {
    name: String,
    col: String,
}

async fn result_values(Sess(_, s): Sess, Query(q): Query<ValuesQ>) -> ApiResult<Json<Value>> {
    let s = s.lock().await;
    let t = s.data.raw_results.as_ref().and_then(|r| r.get(&q.name)).ok_or_else(|| ApiError::bad("no results"))?;
    Ok(Json(json!({"values": workflows::column_values(t, &q.col)})))
}

#[derive(Deserialize)]
struct SankeyReq {
    name: String,
    cols: Vec<String>,
}

async fn result_sankey(Sess(_, s): Sess, Json(req): Json<SankeyReq>) -> ApiResult<Json<Value>> {
    if req.cols.len() != 4 {
        return Err(ApiError::bad("need four columns"));
    }
    let s = s.lock().await;
    let t = s.data.raw_results.as_ref().and_then(|r| r.get(&req.name)).ok_or_else(|| ApiError::bad("no results"))?;
    Ok(Json(plots::raw_data_sankey(t, &req.cols)))
}

// ---------------------------------------------------------------------------
// Downstream tools
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct NameReq {
    #[serde(default)]
    name: Option<String>,
}

async fn domainmasst(State(app): State<Arc<AppState>>, Sess(sid, s): Sess, Json(req): Json<NameReq>) -> ApiResult<Json<Value>> {
    let app2 = app.clone();
    let id = match req.name {
        Some(name) => app.jobs.spawn(&sid, "domainmasst", move |ctx| workflows::domainmasst(app2, s, ctx, name)),
        None => app.jobs.spawn(&sid, "domainmasst", move |ctx| workflows::domainmasst_intersection(app2, s, ctx)),
    };
    Ok(Json(json!({"job": id})))
}

async fn domainmasst_files(Sess(_, s): Sess) -> ApiResult<Json<Value>> {
    let dir = s.lock().await.dir.clone();
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| n.ends_with(".html") && n.starts_with("topic_masst_"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    Ok(Json(json!({"files": files})))
}

fn check_tool(tool: &str) -> ApiResult<()> {
    if ["lifemasst", "geomasst"].contains(&tool) { Ok(()) } else { Err(ApiError::bad("unknown tool")) }
}

/// Reference data panel ("what this checkout has, and what it still needs").
async fn prep_status(State(app): State<Arc<AppState>>, AxPath(tool): AxPath<String>) -> ApiResult<Json<Value>> {
    check_tool(&tool)?;
    Ok(Json(app.worker.call("prep_status", json!({"tool": tool})).await?))
}

/// "Prepare … data now".
async fn prep_run(State(app): State<Arc<AppState>>, Sess(sid, _): Sess, AxPath(tool): AxPath<String>) -> ApiResult<Json<Value>> {
    check_tool(&tool)?;
    let app2 = app.clone();
    let id = app.jobs.spawn(&sid, &format!("prep_{tool}"), move |ctx| async move {
        workflows::ensure_ready(&app2, &ctx, &tool, true).await?;
        Ok(json!({"message": "Reference data is ready."}))
    });
    Ok(Json(json!({"job": id})))
}

async fn lifemasst_setup(State(app): State<Arc<AppState>>, Sess(sid, s): Sess, Json(req): Json<NameReq>) -> ApiResult<Json<Value>> {
    let app2 = app.clone();
    let id = app.jobs.spawn(&sid, "lifemasst_setup", move |ctx| workflows::lifemasst_setup(app2, s, ctx, req.name));
    Ok(Json(json!({"job": id})))
}

async fn lifemasst_status(Sess(_, s): Sess) -> ApiResult<Json<Value>> {
    let s = s.lock().await;
    let dir = s.dir.join("lifemasst");
    let files_exist = dir.join("structuremasst_input.tsv").exists() && dir.join("lifemasst_input_summary.tsv").exists();
    let status = workflows::LifeStatus {
        has_results: s.data.raw_results.as_ref().map(|r| !r.is_empty()).unwrap_or(false),
        prepared_from_structuremasst: files_exist && s.data.lifemasst_prepared_source.as_deref() == Some("structuremasst"),
        tree_heatmap: dir.join("tree_heatmap.html").exists(),
        metadata: workflows::read_tsv(&dir.join("merged_metadata.tsv")),
        empress_zip: dir.join("empress_results.zip").exists(),
    };
    Ok(Json(serde_json::to_value(status)?))
}

async fn lifemasst_run(State(app): State<Arc<AppState>>, Sess(sid, s): Sess, mp: Multipart) -> ApiResult<Json<Value>> {
    let (p, upload): (workflows::LifeRunParams, _) = read_multipart(mp, "params").await?;
    if let Some(id) = app.jobs.running(&sid, "lifemasst") {
        return Ok(Json(json!({"job": id})));
    }
    external::track_event(&app.http, "LifeMASST Button Clicked");
    let app2 = app.clone();
    let id = app.jobs.spawn(&sid, "lifemasst", move |ctx| workflows::lifemasst_run(app2, s, ctx, p, upload));
    Ok(Json(json!({"job": id})))
}

async fn geomasst(State(app): State<Arc<AppState>>, Sess(sid, s): Sess, mp: Multipart) -> ApiResult<Json<Value>> {
    let (p, upload): (workflows::GeoParams, _) = read_multipart(mp, "params").await?;
    if !["results", "search"].contains(&p.source.as_str()) {
        return Err(ApiError::bad("invalid source"));
    }
    external::track_event(&app.http, "GeoMASST Button Clicked");
    let app2 = app.clone();
    let id = app.jobs.spawn(&sid, "geomasst", move |ctx| workflows::geomasst(app2, s, ctx, p, upload));
    Ok(Json(json!({"job": id})))
}

/// Files produced in the session folder (DomainMASST/GeoMASST html, LifeMASST outputs).
async fn session_file(Sess(_, s): Sess, AxPath(path): AxPath<String>) -> Response {
    let dir = s.lock().await.dir.clone();
    let rel = std::path::Path::new(&path);
    if rel.components().any(|c| !matches!(c, std::path::Component::Normal(_))) || path.ends_with("state.json") {
        return StatusCode::NOT_FOUND.into_response();
    }
    let full = dir.join(rel);
    let Ok(bytes) = tokio::fs::read(&full).await else { return StatusCode::NOT_FOUND.into_response() };
    let ctype = match full.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "zip" => "application/zip",
        "tsv" | "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    };
    let mut resp = Response::new(Body::from(bytes));
    resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(ctype));
    if ctype.starts_with("text/html") || ctype == "image/svg+xml" {
        // Reports are generated from user input (names, queries). Serve them in a
        // sandbox with an opaque origin: their scripts run, but cannot read this
        // app's session or call its API.
        resp.headers_mut().insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static("sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox allow-downloads"),
        );
    }
    if ctype == "application/zip" {
        let fname = full.file_name().and_then(|n| n.to_str()).unwrap_or("download.zip");
        if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{fname}\"")) {
            resp.headers_mut().insert(header::CONTENT_DISPOSITION, v);
        }
    }
    resp
}
