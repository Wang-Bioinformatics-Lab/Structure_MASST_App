//! StructureMASST web server.
//!
//! Serves the browser UI (static/), the JSON API (src/api.rs) and runs the
//! chemistry/pandas work in a Python compute worker (bin/compute_worker.py).

mod api;
mod external;
mod jobs;
mod links;
mod plots;
mod session;
mod table;
mod validate;
mod worker;
mod workflows;

use axum::routing::get_service;
use axum::Router;
use base64::Engine;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower_http::compression::CompressionLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::trace::TraceLayer;
use axum::http::{header, HeaderValue};

/// Content Security Policy for the app pages: only this server's scripts, the
/// pinned CDN libraries (Tabulator, Plotly) and the GNPS analytics may run.
const PAGE_CSP: &str = "default-src 'self'; \
script-src 'self' https://unpkg.com https://cdn.jsdelivr.net https://analytics.gnps2.org https://analytics-api.gnps2.org; \
style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; \
font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; \
connect-src 'self' https://analytics.gnps2.org https://analytics-api.gnps2.org; \
frame-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'self'";

pub struct Config {
    pub root: PathBuf,
    pub python: String,
    pub nextflow: String,
    pub host: String,
    pub port: u16,
    pub session_dir: PathBuf,
    pub smarts_api_key: String,
}

impl Config {
    fn from_env() -> Self {
        let root = std::env::var("STRUCTUREMASST_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().expect("cwd"));
        // keys.env holds SMARTS_API_KEY (same file the Streamlit app read)
        let _ = dotenvy::from_path(root.join("keys.env"));
        Config {
            python: std::env::var("STRUCTUREMASST_PYTHON").unwrap_or_else(|_| "python".into()),
            nextflow: std::env::var("STRUCTUREMASST_NEXTFLOW").unwrap_or_else(|_| "nextflow".into()),
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(5000),
            session_dir: std::env::var("SESSION_DIR").map(PathBuf::from).unwrap_or_else(|_| root.join("sessionoutput")),
            smarts_api_key: std::env::var("SMARTS_API_KEY").unwrap_or_default(),
            root,
        }
    }
}

pub struct AppState {
    pub cfg: Config,
    pub sessions: session::SessionStore,
    pub jobs: jobs::JobStore,
    pub worker: worker::PyWorker,
    pub http: reqwest::Client,
    /// ClassyFire class labels (loaded once from the worker)
    pub classes: tokio::sync::OnceCell<Vec<String>>,
    render_cache: Mutex<HashMap<(String, u32), Vec<u8>>>,
}

impl AppState {
    /// RDKit depiction of a SMILES (cached).
    pub async fn render_png(&self, smiles: &str, size: u32) -> Option<Vec<u8>> {
        let key = (smiles.to_string(), size);
        if let Some(png) = self.render_cache.lock().unwrap().get(&key) {
            return Some(png.clone());
        }
        let v = self.worker.call("render", serde_json::json!({"smiles": smiles, "size": size})).await.ok()?;
        let png = base64::engine::general_purpose::STANDARD.decode(v.as_str()?).ok()?;
        let mut cache = self.render_cache.lock().unwrap();
        if cache.len() > 5000 {
            cache.clear();
        }
        cache.insert(key, png.clone());
        Some(png)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=warn".into()))
        .init();

    let cfg = Config::from_env();
    std::fs::create_dir_all(&cfg.session_dir)?;
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let static_dir = cfg.root.join("static");

    let state = Arc::new(AppState {
        sessions: session::SessionStore::new(cfg.session_dir.clone()),
        jobs: jobs::JobStore::default(),
        worker: worker::PyWorker::new(cfg.python.clone(), cfg.root.clone()),
        http: reqwest::Client::builder().build()?,
        classes: tokio::sync::OnceCell::new(),
        render_cache: Mutex::new(HashMap::new()),
        cfg,
    });
    state.worker.start().await?;

    // Housekeeping: drop idle sessions from memory and old finished jobs.
    {
        let st = state.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(300));
            loop {
                tick.tick().await;
                st.sessions.evict_idle(Duration::from_secs(4 * 3600));
                st.jobs.cleanup(Duration::from_secs(3600));
            }
        });
    }

    let page = |f: &str| {
        use tower::Layer;
        let csp = SetResponseHeaderLayer::overriding(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(PAGE_CSP));
        let xfo = SetResponseHeaderLayer::overriding(header::X_FRAME_OPTIONS, HeaderValue::from_static("SAMEORIGIN"));
        get_service(xfo.layer(csp.layer(ServeFile::new(static_dir.join(f)))))
    };
    let app = Router::new()
        .merge(api::router())
        .route_service("/", page("index.html"))
        .route_service("/domainmasst", page("domainmasst.html"))
        .route_service("/lifemasst", page("lifemasst.html"))
        .route_service("/geomasst", page("geomasst.html"))
        .route_service("/logo.png", get_service(ServeFile::new(state.cfg.root.join("logo.png"))))
        .nest_service("/static", ServeDir::new(&static_dir))
        .layer(SetResponseHeaderLayer::if_not_present(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff")))
        .layer(SetResponseHeaderLayer::if_not_present(header::REFERRER_POLICY, HeaderValue::from_static("strict-origin-when-cross-origin")))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    tracing::info!("StructureMASST listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
