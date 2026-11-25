//! Per-session state (the Rust counterpart of `st.session_state`).
//!
//! A session id is a 12-character hex string, as before, and every session has
//! its folder `sessionoutput/<sid>/`. The state is kept in memory and written
//! to `sessionoutput/<sid>/state.json` after each change, so it survives server
//! restarts and idle eviction.

use anyhow::Result;
use indexmap::IndexMap;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::table::Table;

/// Library entries of one structure query: molecule overview + spectra per 2D InChIKey.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Group {
    pub overview: Table,
    pub spectra: IndexMap<String, Table>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionData {
    /// smiles, name, type, searchtype (the former `query_table`)
    pub query_table: Option<Table>,
    /// query name -> library entries (the former `grouped_results` + `molecule_overview`)
    pub groups: IndexMap<String, Group>,
    /// query name -> ReDU-annotated raw data matches (the former `raw_results[name]["redu"]`)
    pub raw_results: Option<IndexMap<String, Table>>,
    /// Mode used for the last raw data search ("FASSTrecords" | "FASST")
    pub raw_mode: Option<String>,
    /// query behind each result name (the former `query_by_name`)
    #[serde(default)]
    pub query_by_name: IndexMap<String, String>,
    /// peaks of uploaded MGF spectra by spectrum id (the former `mgf_spectra_store`)
    #[serde(default)]
    pub mgf_store: IndexMap<String, MgfSpectrum>,
    /// "structuremasst" when LifeMASST input was set up from the results page
    #[serde(default)]
    pub lifemasst_prepared_source: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MgfSpectrum {
    pub precursor_mz: serde_json::Value,
    pub peaks: serde_json::Value,
}

pub struct Session {
    pub sid: String,
    pub dir: PathBuf,
    pub data: SessionData,
    last_access: Instant,
}

impl Session {
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let tmp = self.dir.join("state.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&self.data)?)?;
        std::fs::rename(tmp, self.dir.join("state.json"))?;
        Ok(())
    }
}

pub type SessionRef = Arc<Mutex<Session>>;

pub struct SessionStore {
    root: PathBuf,
    sessions: std::sync::Mutex<HashMap<String, SessionRef>>,
}

/// 32 hex characters (128 bits, unguessable); 12-character ids from the
/// Streamlit era are still accepted for existing session folders.
pub fn valid_sid(sid: &str) -> bool {
    (sid.len() == 32 || sid.len() == 12) && sid.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn new_sid() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl SessionStore {
    pub fn new(root: PathBuf) -> Self {
        SessionStore { root, sessions: std::sync::Mutex::new(HashMap::new()) }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn create(&self) -> Result<String> {
        let sid = new_sid();
        std::fs::create_dir_all(self.root.join(&sid))?;
        Ok(sid)
    }

    /// Get (or lazily load/create) a session. Returns None for malformed ids.
    pub fn get(&self, sid: &str) -> Option<SessionRef> {
        if !valid_sid(sid) {
            return None;
        }
        let mut map = self.sessions.lock().unwrap();
        if let Some(s) = map.get(sid) {
            if let Ok(mut g) = s.try_lock() {
                g.last_access = Instant::now();
            }
            return Some(s.clone());
        }
        let dir = self.root.join(sid);
        let data = std::fs::read(dir.join("state.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<SessionData>(&b).ok())
            .unwrap_or_default();
        let _ = std::fs::create_dir_all(&dir);
        let s = Arc::new(Mutex::new(Session { sid: sid.to_string(), dir, data, last_access: Instant::now() }));
        map.insert(sid.to_string(), s.clone());
        Some(s)
    }

    /// Drop sessions from memory that were idle for longer than `max_idle` (state stays on disk).
    pub fn evict_idle(&self, max_idle: Duration) {
        let mut map = self.sessions.lock().unwrap();
        map.retain(|_, s| match s.try_lock() {
            Ok(g) => g.last_access.elapsed() < max_idle,
            Err(_) => true,
        });
    }
}
