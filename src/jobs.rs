//! Background jobs. Long-running actions (library search, raw data search,
//! DomainMASST, LifeMASST, GeoMASST) run as tokio tasks; the browser polls
//! `/api/jobs/{id}`. This replaces the blocking Streamlit spinners and the
//! Celery result polling.

use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Running,
    Done,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct JobView {
    pub id: String,
    pub kind: String,
    pub status: Status,
    pub message: String,
    pub warnings: Vec<String>,
    /// Live output of external steps (data preparation, Nextflow); last lines only.
    pub log: Vec<String>,
    pub error: Option<String>,
    pub result: Option<Value>,
    pub elapsed_s: u64,
}

struct Job {
    sid: String,
    view: JobView,
    started: Instant,
    finished: Option<Instant>,
}

#[derive(Clone, Default)]
pub struct JobStore {
    jobs: Arc<Mutex<HashMap<String, Job>>>,
}

/// Handle given to a running job to report progress and warnings.
#[derive(Clone)]
pub struct JobCtx {
    store: JobStore,
    pub id: String,
}

impl JobCtx {
    pub fn progress(&self, msg: impl Into<String>) {
        if let Some(j) = self.store.jobs.lock().unwrap().get_mut(&self.id) {
            j.view.message = msg.into();
        }
    }

    pub fn log(&self, line: impl Into<String>) {
        if let Some(j) = self.store.jobs.lock().unwrap().get_mut(&self.id) {
            j.view.log.push(line.into());
            let n = j.view.log.len();
            if n > 400 {
                j.view.log.drain(..n - 400);
            }
        }
    }

    pub fn warn(&self, msg: impl Into<String>) {
        if let Some(j) = self.store.jobs.lock().unwrap().get_mut(&self.id) {
            j.view.warnings.push(msg.into());
        }
    }
}

impl JobStore {
    pub fn spawn<F, Fut>(&self, sid: &str, kind: &str, f: F) -> String
    where
        F: FnOnce(JobCtx) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<Value>> + Send + 'static,
    {
        let id = crate::session::new_sid() + &crate::session::new_sid();
        let job = Job {
            sid: sid.to_string(),
            view: JobView {
                id: id.clone(),
                kind: kind.to_string(),
                status: Status::Running,
                message: String::new(),
                warnings: vec![],
                log: vec![],
                error: None,
                result: None,
                elapsed_s: 0,
            },
            started: Instant::now(),
            finished: None,
        };
        self.jobs.lock().unwrap().insert(id.clone(), job);
        let ctx = JobCtx { store: self.clone(), id: id.clone() };
        let store = self.clone();
        let jid = id.clone();
        tokio::spawn(async move {
            // Run in its own task so a panic becomes a job error instead of a lost job.
            let res = tokio::spawn(f(ctx)).await;
            let mut jobs = store.jobs.lock().unwrap();
            if let Some(j) = jobs.get_mut(&jid) {
                j.finished = Some(Instant::now());
                match res {
                    Ok(Ok(v)) => {
                        j.view.status = Status::Done;
                        j.view.result = Some(v);
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("job {} ({}) failed: {:#}", jid, j.view.kind, e);
                        j.view.status = Status::Error;
                        j.view.error = Some(format!("{e:#}"));
                    }
                    Err(e) => {
                        tracing::error!("job {} panicked: {}", jid, e);
                        j.view.status = Status::Error;
                        j.view.error = Some("internal error".into());
                    }
                }
            }
        });
        id
    }

    /// View of a job, only for the session that started it.
    pub fn view(&self, sid: &str, id: &str) -> Option<JobView> {
        let jobs = self.jobs.lock().unwrap();
        let j = jobs.get(id)?;
        if j.sid != sid {
            return None;
        }
        let mut v = j.view.clone();
        v.elapsed_s = j.finished.unwrap_or_else(Instant::now).duration_since(j.started).as_secs();
        Some(v)
    }

    /// Is a job of this kind already running for the session?
    pub fn running(&self, sid: &str, kind: &str) -> Option<String> {
        self.jobs
            .lock()
            .unwrap()
            .values()
            .find(|j| j.sid == sid && j.view.kind == kind && j.view.status == Status::Running)
            .map(|j| j.view.id.clone())
    }

    pub fn cleanup(&self, keep_finished: Duration) {
        self.jobs
            .lock()
            .unwrap()
            .retain(|_, j| j.finished.map(|f| f.elapsed() < keep_finished).unwrap_or(true));
    }
}
