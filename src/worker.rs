//! Bridge to the Python compute worker (bin/compute_worker.py).
//!
//! One long-lived Python process; requests are multiplexed over its stdin and
//! answered (possibly out of order) on its stdout. If the process dies it is
//! restarted on the next call.

use anyhow::{anyhow, Context, Result};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};

/// Upper bound for a single worker request (long FASST searches included).
const CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(4 * 3600);

type Pending =Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

struct Running {
    stdin: ChildStdin,
    _child: Child,
    alive: Arc<std::sync::atomic::AtomicBool>,
}

pub struct PyWorker {
    python: String,
    root: PathBuf,
    next_id: AtomicU64,
    pending: Pending,
    running: Mutex<Option<Running>>,
}

impl PyWorker {
    pub fn new(python: String, root: PathBuf) -> Self {
        PyWorker {
            python,
            root,
            next_id: AtomicU64::new(1),
            pending: Arc::new(Mutex::new(HashMap::new())),
            running: Mutex::new(None),
        }
    }

    async fn spawn(&self) -> Result<Running> {
        tracing::info!("starting python compute worker");
        let mut child = Command::new(&self.python)
            .arg("-u")
            .arg(self.root.join("bin/compute_worker.py"))
            .current_dir(&self.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to start {} bin/compute_worker.py", self.python))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let alive = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let pending = self.pending.clone();
        let alive2 = alive.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    tracing::warn!("worker: unparsable line: {}", line.chars().take(200).collect::<String>());
                    continue;
                };
                let id = msg["id"].as_u64().unwrap_or(0);
                if id == 0 {
                    continue; // ready banner
                }
                if let Some(tx) = pending.lock().await.remove(&id) {
                    let res = if msg["ok"].as_bool() == Some(true) {
                        Ok(msg.get("result").cloned().unwrap_or(Value::Null))
                    } else {
                        Err(msg["error"].as_str().unwrap_or("unknown worker error").to_string())
                    };
                    let _ = tx.send(res);
                }
            }
            alive2.store(false, Ordering::SeqCst);
            tracing::error!("python compute worker exited");
            for (_, tx) in pending.lock().await.drain() {
                let _ = tx.send(Err("compute worker exited".into()));
            }
        });
        Ok(Running { stdin, _child: child, alive })
    }

    /// Start the worker eagerly (it preloads the ReDU table).
    pub async fn start(&self) -> Result<()> {
        let mut guard = self.running.lock().await;
        if guard.is_none() {
            *guard = Some(self.spawn().await?);
        }
        Ok(())
    }

    pub async fn call(&self, op: &str, args: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        let line = serde_json::to_string(&json!({"id": id, "op": op, "args": args}))? + "\n";
        {
            let mut guard = self.running.lock().await;
            let needs_restart = guard.as_ref().map(|r| !r.alive.load(Ordering::SeqCst)).unwrap_or(true);
            if needs_restart {
                *guard = Some(self.spawn().await?);
            }
            let running = guard.as_mut().unwrap();
            if let Err(e) = running.stdin.write_all(line.as_bytes()).await {
                self.pending.lock().await.remove(&id);
                *guard = None;
                return Err(anyhow!("failed to send request to worker: {e}"));
            }
            running.stdin.flush().await.ok();
        }
        match tokio::time::timeout(CALL_TIMEOUT, rx).await {
            Ok(Ok(Ok(v))) => Ok(v),
            Ok(Ok(Err(e))) => Err(anyhow!(e)),
            Ok(Err(_)) => Err(anyhow!("compute worker dropped the request")),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(anyhow!("the computation did not finish within {} hours", CALL_TIMEOUT.as_secs() / 3600))
            }
        }
    }

    pub async fn call_as<T: DeserializeOwned>(&self, op: &str, args: Value) -> Result<T> {
        let v = self.call(op, args).await?;
        serde_json::from_value(v).with_context(|| format!("unexpected result from worker op {op}"))
    }
}
