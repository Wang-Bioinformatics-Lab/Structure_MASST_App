//! External HTTP services: PubChem name lookup (port of bin/pubchem_handling.py),
//! SMARTS.plus rendering (port of bin/smarts_api.py) and umami event tracking.

use base64::Engine;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

use crate::links::py_quote_slash;

const TIMEOUT: Duration = Duration::from_secs(15);
const TRIES: u32 = 3;
const BACKOFF: f64 = 0.7;

/// GET with retries/backoff (same policy as the Python helper).
async fn get_with_retries(client: &Client, url: &str, query: &[(&str, String)], accept_json: bool) -> Option<reqwest::Response> {
    let accept = if accept_json { "application/json" } else { "*/*" };
    for i in 0..TRIES {
        let req = client
            .get(url)
            .query(query)
            .header("User-Agent", "pubchem-utils/1.0")
            .header("Accept", accept)
            .timeout(TIMEOUT);
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                return Some(resp);
            }
        }
        tokio::time::sleep(Duration::from_secs_f64(BACKOFF * 2f64.powi(i as i32))).await;
    }
    None
}

/// PubChem autocomplete (typo tolerant) → suggested names.
pub async fn pubchem_autocomplete(client: &Client, query: &str) -> Vec<String> {
    if query.chars().count() < 2 {
        return vec![];
    }
    let url = format!("https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{}/JSON", py_quote_slash(query));
    let Some(resp) = get_with_retries(client, &url, &[("limit", "25".into())], true).await else { return vec![] };
    let Ok(v) = resp.json::<Value>().await else { return vec![] };
    v["dictionary_terms"]["compound"]
        .as_array()
        .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).take(25).collect())
        .unwrap_or_default()
}

async fn name_to_cid(client: &Client, name: &str) -> Option<i64> {
    let url = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON", py_quote_slash(name));
    let v: Value = get_with_retries(client, &url, &[], true).await?.json().await.ok()?;
    v["IdentifierList"]["CID"].as_array()?.first()?.as_i64()
}

async fn cid_to_canonical_smiles(client: &Client, cid: i64) -> Option<String> {
    let url_txt = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT");
    if let Some(r) = get_with_retries(client, &url_txt, &[], false).await {
        if let Ok(t) = r.text().await {
            let t = t.trim();
            let low = t.to_lowercase();
            if !t.is_empty() && !["status", "error", "<!doctype", "<html"].iter().any(|p| low.starts_with(p)) {
                return Some(t.to_string());
            }
        }
    }
    let url_json = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON");
    let v: Value = get_with_retries(client, &url_json, &[], true).await?.json().await.ok()?;
    let props = v["PropertyTable"]["Properties"].as_array()?;
    let p = props.first()?;
    // PubChem renamed CanonicalSMILES → ConnectivitySMILES in 2025; accept either.
    p.get("CanonicalSMILES")
        .or_else(|| p.get("ConnectivitySMILES"))
        .and_then(|s| s.as_str())
        .map(String::from)
}

/// Name → CID → SMILES with the single-component rule.
/// Returns (smiles, warning).
pub async fn resolve_name(client: &Client, name: &str) -> (String, Option<String>) {
    if name.is_empty() {
        return (String::new(), None);
    }
    let smiles = match name_to_cid(client, name).await {
        Some(cid) => cid_to_canonical_smiles(client, cid).await,
        None => None,
    };
    match smiles {
        Some(s) if !s.contains('.') => (s, None),
        _ => (
            String::new(),
            Some(
                "PubChem entry does not represent a singular molecule \
                 (no SMILES available or multi-component SMILES containing '.')."
                    .into(),
            ),
        ),
    }
}

/// Render a SMARTS pattern via SMARTS.plus; returns PNG bytes.
pub async fn smarts_image(client: &Client, smarts: &str, api_key: &str, job_id: &str) -> Option<Vec<u8>> {
    let body = json!({
        "job_id": job_id,
        "query": {"smarts": smarts, "parameters": {"file_format": "png", "legend_mode": "3"}}
    });
    let send = || {
        client
            .post("https://api.smarts.plus/smartsView/")
            .header("X-API-Key", api_key)
            .json(&body)
            .timeout(Duration::from_secs(60))
            .send()
    };
    let mut resp = send().await.ok()?;
    if !resp.status().is_success() {
        tracing::warn!("SMARTSPlus request failed with status {}. Retrying...", resp.status());
        tokio::time::sleep(Duration::from_secs(2)).await;
        resp = send().await.ok()?;
    }
    let v: Value = resp.json().await.ok()?;
    let img = v["result"]["image"].as_str()?;
    base64::engine::general_purpose::STANDARD.decode(img).ok()
}

/// Server-side analytics event (what `umami.new_event(...)` did). Fire and forget.
pub fn track_event(client: &Client, event_name: &str) {
    let client = client.clone();
    let body = json!({
        "type": "event",
        "payload": {
            "hostname": "analytics-api.gnps2.org",
            "language": "en-US",
            "referrer": "",
            "screen": "1920x1080",
            "title": "",
            "url": "/event-api-endpoint",
            "website": "032bfca4-a353-4586-b637-8908d8b71c85",
            "name": event_name,
            "data": {}
        }
    });
    tokio::spawn(async move {
        let res = client
            .post("https://analytics-api.gnps2.org/api/send")
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) structure-masst-server")
            .json(&body)
            .timeout(Duration::from_secs(10))
            .send()
            .await;
        if let Err(e) = res {
            tracing::debug!("Error tracking event: {e}");
        }
    });
}
