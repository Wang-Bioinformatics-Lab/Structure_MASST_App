//! Server-side checks on everything a user can type or upload, before it
//! reaches the Python worker, SQL, FASST or the filesystem. The browser UI does
//! not need to be trusted for any of these.

use anyhow::{bail, Result};

/// Longest SMILES/SMARTS/USI/class label accepted.
pub const MAX_QUERY_LEN: usize = 2000;
/// Longest name (PubChem search text, batch names).
pub const MAX_NAME_LEN: usize = 200;

/// Upper bound on queries in one batch file (override with MAX_BATCH_QUERIES).
pub fn max_batch() -> usize {
    std::env::var("MAX_BATCH_QUERIES").ok().and_then(|v| v.parse().ok()).unwrap_or(500)
}

fn no_control(s: &str, what: &str) -> Result<()> {
    if s.chars().any(|c| c.is_control()) {
        bail!("{what} contains control characters.");
    }
    Ok(())
}

/// SMILES / SMARTS text.
pub fn structure(s: &str) -> Result<()> {
    if s.len() > MAX_QUERY_LEN {
        bail!("SMILES/SMARTS is too long (max {MAX_QUERY_LEN} characters).");
    }
    if s.chars().any(char::is_whitespace) {
        bail!("SMILES/SMARTS must not contain spaces.");
    }
    no_control(s, "SMILES/SMARTS")
}

/// A spectrum USI: `mzspec:` followed by printable, space-free text.
pub fn usi(s: &str) -> Result<()> {
    if !s.starts_with("mzspec:") {
        bail!("A USI starts with 'mzspec:'.");
    }
    if s.len() > MAX_QUERY_LEN || s.chars().any(char::is_whitespace) {
        bail!("Not a valid USI.");
    }
    no_control(s, "USI")
}

/// Free-text names (PubChem search, batch names).
pub fn name(s: &str) -> Result<()> {
    if s.chars().count() > MAX_NAME_LEN {
        bail!("Name is too long (max {MAX_NAME_LEN} characters).");
    }
    no_control(s, "Name")
}

/// Fixed molecular formula / allowed element difference / modification formula:
/// letters and digits only ("any" and "" mean no filter).
pub fn formula_like(s: &str, what: &str) -> Result<()> {
    if s.len() > 100 || !s.chars().all(|c| c.is_ascii_alphanumeric()) {
        bail!("{what} may only contain letters and digits (e.g. C8H10N4O2 or NO).");
    }
    Ok(())
}

/// Numbers typed into text boxes.
pub fn number(s: &str, what: &str) -> Result<f64> {
    let v: f64 = s.trim().parse().map_err(|_| anyhow::anyhow!("{what} must be a number."))?;
    if !v.is_finite() {
        bail!("{what} must be a number.");
    }
    Ok(v)
}

pub fn one_of(s: &str, allowed: &[&str], what: &str) -> Result<()> {
    if !allowed.contains(&s) {
        bail!("Invalid {what}.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_injection_shaped_input() {
        assert!(usi("mzspec:MSV1:a.mzML:scan:1").is_ok());
        assert!(usi("mzspec:MSV1:../../etc/passwd x").is_err());
        assert!(usi("http://evil").is_err());
        assert!(structure("CCO").is_ok());
        assert!(structure("CCO; rm -rf /").is_err());
        assert!(formula_like("C8H10N4O2", "f").is_ok());
        assert!(formula_like("' OR 1=1 --", "f").is_err());
        assert!(name("caffeine\u{0}").is_err());
        assert!(number("0.8", "t").is_ok());
        assert!(number("NaN", "t").is_err());
    }
}
