//! Link builders (ported from bin/linkouts.py and the Streamlit page) so the
//! generated URLs stay byte-identical to the Python versions.

use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

use crate::table::{format_float, value_as_f64};

/// Python's `urllib.parse.quote(s, safe='')`: keep only `A-Za-z0-9_.-~`.
const PY_QUOTE: &AsciiSet = &NON_ALPHANUMERIC.remove(b'_').remove(b'.').remove(b'-').remove(b'~');
/// Python's `urllib.parse.quote(s)` (default safe='/').
const PY_QUOTE_SLASH: &AsciiSet = &PY_QUOTE.remove(b'/');

pub fn py_quote(s: &str) -> String {
    utf8_percent_encode(s, PY_QUOTE).to_string()
}

pub fn py_quote_slash(s: &str) -> String {
    utf8_percent_encode(s, PY_QUOTE_SLASH).to_string()
}

/// Python's `urllib.parse.quote_plus(s)`.
pub fn py_quote_plus(s: &str) -> String {
    s.split(' ').map(py_quote).collect::<Vec<_>>().join("+")
}

/// Library spectrum viewer link used in the "Molecules by InChIKey" tables.
/// (port of `make_dashinterface_link`; a USI is used as is)
pub fn library_spectrum_link(spectrum_id: &str) -> String {
    format!(
        "http://metabolomics-usi.gnps2.org/dashinterface?usi1={}&width=10.0&height=6.0&mz_min=None&mz_max=None&max_intensity=125&annotate_precision=4&annotation_rotation=90&cosine=standard&fragment_mz_tolerance=0.02&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D",
        py_quote(&library_usi(spectrum_id))
    )
}

/// `mzspec:` USI for a library spectrum id (a USI is returned unchanged).
pub fn library_usi(spectrum_id: &str) -> String {
    if spectrum_id.starts_with("mzspec:") {
        spectrum_id.to_string()
    } else if spectrum_id.starts_with("CCMSLIB") {
        format!("mzspec:GNPS:GNPS-LIBRARY:accession:{spectrum_id}")
    } else {
        format!("mzspec:MASSBANK::accession:{spectrum_id}")
    }
}

/// Port of `build_spectraresolver_link`.
pub fn spectraresolver_link(usi1: &str, usi2: &str) -> String {
    format!(
        "http://metabolomics-usi.gnps2.org/dashinterface?usi1={}&usi2={}&width=10.0&height=6.0&mz_min=None&mz_max=None&max_intensity=125&annotate_precision=4&annotation_rotation=90&cosine=standard&fragment_mz_tolerance=0.05&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D",
        py_quote_plus(usi1),
        py_quote_plus(usi2)
    )
}

static SCAN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r":scan:(\d+)$").unwrap());

/// Python `json.dumps` of a flat string map (", " and ": " separators).
fn py_json_dumps_pairs(pairs: &[(&str, String)]) -> String {
    let body: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), v))
        .collect();
    format!("{{{}}}", body.join(", "))
}

/// Port of `build_dashboard_eic_url(usi, xic_mz, xic_tolerance=0.05)`.
pub fn dashboard_eic_url(usi: &str, xic_mz: &Value, xic_tolerance: f64) -> String {
    let ms2_identifier = SCAN_RE
        .captures(usi)
        .map(|c| format!("MS2:{}", &c[1]))
        .unwrap_or_else(|| "None".to_string());
    let xmz = value_as_f64(xic_mz).filter(|f| f.is_finite());

    let mut params: Vec<(&str, String)> = vec![
        ("xic_mz", xmz.map(|v| format!("{v:.6}")).unwrap_or_default()),
        ("xic_formula", String::new()),
        ("xic_peptide", String::new()),
        ("xic_tolerance", format_float(xic_tolerance)),
        ("xic_ppm_tolerance", "10".into()),
        ("xic_tolerance_unit", "Da".into()),
        ("xic_rt_window", String::new()),
        ("xic_norm", "False".into()),
        ("xic_file_grouping", "MZ".into()),
        ("xic_integration_type", "AUC".into()),
        ("show_ms2_markers", "True".into()),
        ("ms2marker_color", "blue".into()),
        ("ms2marker_size", "5".into()),
        ("ms2_identifier", ms2_identifier),
        ("show_lcms_2nd_map", "False".into()),
        ("polarity_filtering", "None".into()),
        ("polarity_filtering2", "None".into()),
        ("tic_option", "TIC".into()),
        ("overlay_usi", "None".into()),
        ("overlay_mz", "None".into()),
        ("overlay_rt", "None".into()),
        ("overlay_color", String::new()),
        ("overlay_size", String::new()),
        ("overlay_hover", String::new()),
        ("overlay_filter_column", String::new()),
        ("overlay_filter_value", String::new()),
        ("feature_finding_type", "Off".into()),
        ("feature_finding_ppm", "10".into()),
        ("feature_finding_noise", "10000".into()),
        ("feature_finding_min_peak_rt", "0.05".into()),
        ("feature_finding_max_peak_rt", "1.5".into()),
        ("feature_finding_rt_tolerance", "0.3".into()),
        ("massql_statement", "QUERY scaninfo(MS2DATA)".into()),
        ("sychronization_session_id", "cd9f3c5172c64c55b2c3efd6d2d49b73".into()),
        ("chromatogram_options", "[]".into()),
        ("comment", String::new()),
        ("map_plot_color_scale", "Hot_r".into()),
        ("map_plot_quantization_level", "Medium".into()),
        ("plot_theme", "plotly_white".into()),
    ];
    if let Some(x) = xmz {
        let zoom = py_json_dumps_pairs(&[
            ("xaxis.range[0]", "0".into()),
            ("xaxis.range[1]", "1".into()),
            ("yaxis.range[0]", format_float(x - 3.0)),
            ("yaxis.range[1]", format_float(x + 3.0)),
        ]);
        params.push(("map_plot_zoom", zoom));
    }
    let query: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}={}", py_quote(k), py_quote(v)))
        .collect();
    let usi_json = serde_json::to_string(usi).unwrap();
    let fragment = format!("{{\"usi\": {usi_json}, \"usi_select\": {usi_json}, \"usi2\": \"\"}}");
    format!("https://dashboard.gnps2.org//?{}#{}", query.join("&"), py_quote_slash(&fragment))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eic_url_matches_python() {
        // Generated with bin/linkouts.py:
        // build_dashboard_eic_url("mzspec:MSV000079949:ccms_peak/a b.mzXML:scan:2449", xic_mz=556.3622, xic_tolerance=0.05)
        let url = dashboard_eic_url("mzspec:MSV000079949:ccms_peak/a b.mzXML:scan:2449", &json!(556.3622), 0.05);
        let expected = include_str!("../tests/fixtures/eic_url.txt").trim();
        assert_eq!(url, expected);
    }

    #[test]
    fn spectraresolver_matches_python() {
        let url = spectraresolver_link("mzspec:MSV1:a b/c.mzML:scan:1", "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB1");
        let expected = include_str!("../tests/fixtures/spectraresolver_url.txt").trim();
        assert_eq!(url, expected);
    }
}

#[cfg(test)]
mod dashinterface_tests {
    use super::*;

    #[test]
    fn library_links_match_python() {
        // Generated with make_dashinterface_link() from structureMASST.py
        let expected: Vec<&str> = include_str!("../tests/fixtures/dashinterface_links.txt").lines().collect();
        let ids = ["CCMSLIB00005435899", "MSBNK-RIKEN-PR100121", "mzspec:MSV000084494:ccms_peak/raw/a b.mzML:scan:1043"];
        for (id, want) in ids.iter().zip(expected) {
            assert_eq!(library_spectrum_link(id), want);
        }
    }
}
