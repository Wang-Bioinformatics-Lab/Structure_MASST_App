//! Plotly figure builders (JSON) for the two Sankey diagrams. Ported from
//! structureMASST.py (library Sankey) and bin/plotting.py::raw_data_sankey.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

use crate::table::{value_to_string, Table};

/// plotly.express.colors.qualitative.Safe
pub const SAFE: [&str; 11] = [
    "rgb(136, 204, 238)",
    "rgb(204, 102, 119)",
    "rgb(221, 204, 119)",
    "rgb(17, 119, 51)",
    "rgb(51, 34, 136)",
    "rgb(170, 68, 153)",
    "rgb(68, 170, 153)",
    "rgb(153, 153, 51)",
    "rgb(136, 34, 85)",
    "rgb(102, 17, 0)",
    "rgb(136, 136, 136)",
];

fn unique_in_order(values: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    values.iter().filter(|v| seen.insert((*v).clone())).cloned().collect()
}

fn stage_annotations(names: &[&str]) -> Vec<Value> {
    let n = (names.len() - 1) as f64;
    names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let xanchor = if i == 0 { "left" } else if i == names.len() - 1 { "right" } else { "center" };
            json!({
                "x": i as f64 / n, "y": 1.02, "xref": "paper", "yref": "paper",
                "text": name, "showarrow": false,
                "font": {"size": 14, "color": "black"}, "xanchor": xanchor
            })
        })
        .collect()
}

fn sankey_figure(labels: &[String], source: Vec<usize>, target: Vec<usize>, value: Vec<usize>, colors: Vec<String>, stage_names: &[&str]) -> Value {
    json!({
        "data": [{
            "type": "sankey",
            "arrangement": "snap",
            "textfont": {"family": "Arial, sans-serif", "size": 12, "color": "black"},
            "node": {
                "label": labels,
                "color": vec!["#F2F2F2"; labels.len()],
                "pad": 15, "thickness": 20,
                "line": {"color": "black", "width": 0.5}
            },
            "link": {"source": source, "target": target, "value": value, "color": colors}
        }],
        "layout": {
            "font": {"family": "Arial, sans-serif", "size": 12},
            "margin": {"l": 60, "r": 60, "t": 120, "b": 20},
            "annotations": stage_annotations(stage_names)
        }
    })
}

/// Sankey of the available library spectra: analyzer → ion mode → adduct → collision energy.
/// Returns None when one of the required columns is missing (as the Streamlit page did).
pub fn library_sankey(df: &Table) -> Option<Value> {
    let stages = ["msMassAnalyzer", "Ion_Mode", "Adduct", "collision_energy"];
    if stages.iter().any(|s| !df.has(s)) {
        return None;
    }
    let fills = ["Unknown_1", "Unknown_2", "Unknown_3", "Unknown_4"];
    // Stage values as strings with the Unknown_<n> fill for missing/"nan".
    let cols: Vec<Vec<String>> = stages
        .iter()
        .zip(fills)
        .map(|(s, fill)| {
            df.column_values(s)
                .iter()
                .map(|v| {
                    let s = value_to_string(v);
                    if v.is_null() || s == "nan" { fill.to_string() } else { s }
                })
                .collect()
        })
        .collect();

    let mut all = Vec::new();
    for c in &cols {
        all.extend(unique_in_order(c));
    }
    let labels = unique_in_order(&all);
    let index: HashMap<&String, usize> = labels.iter().enumerate().map(|(i, l)| (l, i)).collect();

    let ms_cats = unique_in_order(&cols[0]);
    let color_map: HashMap<&String, String> = ms_cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c, SAFE[i % SAFE.len()].replace("rgb", "rgba").replace(')', ", 0.3)")))
        .collect();

    let (mut source, mut target, mut value, mut colors) = (vec![], vec![], vec![], vec![]);
    for i in 0..stages.len() - 1 {
        for cat in &ms_cats {
            let mut counts: BTreeMap<(&String, &String), usize> = BTreeMap::new();
            for r in 0..df.len() {
                if &cols[0][r] == cat {
                    *counts.entry((&cols[i][r], &cols[i + 1][r])).or_default() += 1;
                }
            }
            for ((a, b), n) in counts {
                source.push(index[a]);
                target.push(index[b]);
                value.push(n);
                colors.push(color_map[cat].clone());
            }
        }
    }
    Some(sankey_figure(&labels, source, target, value, colors, &["Mass Analyzer", "Ion Mode", "Adduct", "Collision Energy"]))
}

/// Port of `raw_data_sankey(df, col1, col2, col3, col4)`.
pub fn raw_data_sankey(df: &Table, cols: &[String]) -> Value {
    // 1) Collapse each column to its top 10 values + "others" (NaN → others).
    let stage_vals: Vec<Vec<String>> = cols
        .iter()
        .map(|c| {
            let vals = df.column_values(c);
            let mut counts: Vec<(String, usize, usize)> = Vec::new(); // (key, count, first_seen)
            let mut pos: HashMap<String, usize> = HashMap::new();
            for (i, v) in vals.iter().enumerate() {
                if v.is_null() {
                    continue;
                }
                let k = value_to_string(v);
                match pos.get(&k) {
                    Some(&p) => counts[p].1 += 1,
                    None => {
                        pos.insert(k.clone(), counts.len());
                        counts.push((k, 1, i));
                    }
                }
            }
            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
            let top: std::collections::HashSet<String> = counts.into_iter().take(10).map(|c| c.0).collect();
            vals.iter()
                .map(|v| {
                    let k = value_to_string(v);
                    if !v.is_null() && top.contains(&k) { k } else { "others".to_string() }
                })
                .collect()
        })
        .collect();

    let mut labels: Vec<String> = Vec::new();
    for (i, vals) in stage_vals.iter().enumerate() {
        for u in unique_in_order(vals) {
            labels.push(format!("{}_{}", i + 1, u));
        }
    }
    let labels = unique_in_order(&labels);
    let index: HashMap<&String, usize> = labels.iter().enumerate().map(|(i, l)| (l, i)).collect();

    let column_1_vals = unique_in_order(&stage_vals[0]);
    let mut color_map: HashMap<String, String> = column_1_vals
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), SAFE[i % SAFE.len()].to_string()))
        .collect();
    color_map.insert("others".into(), "#B0B0B0".into());

    let (mut source, mut target, mut value, mut colors) = (vec![], vec![], vec![], vec![]);
    for i in 0..stage_vals.len() - 1 {
        let mut counts: BTreeMap<(&String, &String, &String), usize> = BTreeMap::new();
        for r in 0..df.len() {
            *counts.entry((&stage_vals[i][r], &stage_vals[i + 1][r], &stage_vals[0][r])).or_default() += 1;
        }
        for ((a, b, ck), n) in counts {
            source.push(index[&format!("{}_{}", i + 1, a)]);
            target.push(index[&format!("{}_{}", i + 2, b)]);
            value.push(n);
            colors.push(color_map.get(ck).cloned().unwrap_or_else(|| "rgba(0,0,0,0.3)".into()));
        }
    }
    let names: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
    sankey_figure(&labels, source, target, value, colors, &names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_sankey_links() {
        let df = Table {
            columns: vec!["a".into(), "b".into()],
            rows: vec![
                vec![json!("x"), json!(1)],
                vec![json!("x"), json!(1)],
                vec![json!("y"), Value::Null],
            ],
        };
        let fig = raw_data_sankey(&df, &["a".to_string(), "b".to_string()]);
        let labels = fig["data"][0]["node"]["label"].as_array().unwrap();
        assert_eq!(labels, &vec![json!("1_x"), json!("1_y"), json!("2_1"), json!("2_others")]);
        assert_eq!(fig["data"][0]["link"]["value"], json!([2, 1]));
    }
}
