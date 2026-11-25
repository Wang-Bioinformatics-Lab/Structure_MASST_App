//! A small row-major table type that stands in for the pandas DataFrames the
//! Streamlit app kept in its session state. Tables travel between Rust, the
//! Python worker and the browser as `{"columns": [...], "rows": [[...]]}`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Table {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// String form of a value the way pandas' `astype(str)` would print it.
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::Null => "nan".to_string(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                format_float(f)
            }
        }
        other => other.to_string(),
    }
}

/// Python `repr(float)`-like formatting (always keeps a decimal point).
pub fn format_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{}", f);
    if s.contains('.') || s.contains('e') || s.contains("inf") {
        s
    } else {
        format!("{}.0", s)
    }
}

pub fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub fn value_as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<f64>().ok().map(|f| f as i64),
        _ => None,
    }
}

impl Table {
    pub fn new(columns: Vec<String>) -> Self {
        Table { columns, rows: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn col(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.col(name).is_some()
    }

    pub fn cell<'a>(&'a self, row: usize, name: &str) -> Option<&'a Value> {
        self.col(name).and_then(|c| self.rows.get(row).and_then(|r| r.get(c)))
    }

    /// Column values (Null when the column is missing).
    pub fn column_values(&self, name: &str) -> Vec<Value> {
        match self.col(name) {
            Some(c) => self.rows.iter().map(|r| r.get(c).cloned().unwrap_or(Value::Null)).collect(),
            None => vec![Value::Null; self.rows.len()],
        }
    }

    /// Set (or append) a column from a vector of values.
    pub fn set_column(&mut self, name: &str, values: Vec<Value>) {
        debug_assert_eq!(values.len(), self.rows.len());
        match self.col(name) {
            Some(c) => {
                for (r, v) in self.rows.iter_mut().zip(values) {
                    if r.len() <= c {
                        r.resize(c + 1, Value::Null);
                    }
                    r[c] = v;
                }
            }
            None => {
                self.columns.push(name.to_string());
                for (r, v) in self.rows.iter_mut().zip(values) {
                    r.push(v);
                }
            }
        }
    }

    pub fn drop_column(&mut self, name: &str) {
        if let Some(c) = self.col(name) {
            self.columns.remove(c);
            for r in self.rows.iter_mut() {
                if c < r.len() {
                    r.remove(c);
                }
            }
        }
    }

    /// Keep only the given row positions (in the given order).
    pub fn take_rows(&self, idx: &[usize]) -> Table {
        Table {
            columns: self.columns.clone(),
            rows: idx.iter().filter_map(|&i| self.rows.get(i).cloned()).collect(),
        }
    }

    /// Drop the given row positions.
    pub fn drop_rows(&self, idx: &[usize]) -> Table {
        let drop: HashSet<usize> = idx.iter().copied().collect();
        Table {
            columns: self.columns.clone(),
            rows: self
                .rows
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop.contains(i))
                .map(|(_, r)| r.clone())
                .collect(),
        }
    }

    /// Reorder columns: `first` (those present) come first, the rest keep their order.
    pub fn reorder_first(&self, first: &[&str]) -> Table {
        let mut order: Vec<usize> = first.iter().filter_map(|n| self.col(n)).collect();
        for i in 0..self.columns.len() {
            if !order.contains(&i) {
                order.push(i);
            }
        }
        Table {
            columns: order.iter().map(|&i| self.columns[i].clone()).collect(),
            rows: self
                .rows
                .iter()
                .map(|r| order.iter().map(|&i| r.get(i).cloned().unwrap_or(Value::Null)).collect())
                .collect(),
        }
    }

    /// `pd.concat([...], ignore_index=True)`: outer join on columns, first-appearance order.
    pub fn concat(tables: &[&Table]) -> Table {
        let mut columns: Vec<String> = Vec::new();
        for t in tables {
            for c in &t.columns {
                if !columns.contains(c) {
                    columns.push(c.clone());
                }
            }
        }
        let mut out = Table::new(columns.clone());
        for t in tables {
            let map: Vec<Option<usize>> = columns.iter().map(|c| t.col(c)).collect();
            for r in &t.rows {
                out.rows.push(
                    map.iter()
                        .map(|m| m.and_then(|i| r.get(i).cloned()).unwrap_or(Value::Null))
                        .collect(),
                );
            }
        }
        out
    }

    /// `left.merge(right, on=key, how="left")` with pandas' `_x`/`_y` suffixes
    /// for overlapping non-key columns.
    pub fn merge_left(&self, right: &Table, key: &str) -> Table {
        let (Some(lk), Some(rk)) = (self.col(key), right.col(key)) else {
            return self.clone();
        };
        let right_cols: Vec<usize> = (0..right.columns.len()).filter(|&i| i != rk).collect();
        let overlap: HashSet<&String> = right_cols
            .iter()
            .map(|&i| &right.columns[i])
            .filter(|c| self.columns.contains(c))
            .collect();
        let mut columns: Vec<String> = self
            .columns
            .iter()
            .map(|c| if overlap.contains(c) { format!("{c}_x") } else { c.clone() })
            .collect();
        for &i in &right_cols {
            let c = &right.columns[i];
            columns.push(if overlap.contains(c) { format!("{c}_y") } else { c.clone() });
        }
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        for (ri, r) in right.rows.iter().enumerate() {
            let k = r.get(rk).cloned().unwrap_or(Value::Null);
            if !k.is_null() {
                index.entry(value_to_string(&k)).or_default().push(ri);
            }
        }
        let mut out = Table::new(columns);
        for r in &self.rows {
            let k = r.get(lk).cloned().unwrap_or(Value::Null);
            let matches = if k.is_null() { None } else { index.get(&value_to_string(&k)) };
            match matches {
                Some(ms) if !ms.is_empty() => {
                    for &ri in ms {
                        let mut row = r.clone();
                        row.resize(self.columns.len(), Value::Null);
                        for &i in &right_cols {
                            row.push(right.rows[ri].get(i).cloned().unwrap_or(Value::Null));
                        }
                        out.rows.push(row);
                    }
                }
                _ => {
                    let mut row = r.clone();
                    row.resize(self.columns.len(), Value::Null);
                    row.extend(std::iter::repeat(Value::Null).take(right_cols.len()));
                    out.rows.push(row);
                }
            }
        }
        out
    }

    /// Distinct non-null values of a column, in order of first appearance.
    pub fn unique(&self, name: &str) -> Vec<Value> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for v in self.column_values(name) {
            if v.is_null() {
                continue;
            }
            let k = value_to_string(&v);
            if seen.insert(k) {
                out.push(v);
            }
        }
        out
    }

    /// Rows whose `name` column equals `value` (string comparison).
    pub fn filter_eq(&self, name: &str, value: &str) -> Table {
        let Some(c) = self.col(name) else { return Table::new(self.columns.clone()) };
        Table {
            columns: self.columns.clone(),
            rows: self
                .rows
                .iter()
                .filter(|r| r.get(c).map(|v| !v.is_null() && value_to_string(v) == value).unwrap_or(false))
                .cloned()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn t(cols: &[&str], rows: Vec<Vec<Value>>) -> Table {
        Table { columns: cols.iter().map(|s| s.to_string()).collect(), rows }
    }

    #[test]
    fn merge_left_suffixes_and_missing() {
        let a = t(&["k", "Compound_Name", "v"], vec![vec![json!("A"), json!("x"), json!(1)], vec![json!("B"), json!("y"), json!(2)]]);
        let b = t(&["k", "Compound_Name"], vec![vec![json!("A"), json!("Caffeine")]]);
        let m = a.merge_left(&b, "k");
        assert_eq!(m.columns, vec!["k", "Compound_Name_x", "v", "Compound_Name_y"]);
        assert_eq!(m.rows[0][3], json!("Caffeine"));
        assert_eq!(m.rows[1][3], Value::Null);
    }

    #[test]
    fn concat_outer() {
        let a = t(&["a", "b"], vec![vec![json!(1), json!(2)]]);
        let b = t(&["b", "c"], vec![vec![json!(3), json!(4)]]);
        let c = Table::concat(&[&a, &b]);
        assert_eq!(c.columns, vec!["a", "b", "c"]);
        assert_eq!(c.rows[1], vec![Value::Null, json!(3), json!(4)]);
    }

    #[test]
    fn float_formatting_matches_python() {
        assert_eq!(value_to_string(&json!(30.0)), "30.0");
        assert_eq!(value_to_string(&json!(30)), "30");
        assert_eq!(value_to_string(&json!(0.85)), "0.85");
    }
}
