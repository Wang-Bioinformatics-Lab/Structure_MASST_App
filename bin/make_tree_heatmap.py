#!/usr/bin/env python3
"""
make_tree_heatmap.py

Generate a standalone HTML file showing the phylogenetic tree + MetaboTree
molecule hit columns (MASST red + Wikidata blue), mirroring the empress
"Load MetaboTree format" view without requiring Empress.

Tree rendering follows the same logic as make_tree_plot_file_count.py:
  - Bio.Phylo for parsing
  - Pruned to leaves that have at least one data hit
  - Ladderized (smaller clades first)
  - Actual branch lengths used for x positions (phylogram)

Usage:
    python bin/make_tree_heatmap.py \
        --metadata  <lifemasst_dir>/merged_metadata.tsv \
        --tree      <work_dir>/labelled_supertree_subset_prepped.nwk \
        --output    tree_heatmap.html \
        [--row-height 0.3]     pixels per leaf row
        [--cell-width 10]      pixels per data column
        [--tree-width 2000]    pixels for the tree panel
        [--kingdom animalia]   filter to one NCBIKingdom (case-insensitive)
        [--min-hits 1]         prune leaves with fewer hits
"""

import argparse
import base64
import copy
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo
from PIL import Image
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT = True
except ImportError:
    _RDKIT = False

# ── Colour maps (match empress_defaults_w_sparql.txt) ────────────────────────

_KINGDOM_HEX = {
    "plantae":        "#3a9e3a",
    "animalia":       "#e06400",
    "fungi":          "#9b59b6",
    "bacillati":      "#2980b9",
    "pseudomonadati": "#1abc9c",
    "bacteria":       "#2980b9",
    "protozoa":       "#e74c3c",
    "chromista":      "#f39c12",
    "fusobacteriati": "#8e44ad",
}
_KINGDOM_DEFAULT_HEX = "#aaaaaa"

_MASST_HEX = {     # red gradient – darkest = most specific
    "subspecies": "#ff0000",  # bright pure red – direct subspecies hit
    "species": "#cc0000",
    "genus":   "#e05050",
    "family":  "#eb8080",
    "order":   "#f0a0a0",
    "class":   "#f5c0c0",
    "phylum":  "#fae0e0",
    "kingdom": "#fdf0f0",
}

_WD_HEX = {        # blue gradient
    "species": "#0000cc",
    "genus":   "#5050e0",
    "family":  "#8080eb",
    "order":   "#a0a0f0",
    "class":   "#c0c0f5",
    "phylum":  "#e0e0fa",
    "kingdom": "#f0f0fd",
}

# Categorical palette for NCBIClass (ColorBrewer Set1 + Dark2 extended)
_CLASS_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#1b9e77", "#d95f02", "#7570b3",
    "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
    "#fdb462", "#b3de69", "#fccde5",
]
_CLASS_DEFAULT_HEX = "#cccccc"


def _h2rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


_KG_RGB = {k: _h2rgb(v) for k, v in _KINGDOM_HEX.items()}
_KG_DEF = _h2rgb(_KINGDOM_DEFAULT_HEX)
_MS_RGB = {k: _h2rgb(v) for k, v in _MASST_HEX.items()}
_WD_RGB = {k: _h2rgb(v) for k, v in _WD_HEX.items()}
_LGRAY  = (238, 238, 238)


def _build_strip_colormap(df: pd.DataFrame, col: str,
                          min_leaves: int = 4) -> tuple[dict[str, str], set[str], str]:
    """Assign colours to values of *col* that have >= min_leaves leaves.

    For NCBIKingdom the fixed _KINGDOM_HEX palette is used (every value is
    'common').  For all other columns the categorical _CLASS_PALETTE is used and
    values with < min_leaves leaves are collected in *rare*.

    Returns:
        colormap  – {value_lowercase: hex_color} for common values
        rare      – set of lowercase values with too few leaves (empty for kingdom)
        default_hex – hex string for values not in colormap
    """
    if col not in df.columns:
        return {}, set(), _KINGDOM_DEFAULT_HEX

    if col == "NCBIKingdom":
        colormap = {k: v for k, v in _KINGDOM_HEX.items()}
        return colormap, set(), _KINGDOM_DEFAULT_HEX

    counts  = df.groupby(col).apply(lambda g: g.index.nunique(), include_groups=False)
    rare    = {str(v).lower() for v, n in counts.items() if n < min_leaves}
    common  = sorted(str(v) for v, n in counts.items() if n >= min_leaves)
    colormap = {v: _CLASS_PALETTE[i % len(_CLASS_PALETTE)] for i, v in enumerate(common)}
    return colormap, rare, _CLASS_DEFAULT_HEX


def _build_class_colormap(df: pd.DataFrame,
                          min_leaves: int = 4) -> tuple[dict[str, str], set[str]]:
    """Legacy wrapper kept for backward compatibility."""
    cm, rare, _ = _build_strip_colormap(df, "NCBIClass", min_leaves)
    return cm, rare


def _load_smiles_map(metadata_path: str, molecules_path: str | None = None) -> dict[str, str]:
    """Load name→SMILES from an explicit molecules file or from a sibling structuremasst_input.tsv."""
    if molecules_path:
        candidates = [Path(molecules_path)]
    else:
        parent = Path(metadata_path).parent
        candidates = [
            parent / "structuremasst_input_unique.tsv",
            parent / "structuremasst_input.tsv",
        ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t", low_memory=False)
        if "name" not in df.columns or "query" not in df.columns:
            continue
        mask = df.get("type", pd.Series(["smiles"] * len(df))).str.lower() == "smiles"
        return dict(zip(df.loc[mask, "name"].astype(str),
                        df.loc[mask, "query"].astype(str)))
    return {}


def _render_mol_images(smiles_map: dict, pairs: list,
                       width: int, height: int,
                       render_scale: int = 10) -> dict[str, str]:
    """Render each molecule as a base64 PNG string (suffix → b64). Requires RDKit.

    Renders at render_scale × the display size so the browser downscales to a
    sharp image. Atom labels are made small and the minimum font floor is removed
    so they don't dominate tiny display sizes.
    """
    if not _RDKIT:
        print("  RDKit not available — molecule images skipped")
        return {}
    from rdkit.Chem import Draw
    rw = width  * render_scale
    rh = height * render_scale
    result = {}
    for suf, _, _ in pairs:
        smiles = smiles_map.get(suf)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            pil_img = Draw.MolToImage(mol, size=(rw, rh))
            buf = io.BytesIO()
            pil_img.save(buf, "PNG")
            result[suf] = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"  Warning: could not render {suf}: {e}")
    print(f"  Rendered {len(result)}/{len(pairs)} molecule images ({rw}×{rh} px each)")
    return result


# ── Tree handling (Bio.Phylo, matching make_tree_plot_file_count.py) ──────────

def _normalize_ott(x) -> str | None:
    if not x:
        return None
    s = str(x).strip()
    m = re.fullmatch(r"ott(\d+)", s, re.I)
    if m:
        return f"ott{m.group(1)}"
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return f"ott{m.group(1)}"
    return s


def _prune(tree, keep_ids: set):
    """Remove all leaves whose normalized OTT ID is not in keep_ids."""
    tree = copy.deepcopy(tree)
    for term in list(tree.get_terminals()):
        if _normalize_ott(term.name) not in keep_ids:
            tree.prune(term)
    return tree


def _ladderize(tree):
    """Sort children at each node so smaller clades come first."""
    tree = copy.deepcopy(tree)

    def _sort(clade):
        for c in clade.clades:
            _sort(c)
        if len(clade.clades) > 1:
            clade.clades.sort(
                key=lambda c: (
                    len(c.get_terminals()),
                    min((str(t.name) for t in c.get_terminals() if t.name), default=""),
                )
            )

    _sort(tree.root)
    return tree


def _layout(tree):
    """
    Compute node positions for a rectangular phylogram.

    Returns:
        leaf_names : list[str]   – OTT IDs in top-to-bottom leaf order
        x_map      : dict[int, float]  – id(clade) → cumulative branch length
        y_map      : dict[int, float]  – id(clade) → leaf-index y (0-based)
        max_x      : float
    """
    depths = tree.depths(unit_branch_lengths=False)
    max_x  = max(depths.values()) if depths else 1.0
    if max_x == 0.0:
        depths = tree.depths(unit_branch_lengths=True)
        max_x  = max(depths.values()) or 1.0
    x_map = {id(c): d for c, d in depths.items()}

    leaves     = tree.get_terminals()
    leaf_y     = {id(lf): float(i) for i, lf in enumerate(leaves)}
    leaf_names = [_normalize_ott(lf.name) for lf in leaves]

    y_map: dict[int, float] = {}

    def _set_y(clade):
        if not clade.clades:
            y_map[id(clade)] = leaf_y[id(clade)]
        else:
            for c in clade.clades:
                _set_y(c)
            ys = [y_map[id(c)] for c in clade.clades]
            y_map[id(clade)] = (min(ys) + max(ys)) / 2

    _set_y(tree.root)
    return leaf_names, x_map, y_map, max_x


def _tree_to_json(tree, x_map, max_x: float) -> dict:
    """Convert the Bio.Phylo tree to a compact nested dict for JS re-layout.

    Each leaf node:      {"x": <depth_frac>, "i": <leaf_index>}
    Each internal node:  {"x": <depth_frac>, "id": <unique_int>, "ch": [...]}

    Leaf indices match the order returned by _layout() (top-to-bottom).
    """
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    counters = {"leaf": 0, "node": 0}

    def to_node(clade):
        xf = round(x_map[id(clade)] / max_x, 6)
        if not clade.clades:
            i = counters["leaf"]
            counters["leaf"] += 1
            return {"x": xf, "i": i}
        nid = counters["node"]
        counters["node"] += 1
        return {"x": xf, "id": nid, "ch": [to_node(c) for c in clade.clades]}

    return to_node(tree.root)


def _tree_svg(tree, x_map, y_map, max_x, tree_w: int, n_leaves: int,
              display_h: int, name_map: dict | None = None,
              leaf_hit_colors: dict | None = None,
              label_col_w: int = 120) -> str:
    """Rectangular phylogram SVG whose viewBox includes the leaf-label area.

    Labels live at x = tree_w + 4, inside the viewBox (0 to tree_w +
    label_col_w), so no overflow is needed and the SVG width directly
    controls how far right the heatmap starts.
    leaf_hit_colors: {ott_id: fill_hex} — red/blue/orange by hit type.
    """
    row_px   = display_h / n_leaves if n_leaves > 0 else 1.0
    font_sz  = max(2.0, min(6.0, row_px * 0.7))
    total_w  = tree_w + label_col_w

    segs  = []
    texts = []

    for clade in tree.find_clades():
        px = x_map[id(clade)] / max_x * tree_w
        py = (y_map[id(clade)] + 0.5) * row_px

        for child in clade.clades:
            cx = x_map[id(child)] / max_x * tree_w
            cy = (y_map[id(child)] + 0.5) * row_px
            segs.append(f"M{px:.2f},{py:.2f}L{px:.2f},{cy:.2f}L{cx:.2f},{cy:.2f}")

        if not clade.clades and name_map:
            ott_id = _normalize_ott(clade.name)
            label  = name_map.get(ott_id, "")
            if label:
                safe = (label.replace("&", "&amp;")
                             .replace("<", "&lt;")
                             .replace(">", "&gt;"))
                fill = (leaf_hit_colors or {}).get(ott_id, "#555555")
                texts.append(
                    f'<text x="{tree_w + 4:.1f}" y="{py:.1f}" '
                    f'font-size="{font_sz:.1f}" data-base-fs="{font_sz:.1f}" '
                    f'font-family="sans-serif" '
                    f'dominant-baseline="middle" fill="{fill}">{safe}</text>'
                )

    d = " ".join(segs)
    return (
        f'<svg id="tree-svg" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w} {display_h}" '
        f'width="{total_w}" height="{display_h}" '
        f'style="display:block;flex:none;">'
        f'<path d="{d}" fill="none" stroke="#444444" stroke-width="0.4"/>'
        + "".join(texts)
        + "</svg>"
    )


# ── Molecule order ───────────────────────────────────────────────────────────

def _order_from_molecules(pairs: list, molecules_path: str | None) -> list | None:
    """
    Put the molecules in the order the input file lists them.

    The pair list is built from dataframe column order, which follows the merge
    rather than anything the user chose, and was then reordered again by
    similarity clustering. When a batch file exists it states an order, and that
    is the one worth keeping: it is how the person reading the tree expects to
    find their molecules, and it is the same order in both heatmaps.

    Returns None when there is no usable file, so the caller can fall back.
    """
    if not molecules_path:
        return None
    path = Path(molecules_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return None
    if "name" not in df.columns:
        return None

    def key(name: str) -> str:
        return re.sub(r"[\s\W]+", "_", str(name)).strip("_")

    rank, seen = {}, set()
    for name in df["name"].astype(str):
        for form in (name, key(name)):
            if form and form not in seen:
                seen.add(form)
                rank.setdefault(form, len(rank))
    if not rank:
        return None

    known = [pr for pr in pairs if pr[0] in rank or key(pr[0]) in rank]
    rest = [pr for pr in pairs if pr not in known]
    known.sort(key=lambda pr: rank.get(pr[0], rank.get(key(pr[0]), 0)))
    return known + rest


# ── Molecule reordering by Animalia MASST similarity ─────────────────────────

def _reorder_by_animalia(df: pd.DataFrame, pairs: list) -> list:
    """Reorder molecule pairs by Jaccard similarity of Animalia MASST profiles."""
    if "NCBIKingdom" not in df.columns or len(pairs) < 2:
        return pairs

    anim = df[df["NCBIKingdom"].str.lower() == "animalia"]
    if len(anim) < 2:
        return pairs

    masst_cols = [fc for _, fc, _ in pairs if fc and fc in anim.columns]
    if len(masst_cols) < 2:
        return pairs

    # binary hit matrix: rows = molecules, cols = animalia leaves
    M = anim[masst_cols].notna().values.T.astype(float)  # (n_molecules, n_leaves)

    row_sums   = M.sum(axis=1)
    valid_mask = row_sums > 0
    if valid_mask.sum() < 2:
        return pairs

    valid_idx = np.where(valid_mask)[0]
    inval_idx = np.where(~valid_mask)[0]

    try:
        D = pdist(M[valid_mask], metric="jaccard")
        Z = linkage(D, method="average")
        clustered = valid_idx[leaves_list(Z)]
        full_order = list(clustered) + list(inval_idx)
        return [pairs[i] for i in full_order]
    except Exception:
        return pairs


def _cluster_leaf_order(leaf_names: list, df: pd.DataFrame, pairs: list) -> list:
    """Return list of indices into leaf_names sorted by MASST hit similarity.

    Leaves with no MASST hits at all are appended at the end in original order.
    """
    masst_cols = [fc for _, fc, _ in pairs if fc and fc in df.columns]
    n = len(leaf_names)
    if not masst_cols or n < 2:
        return list(range(n))

    in_df = set(df.index)
    M = np.zeros((n, len(masst_cols)), dtype=float)
    for i, ott in enumerate(leaf_names):
        if ott not in in_df:
            continue
        row = df.loc[ott, masst_cols]
        for j, v in enumerate(row):
            if not (v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))):
                M[i, j] = 1.0

    valid_mask = M.sum(axis=1) > 0
    valid_idx  = np.where(valid_mask)[0]
    zero_idx   = np.where(~valid_mask)[0]

    if valid_mask.sum() < 2:
        return list(range(n))
    try:
        D = pdist(M[valid_mask], metric="jaccard")
        Z = linkage(D, method="average")
        clustered = valid_idx[leaves_list(Z)]
        return [int(x) for x in clustered] + [int(x) for x in zero_idx]
    except Exception:
        return list(range(n))


# ── Column helpers ────────────────────────────────────────────────────────────

_CAT_RE = re.compile(r"^(db|nodb|iso|co|st|an)_", re.I)


def _display(col: str, max_len: int = 32) -> str:
    s = col
    for pre in ("masstFlexibleMatch_", "wd_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    s = _CAT_RE.sub("", s)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def _dedup_pairs_by_smiles(pairs: list, smiles_map: dict) -> tuple[list, dict]:
    """Remove pairs whose SMILES duplicates an earlier pair.

    Returns (deduped_pairs, label_overrides) where label_overrides maps
    primary suffix → combined display label (only populated when ≥2 entries
    share a SMILES).
    """
    seen: dict[str, str] = {}     # smiles → primary suf
    label_groups: dict[str, list] = {}
    deduped: list = []

    for suf, fc, wc in pairs:
        smi = smiles_map.get(suf)
        if smi and smi in seen:
            label_groups[seen[smi]].append(_display(fc or wc or suf))
        else:
            deduped.append((suf, fc, wc))
            label_groups[suf] = [_display(fc or wc or suf)]
            if smi:
                seen[smi] = suf

    label_overrides = {
        suf: " / ".join(names)
        for suf, names in label_groups.items()
        if len(names) > 1
    }
    return deduped, label_overrides


def _get_pairs(df: pd.DataFrame):
    """[(suffix, masst_col, wd_col), ...] in column order."""
    flex   = [c for c in df.columns if c.startswith("masstFlexibleMatch_")]
    wd_map = {c[3:]: c for c in df.columns if c.startswith("wd_")}
    pairs, seen = [], set()
    for fc in flex:
        suf = fc[len("masstFlexibleMatch_"):]
        if suf in seen:
            continue
        seen.add(suf)
        pairs.append((suf, fc, wd_map.get(suf)))
    return pairs


# ── PhyloPic helpers ─────────────────────────────────────────────────────────

_PHYLOPIC_TAX_LEVELS = ['NCBIKingdom', 'NCBIPhylum', 'NCBIClass', 'NCBIOrder',
                        'NCBIFamily', 'NCBIGenus', 'NCBISpecies']
_LEVEL_DISPLAY = {
    'NCBISpecies': 'Species', 'NCBIGenus': 'Genus', 'NCBIFamily': 'Family',
    'NCBIOrder': 'Order', 'NCBIClass': 'Class', 'NCBIPhylum': 'Phylum',
    'NCBIKingdom': 'Kingdom',
}

# Specificity ordering: a match at level L only counts as evidence at resolution L.
# Used to avoid crediting kingdom-level MASST hits when evaluating family enrichment.
_TAX_FINENESS: dict[str, int] = {
    "kingdom": 0, "phylum": 1, "class": 2, "order": 3,
    "family": 4, "genus": 5, "species": 6, "subspecies": 7,
}
_PHYLOPIC_COL_TO_FINENESS: dict[str, int] = {
    "NCBIKingdom": 0, "NCBIPhylum": 1, "NCBIClass": 2, "NCBIOrder": 3,
    "NCBIFamily": 4, "NCBIGenus": 5, "NCBISpecies": 6,
}

# Maximum level index (into _PHYLOPIC_TAX_LEVELS) for decomposition / enrichment per kingdom.
# Kingdoms not listed here use the full range (NCBISpecies = index 6).
# Bacteria and fungi have too much taxonomic noise below family to be useful.
_KINGDOM_MAX_LV_IDX: dict[str, int] = {
    "bacteria":       4,   # NCBIFamily
    "bacillati":      4,
    "pseudomonadati": 4,
    "fusobacteriati": 4,
    "fungi":          4,
}
_KINGDOM_MIN_SAMPLES = 10   # kingdoms with fewer total tree leaves stay at kingdom level

# Curated PhyloPic lookup overrides for broad / commonly misidentified taxa.
# Keys are lowercase taxon names as they appear in the data; values are the
# search term sent to the PhyloPic autocomplete API.
_PHYLOPIC_NAME_OVERRIDES: dict[str, str] = {
    "animalia":       "Vertebrata",
    "plantae":        "arabidopsis",
    "fungi":          "Ascomycota",
    "bacteria":       "Bacteria",
    "bacillati":      "Bacteria",
    "pseudomonadati": "Pseudomonas",
    "chromista":      "Diatom",
    "protozoa":       "Amoeba",
    "fusobacteriati": "Fusobacterium",
}


_PUBLIC_DOMAIN_LICENSES = (
    "https://creativecommons.org/publicdomain/zero/1.0/",   # CC0
    "https://creativecommons.org/publicdomain/mark/1.0/",   # PDM
)

# Disk cache for PhyloPic thumbnails — avoids re-fetching on repeat runs.
_PHYLOPIC_CACHE_DIR = Path.home() / ".cache" / "metabotree_phylopic"


def _phylopic_cache_path(taxon_name: str, size: int) -> Path:
    slug = re.sub(r"[^\w]+", "_", taxon_name.lower()).strip("_")
    return _PHYLOPIC_CACHE_DIR / f"{slug}_{size}.b64"


def _fetch_phylopic_b64(taxon_name: str, size: int = 40) -> str | None:
    """Fetch a CC0/PDM PhyloPic silhouette. Returns base64 PNG or None if none available.

    Tries taxon_name first, then falls back to the genus (first word) if the
    species is not in PhyloPic.  Results are cached in
    ~/.cache/metabotree_phylopic/ so repeat runs skip the API.  A cached
    sentinel 'NONE' means all fallbacks were tried and nothing was found.
    """
    cache_path = _phylopic_cache_path(taxon_name, size)

    # --- Check disk cache ---
    if cache_path.exists():
        data = cache_path.read_text().strip()
        return None if data == "NONE" else data

    # Build list of names to try: specific name first, then genus if multi-word
    names_to_try = [taxon_name]
    parts = taxon_name.strip().split()
    if len(parts) > 1:
        names_to_try.append(parts[0])   # genus fallback

    # --- Fetch from API ---
    result: str | None = None
    try:
        import requests as _req
        base = "https://api.phylopic.org"
        hdrs = {"Accept": "application/json"}

        # 1. Get current build (shared across all name attempts)
        build = _req.get(base, headers=hdrs, timeout=8).json().get("build", 538)

        for attempt_name in names_to_try:
            # 2. Autocomplete → only accept an exact case-insensitive match.
            # Prefix matches (e.g. "Mino" → "minog morski" = sea lamprey) return
            # completely wrong taxa; fall back to the raw lowercase name instead.
            ac = _req.get(f"{base}/autocomplete", params={"query": attempt_name},
                          timeout=8).json().get("matches", [])
            exact = next((m for m in ac if m.lower() == attempt_name.lower()), None)
            lookup = exact if exact else attempt_name.lower()

            # 3. Find node UUID
            nd = _req.get(f"{base}/nodes",
                          params={"filter_name": lookup, "build": build, "page": 0},
                          headers=hdrs, timeout=8).json()
            node_hrefs = nd.get("_links", {}).get("items", [])
            if not node_hrefs:
                if attempt_name == names_to_try[-1]:
                    print(f"  PhyloPic '{taxon_name}': node not found — skipping")
                continue
            node_uuid = node_hrefs[0]["href"].split("/nodes/")[-1].split("?")[0]

            # 3b. Verify the returned node actually carries the queried name.
            # filter_name can return synonym nodes whose canonical name is
            # completely different; if our name isn't listed, the image is wrong.
            node_info = _req.get(f"{base}/nodes/{node_uuid}",
                                 params={"build": build}, headers=hdrs, timeout=8).json()
            node_names_lower = {
                n["text"].lower()
                for name_list in node_info.get("names", [])
                for n in name_list
            }
            if attempt_name.lower() not in node_names_lower:
                print(f"  PhyloPic '{attempt_name}': node name mismatch "
                      f"(got {sorted(node_names_lower)[:4]}) — skipping")
                continue

            # 4. Find a public-domain image (CC0 first, then PDM)
            img_uuid = None
            for lic in _PUBLIC_DOMAIN_LICENSES:
                im = _req.get(f"{base}/images",
                              params={"filter_node": node_uuid, "build": build,
                                      "page": 0, "filter_license": lic},
                              headers=hdrs, timeout=8).json()
                items = im.get("_links", {}).get("items", [])
                if items:
                    img_uuid = items[0]["href"].split("/images/")[-1].split("?")[0]
                    break

            if img_uuid is None:
                if attempt_name == names_to_try[-1]:
                    print(f"  PhyloPic '{taxon_name}': no CC0/PDM image — skipping")
                continue

            # 5. Fetch thumbnail closest to requested size
            meta = _req.get(f"{base}/images/{img_uuid}",
                            params={"build": build}, headers=hdrs, timeout=8).json()
            thumbs = meta.get("_links", {}).get("thumbnailFiles", [])
            if thumbs:
                best = min(thumbs,
                           key=lambda t: abs(int(t["sizes"].split("x")[0]) - size))
                r_img = _req.get(best["href"], timeout=8)
                r_img.raise_for_status()
                result = base64.b64encode(r_img.content).decode()
                if attempt_name != taxon_name:
                    print(f"  PhyloPic '{taxon_name}': found via genus '{attempt_name}'")
                break   # success — stop trying fallbacks

    except Exception as exc:
        print(f"  PhyloPic '{taxon_name}': {exc}")

    # --- Write to disk cache ---
    try:
        _PHYLOPIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result if result is not None else "NONE")
    except OSError as exc:
        print(f"  PhyloPic cache write failed: {exc}")

    return result


def _select_phylopic_groups(df: pd.DataFrame, leaf_names: list,
                             leaf_hits: list, n_max: int = 20) -> list:
    """
    Pick ≤ n_max taxonomic groups to annotate with PhyloPic silhouettes.

    Algorithm:
      1. For every MASST-hit leaf, create an initial silhouette at the level
         matching the MASST specificity of that hit (kingdom→kingdom,
         family→family, leaf-level→species/subspecies).  Skip kingdoms with
         < 20 leaves in the tree.
      2. For each silhouette, try to move it to the next coarser level as long
         as ≥ 50% of all leaves in that coarser group have any MASST hit.
      3. Deduplicate — many individual groups converge to the same taxon.
      4. If the count still exceeds n_max, repeatedly find the merge that moves
         one silhouette one level coarser and introduces the smallest fraction
         of non-hit leaves (relative to the new group's total leaves).  Prefer
         merges that immediately combine with an existing silhouette (count -1)
         over moves that only broaden an isolated group.  Repeat until ≤ n_max.

    Returns list of dicts:
      label       – group taxon name (for display)
      lookup_name – taxon name to use for PhyloPic API query
      leaves      – all leaf indices in this group (for Y-span)
      level       – human-readable rank label
    """
    _KINGDOM_MIN_LEAF = 20   # skip kingdoms with fewer total tree leaves
    _DENSITY_THRESHOLD = 0.80

    masst_hit_set = {i for i, hits in enumerate(leaf_hits) if any(h[1] for h in hits)}
    if not masst_hit_set:
        return []

    # ── Build OTT→taxonomy value dicts for each level ────────────────────────
    def _level_dict(level: str) -> dict:
        if level not in df.columns:
            return {}
        d: dict[str, str] = {}
        for k, v in df[level].items():
            s = str(v).strip()
            if s and s.lower() not in ("nan", "na", "none", ""):
                nk = _normalize_ott(str(k))
                if nk:
                    d[nk] = s
        return d

    ldicts = [_level_dict(lv) for lv in _PHYLOPIC_TAX_LEVELS]

    # ── Identify small kingdoms (< 20 tree leaves) ───────────────────────────
    king_ld = ldicts[0]
    kingdoms_all: dict[str, list] = {}
    for i, ott in enumerate(leaf_names):
        kv = king_ld.get(ott, "").strip()
        if kv:
            kingdoms_all.setdefault(kv, []).append(i)
    small_kingdoms: set[str] = {
        kv for kv, idxs in kingdoms_all.items() if len(idxs) < _KINGDOM_MIN_LEAF
    }

    # ── Precompute taxon→{all_leaves, hit_leaves} for each (lv_idx, taxon) ──
    # Excludes leaves from small kingdoms so density denominators stay clean.
    TaxInfo = dict  # {"all": list[int], "hits": list[int]}
    taxon_info: dict[tuple, TaxInfo] = {}
    for i, ott in enumerate(leaf_names):
        kg = king_ld.get(ott, "").strip()
        if kg in small_kingdoms:
            continue
        is_hit = i in masst_hit_set
        for lv_idx, lv_ld in enumerate(ldicts):
            taxon = lv_ld.get(ott, "").strip()
            if not taxon:
                continue
            key = (lv_idx, taxon)
            if key not in taxon_info:
                taxon_info[key] = {"all": [], "hits": []}
            taxon_info[key]["all"].append(i)
            if is_hit:
                taxon_info[key]["hits"].append(i)

    def _density(lv_idx: int, taxon: str) -> float:
        info = taxon_info.get((lv_idx, taxon))
        if not info or not info["all"]:
            return 0.0
        return len(info["hits"]) / len(info["all"])

    def _parent_taxon(leaf_idx: int, parent_lv: int) -> str | None:
        ott = leaf_names[leaf_idx]
        t = ldicts[parent_lv].get(ott, "").strip()
        return t if t and t.lower() not in ("nan", "na", "none", "") else None

    # ── Map MASST fineness → _PHYLOPIC_TAX_LEVELS index ──────────────────────
    # subspecies (7) collapses to species (6), the finest NCBI column we have.
    _fin_to_lvidx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6}

    # ── Step 1: Initial silhouette per (lv_idx, taxon) ───────────────────────
    groups: set[tuple[int, str]] = set()   # (lv_idx, taxon_name)
    for i in masst_hit_set:
        ott = leaf_names[i]
        kg = king_ld.get(ott, "").strip()
        if kg in small_kingdoms:
            continue
        # Finest MASST level for this leaf
        best_f = max(
            (_TAX_FINENESS.get(h[1], -1) for h in leaf_hits[i] if h[1]),
            default=-1,
        )
        if best_f < 0:
            continue
        target_lv = _fin_to_lvidx[best_f]
        # Walk down from target_lv to find the finest level that has a valid taxon
        for lv in range(target_lv, -1, -1):
            t = ldicts[lv].get(ott, "").strip()
            if t and t.lower() not in ("nan", "na", "none", ""):
                groups.add((lv, t))
                break

    if not groups:
        return []

    # ── Step 2: Move each group coarser while parent ≥ 50 % hit density ──────
    # Additional stopping criteria per upward step:
    #   a) The parent must gain ≥ 20 % more MASST hits relative to the current
    #      group (e.g. 10 hits at family → order must add ≥ 2 hits).
    #   b) Exception to (a): if the parent adds ≥ 1 new hit AND the density
    #      does not drop by more than 5 pp, the move is still allowed.
    expanded: set[tuple[int, str]] = set()
    for lv_idx, taxon in groups:
        cur_lv, cur_tax = lv_idx, taxon
        while cur_lv > 0:
            info = taxon_info.get((cur_lv, cur_tax))
            if not info or not info["all"]:
                break
            parent_lv = cur_lv - 1
            pt = _parent_taxon(info["all"][0], parent_lv)
            if not pt:
                break
            pinfo = taxon_info.get((parent_lv, pt))
            if not pinfo or not pinfo["all"]:
                break

            # Check 1: density threshold
            cur_density = len(info["hits"]) / len(info["all"])
            par_density = len(pinfo["hits"]) / len(pinfo["all"])
            if par_density < _DENSITY_THRESHOLD:
                break

            # Check 2: coarsening must add ≥ 20 % more MASST hits
            cur_hits = len(info["hits"])
            new_hits = len(pinfo["hits"]) - cur_hits
            if new_hits < 0.20 * cur_hits:
                # Pass-through: parent covers exactly the same leaves (just a
                # higher-rank label for the same set) — always allow.
                same_leaves = len(pinfo["all"]) == len(info["all"])
                # Exception: ≥ 1 new hit and density drop ≤ 5 pp
                exc = new_hits >= 1 and (cur_density - par_density) <= 0.05
                if not same_leaves and not exc:
                    break

            # Check 3: when promoting to Kingdom, the current group must
            # account for ≥ 5 % of the Kingdom's hits — prevents a small
            # phylum from claiming a Kingdom whose density is driven almost
            # entirely by a sibling phylum (e.g. Mollusca free-riding on
            # Chordata's density within Animalia).
            if parent_lv == 0 and cur_hits < 0.05 * len(pinfo["hits"]):
                break

            cur_lv, cur_tax = parent_lv, pt
        expanded.add((cur_lv, cur_tax))

    groups = expanded

    # ── Step 3-4: Reduce to ≤ n_max by greedy upward merges ─────────────────
    # Each iteration picks the single move (current → parent) that introduces
    # the smallest non-hit fraction in the parent group, preferring moves that
    # immediately merge with an existing group (guaranteed count reduction).
    max_iters = len(groups) * (len(_PHYLOPIC_TAX_LEVELS) + 1)
    for _ in range(max_iters):
        if len(groups) <= n_max:
            break

        best: tuple | None = None
        best_reduces = False
        best_cost = float("inf")

        for lv_idx, taxon in list(groups):
            if lv_idx == 0:
                continue  # can't move kingdom higher
            info = taxon_info.get((lv_idx, taxon))
            if not info or not info["all"]:
                continue
            parent_lv = lv_idx - 1
            pt = _parent_taxon(info["all"][0], parent_lv)
            if not pt:
                continue
            pinfo = taxon_info.get((parent_lv, pt))
            if not pinfo or not pinfo["all"]:
                continue
            non_hit_frac = 1.0 - len(pinfo["hits"]) / len(pinfo["all"])
            reduces = (parent_lv, pt) in groups
            # Never create a new Kingdom-level group in step 4.  If a Kingdom
            # group is warranted it will already be present from step 1/2.
            # Allowing it here lets Chordata "prepare" an Animalia group that
            # small phyla then flood into via reduces=True (bypassing the
            # coverage check), which is exactly the over-promotion we want to
            # prevent.
            if not reduces and parent_lv == 0:
                continue
            # Prefer: (1) moves that reduce count, (2) lower non-hit fraction
            if (reduces and not best_reduces) or \
               (reduces == best_reduces and non_hit_frac < best_cost):
                best = (lv_idx, taxon, parent_lv, pt)
                best_reduces = reduces
                best_cost = non_hit_frac

        if best is None:
            break
        lv, tax, plv, pt = best
        groups.discard((lv, tax))
        groups.add((plv, pt))

    # ── Build result ──────────────────────────────────────────────────────────
    result = []
    for lv_idx, taxon in sorted(groups, key=lambda g: (g[0], g[1])):
        level_name = _PHYLOPIC_TAX_LEVELS[lv_idx]
        info = taxon_info.get((lv_idx, taxon), {})
        leaves = info.get("all") or info.get("hits") or []

        override = _PHYLOPIC_NAME_OVERRIDES.get(taxon.lower())
        lookup_name = override if override else taxon

        result.append({
            "label":       taxon,
            "lookup_name": lookup_name,
            "leaves":      leaves,
            "level":       _LEVEL_DISPLAY.get(level_name, level_name or ""),
            "lv_idx":      lv_idx,
        })

    # ── Remove nested groups ───────────────────────────────────────────────────
    # Group i is suppressed when either:
    #   (a) i is at a strictly finer taxonomic level AND shares at least one
    #       leaf with coarser group j  (handles NaN taxonomy rows via intersection
    #       rather than strict subset), OR
    #   (b) i's leaf-index range [min, max] is fully contained within j's range
    #       and j spans a strictly wider range — covers paraphylogenetic cases
    #       where an organism has NaN at the intermediate rank but still lands
    #       visually inside a broader group's bracket.
    if len(result) > 1:
        leaf_sets  = [frozenset(g["leaves"]) for g in result]
        idx_ranges = [(min(g["leaves"]), max(g["leaves"])) for g in result]
        to_remove: set[int] = set()
        for i in range(len(result)):
            if i in to_remove:
                continue
            for j in range(len(result)):
                if i == j or j in to_remove:
                    continue
                # (a) finer level shares leaves with coarser group
                if result[i]["lv_idx"] > result[j]["lv_idx"] and leaf_sets[i] & leaf_sets[j]:
                    to_remove.add(i)
                    break
                # (b) i's index range is strictly inside j's broader range
                min_i, max_i = idx_ranges[i]
                min_j, max_j = idx_ranges[j]
                if (max_j - min_j > max_i - min_i
                        and min_i >= min_j and max_i <= max_j):
                    to_remove.add(i)
                    break
        result = [g for i, g in enumerate(result) if i not in to_remove]

    # ── Family cap: ≤ 2 silhouettes per NCBIFamily when total ≥ 10 ────────────
    # Look up the NCBIFamily of each group via its first leaf.  Groups at Order
    # level or coarser (lv_idx ≤ 3) span many families so we don't cap them;
    # groups at Family, Genus or Species level that share a family are capped.
    _FAM_LV = 4   # index of NCBIFamily in _PHYLOPIC_TAX_LEVELS
    if len(result) >= 10:
        family_ld = ldicts[_FAM_LV]

        def _group_family(g: dict) -> str | None:
            if not g["leaves"]:
                return None
            ott = leaf_names[g["leaves"][0]]
            fam = family_ld.get(ott, "").strip()
            return fam if fam and fam.lower() not in ("nan", "na", "none", "") else None

        from collections import defaultdict
        fam_buckets: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(result):
            fam = _group_family(g)
            if fam:
                fam_buckets[fam].append(i)

        cap_remove: set[int] = set()
        for fam, idxs in fam_buckets.items():
            if len(idxs) <= 2:
                continue
            # Keep the 2 coarsest (lowest lv_idx); ties broken by most leaves.
            ranked = sorted(idxs, key=lambda i: (result[i]["lv_idx"], -len(result[i]["leaves"])))
            for i in ranked[2:]:
                cap_remove.add(i)

        result = [g for i, g in enumerate(result) if i not in cap_remove]

    return result


# ── PNG heatmap ───────────────────────────────────────────────────────────────

def _val(v) -> str | None:
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "NA", "no_data", "<NA>") else s.lower()


def _build_png(leaf_names: list, df: pd.DataFrame, pairs: list,
               row_h: float, cell_w: int,
               strip1_w: int, strip2_w: int,
               strip1_col: str, strip2_col: str,
               strip1_hex_map: dict, strip1_rare: set, strip1_def_hex: str,
               strip2_hex_map: dict, strip2_rare: set, strip2_def_hex: str,
               display_h: int | None = None,
               col_gap: int = 2, strip_gap: int = 0,
               # legacy compat (ignored when strip* args supplied)
               king_w: int = 0, class_w: int = 0,
               class_hex_map: dict | None = None, rare_classes: set | None = None,
               ) -> tuple[bytes, int, int]:
    """
    Returns (png_bytes, total_pixel_width, display_height_px).
    Builds at 1 px/row internally, then resizes with NEAREST (no blending).
    Nearest-neighbour picks the exact source leaf for each output row, so
    there is no colour bleed between adjacent leaves.

    Column layout: [strip1 | strip2 | mol_1_masst | mol_1_wd | ...]
    """
    total_w  = strip1_w + strip_gap + strip2_w + strip_gap + len(pairs) * (cell_w * 2 + col_gap)
    n_leaves = len(leaf_names)
    img      = np.full((n_leaves, total_w, 3), 255, dtype=np.uint8)

    in_meta   = set(df.index)
    s1_dict   = df[strip1_col].to_dict() if strip1_col and strip1_col in df.columns else {}
    s2_dict   = df[strip2_col].to_dict() if strip2_col and strip2_col in df.columns else {}
    ms_dicts  = {fc: df[fc].to_dict() for _, fc, _ in pairs if fc and fc in df.columns}
    wd_dicts  = {wc: df[wc].to_dict() for _, _, wc in pairs if wc and wc in df.columns}

    s1_rgb_map = {k.lower(): _h2rgb(v) for k, v in strip1_hex_map.items()}
    s1_rgb_def = _h2rgb(strip1_def_hex)
    s2_rgb_map = {k.lower(): _h2rgb(v) for k, v in strip2_hex_map.items()}
    s2_rgb_def = _h2rgb(strip2_def_hex)

    for i, ott in enumerate(leaf_names):
        if ott not in in_meta:
            img[i] = _LGRAY
            continue

        sv1 = _val(s1_dict.get(ott))
        img[i, 0:strip1_w] = s1_rgb_map.get((sv1 or "").lower(), s1_rgb_def)

        sv2 = _val(s2_dict.get(ott))
        if sv2 and sv2.lower() not in strip2_rare:
            s2_color = s2_rgb_map.get(sv2.lower(), s2_rgb_def)
        else:
            s2_color = s2_rgb_def
        img[i, strip1_w + strip_gap:strip1_w + strip_gap + strip2_w] = s2_color

        for j, (_, fc, wc) in enumerate(pairs):
            x0 = strip1_w + strip_gap + strip2_w + strip_gap + j * (cell_w * 2 + col_gap)

            mv = _val(ms_dicts.get(fc, {}).get(ott)) if fc else None
            if mv and mv in _MS_RGB:
                img[i, x0:x0 + cell_w] = _MS_RGB[mv]

            wv = _val(wd_dicts.get(wc, {}).get(ott)) if wc else None
            if wv and wv in _WD_RGB:
                img[i, x0 + cell_w:x0 + cell_w * 2] = _WD_RGB[wv]

    if display_h is None:
        display_h = max(1, round(n_leaves * row_h))
    display_h = max(1, display_h)

    if display_h >= n_leaves:
        # Upscale: NEAREST — each leaf expands cleanly, no blending needed
        pil_img = Image.fromarray(img, "RGB")
        if display_h != n_leaves:
            pil_img = pil_img.resize((total_w, display_h), Image.NEAREST)
    else:
        # Downscale: for each output row collect all contributing input rows and
        # pick the darkest pixel per column (lowest RGB sum = most colourful hit).
        # This preserves real hits rather than silently dropping leaves that
        # happen to fall between NEAREST sample points.
        out_img = np.full((display_h, total_w, 3), 255, dtype=np.uint8)
        col_idx = np.arange(total_w)
        for out_y in range(display_h):
            in_start = int(out_y * n_leaves / display_h)
            in_end   = max(in_start + 1,
                           int(math.ceil((out_y + 1) * n_leaves / display_h)))
            in_end   = min(in_end, n_leaves)
            chunk    = img[in_start:in_end]              # (k, W, 3)
            best     = chunk.sum(axis=2).argmin(axis=0)  # (W,) — darkest row per column
            out_img[out_y] = chunk[best, col_idx]
        pil_img = Image.fromarray(out_img, "RGB")

    buf = io.BytesIO()
    pil_img.save(buf, "PNG", compress_level=6)
    return buf.getvalue(), total_w, display_h


# ── Column-header SVG ─────────────────────────────────────────────────────────

def _header_svg(pairs: list, cell_w: int,
                strip1_w: int, strip2_w: int,
                strip1_label: str, strip2_label: str,
                header_h: int, img_w: int,
                mol_imgs: dict | None = None,
                label_overrides: dict | None = None,
                col_gap: int = 2,
                font_sz: int = 9,
                strip_gap: int = 0,
                # legacy compat
                king_w: int = 0, class_w: int = 0,
                ) -> str:
    """
    mol_imgs: dict suffix → base64 PNG string, rendered at 2*cell_w wide.
    Images are placed in the top portion of the header; text labels rotate
    up from the bottom; colour squares sit at the very bottom edge.
    """
    els = []
    mol_img_w   = cell_w * 2 + col_gap
    sq_reserved = 12                       # pixels at the bottom for colour squares
    img_h       = max(10, header_h // 3)  # height of molecule image strip
    img_top     = header_h - sq_reserved - img_h   # images sit just above the squares
    label_y     = img_top - 6             # text anchor; labels rotate upward from here

    # ── rotated text labels (anchored above the image strip) ─────────────
    def label(x_center, text, color="#333333"):
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<text transform="translate({x_center},{label_y}) rotate(-90)" '
            f'text-anchor="start" dominant-baseline="middle" '
            f'font-size="{font_sz}" font-family="sans-serif" fill="{color}">{safe}</text>'
        )

    els.append(label(strip1_w / 2, strip1_label))
    if strip2_w > 0:
        els.append(label(strip1_w + strip_gap + strip2_w / 2, strip2_label))

    for j, (suf, fc, wc) in enumerate(pairs):
        cx   = strip1_w + strip_gap + strip2_w + strip_gap + j * (cell_w * 2 + col_gap) + cell_w
        name = (label_overrides or {}).get(suf) or _display(fc or wc or suf)
        els.append(label(cx, name))

    # ── molecule structure images (below labels, above squares) ──────────
    if mol_imgs:
        for j, (suf, _, _) in enumerate(pairs):
            b64 = mol_imgs.get(suf)
            if not b64:
                continue
            x = strip1_w + strip_gap + strip2_w + strip_gap + j * (cell_w * 2 + col_gap)
            els.append(
                f'<image href="data:image/png;base64,{b64}" '
                f'x="{x}" y="{img_top}" width="{mol_img_w}" height="{img_h}" '
                f'preserveAspectRatio="xMidYMid meet"/>'
            )

    # ── coloured squares at bottom edge ───────────────────────────────────
    for j in range(len(pairs)):
        xm = strip1_w + strip_gap + strip2_w + strip_gap + j * (cell_w * 2 + col_gap) + cell_w // 2
        xw = xm + cell_w
        for xpos, col in ((xm, "#cc0000"), (xw, "#0000cc")):
            els.append(
                f'<rect x="{xpos - 3}" y="{header_h - 5}" '
                f'width="6" height="4" fill="{col}" rx="1"/>'
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{img_w}" height="{header_h}" '
        f'style="display:block;overflow:visible;">'
        + "".join(els)
        + "</svg>"
    )


# ── Legend ────────────────────────────────────────────────────────────────────

def _legend_html(strip1_hex_map: dict, strip1_label: str, strip1_rare: set | None = None,
                 strip2_hex_map: dict | None = None, strip2_label: str = "",
                 strip2_rare: set | None = None, strip2_def_hex: str = _CLASS_DEFAULT_HEX,
                 present_kingdoms: set | None = None,
                 # legacy compat
                 class_hex_map: dict | None = None, rare_classes: set | None = None,
                 ) -> str:
    def swatches(hexd, attr):
        return "".join(
            f'<div class="lr" {attr}="{k}"><div class="sw" style="background:{hexd[k]}"></div>'
            f'<span>{k}</span></div>'
            for k in ("subspecies", "species", "genus", "family", "order", "class")
            if k in hexd
        )

    def strip_rows(hex_map, attr, rare, def_hex, max_items=15):
        items = sorted(hex_map.items())[:max_items]
        more  = len(hex_map) - len(items)
        rows  = "".join(
            f'<div class="lr" {attr}="{k.lower()}"><div class="sw" style="background:{v}"></div>'
            f'<span>{k.capitalize() if len(k) < 20 else k}</span></div>'
            for k, v in items
        )
        if more > 0:
            rows += f'<div class="lr"><span style="color:#999">+{more} more…</span></div>'
        if rare:
            rows += (
                f'<div class="lr" {attr}="__other__"><div class="sw" style="background:{def_hex};'
                f'border:1px solid #aaa"></div><span>Other (&lt;4 leaves)</span></div>'
            )
        return rows

    # Filter strip1 if it is the kingdom strip and present_kingdoms is given
    if strip1_hex_map is _KINGDOM_HEX or set(strip1_hex_map.keys()) == set(_KINGDOM_HEX.keys()):
        s1_filtered = {k: v for k, v in strip1_hex_map.items()
                       if present_kingdoms is None or k in present_kingdoms}
    else:
        s1_filtered = strip1_hex_map

    s1_rows = strip_rows(s1_filtered, 'data-leg-s1', strip1_rare or set(),
                         _KINGDOM_DEFAULT_HEX if not strip1_rare else _CLASS_DEFAULT_HEX)
    s1_section = f'<b data-leg-hdr="s1">{strip1_label}</b>{s1_rows}' if s1_rows else ""

    s2_section = ""
    if strip2_hex_map is not None:
        s2_rows = strip_rows(strip2_hex_map, 'data-leg-s2', strip2_rare or set(), strip2_def_hex)
        if s2_rows:
            s2_section = f'<b data-leg-hdr="s2">{strip2_label}</b>{s2_rows}'

    return f"""<div id="leg">
  <b data-leg-hdr="masst">MASST hit (red)</b>{swatches(_MASST_HEX, 'data-leg-masst')}
  <b data-leg-hdr="wd">Wikidata (blue)</b>{swatches(_WD_HEX, 'data-leg-wd')}
  <div id="leg-s1-section">{s1_section}</div>
  <div id="leg-s2-section">{s2_section}</div>
  <div class="lr" style="margin-top:4px">
    <div class="sw" style="background:#eeeeee;border:1px solid #aaa"></div>
    <span>not in data</span>
  </div>
</div>"""


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MetaboTree Heatmap</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:white;overflow:hidden;font-family:sans-serif;font-size:11px}}
#outer{{
  width:100vw;height:100vh;overflow:hidden;
  cursor:grab;user-select:none;
}}
#outer.dragging{{cursor:grabbing}}
#canvas{{
  display:inline-block;
  position:relative;
  transform-origin:0 0;
  background:white;
  isolation:isolate;
}}
#header-row{{display:flex;align-items:flex-end;background:#f5f5f5;border-bottom:1px solid #ccc;position:relative;z-index:2;}}
#tree-spacer{{flex:none;border-right:2px solid #bbb}}
#content-row{{display:flex;position:relative;z-index:1;}}
#tree-col{{flex:none;border-right:2px solid #bbb;background:white}}
#hm{{display:block;image-rendering:pixelated;image-rendering:crisp-edges}}
/* Row selection marker — lives inside #canvas so it zooms/pans with it */
#row-marker{{
  position:absolute;left:0;width:100%;z-index:3;
  background:rgba(255,210,0,0.18);
  pointer-events:none;display:none;box-sizing:border-box;
}}
/* Second row marker — for the clustered heatmap (left/width set by JS) */
#row-marker2{{
  position:absolute;z-index:3;
  background:rgba(255,210,0,0.18);
  pointer-events:none;display:none;box-sizing:border-box;
}}
/* Solid centre line — shared by all row/col markers */
#row-marker::after,#row-marker2::after,.row-m::after{{
  content:'';display:block;
  position:absolute;left:0;right:0;
  top:50%;transform:translateY(-50%);
  height:1px;background:rgba(255,120,0,0.9);
}}
/* Multiple-row markers (molecule click) — lives in #mol-markers (z:3) */
#mol-markers{{position:absolute;top:0;left:0;pointer-events:none;z-index:3;}}
.row-m{{
  position:absolute;left:0;width:100%;
  background:rgba(255,210,0,0.18);
  pointer-events:none;box-sizing:border-box;
}}
/* Vertical column marker — lives in #col-markers (z:0), behind content (z:1) */
#col-markers{{position:absolute;top:0;left:0;pointer-events:none;z-index:0;}}
.col-m{{
  position:absolute;
  background:rgba(255,210,0,0.18);
  pointer-events:none;box-sizing:border-box;
}}
.col-m::after{{
  content:'';display:block;
  position:absolute;top:0;bottom:0;
  left:50%;transform:translateX(-50%);
  width:1px;background:rgba(255,120,0,0.9);
}}
/* Tooltip — fixed to viewport */
#tooltip{{
  position:fixed;background:white;
  border:1px solid #ccc;border-radius:5px;
  padding:9px 12px;font-size:11px;line-height:1.75;
  box-shadow:0 3px 12px rgba(0,0,0,.22);
  z-index:100;display:none;
  max-width:420px;max-height:75vh;overflow-y:auto;
  pointer-events:all;
}}
/* Hover tip for green leaf-hit dots — independent of click tooltip */
#dot-tip{{
  position:fixed;background:white;
  border:1px solid #aaa;border-radius:4px;
  padding:6px 10px;font-size:11px;line-height:1.6;
  box-shadow:0 2px 8px rgba(0,0,0,.18);
  z-index:101;display:none;pointer-events:none;
}}
.tt-sep{{border-top:1px solid #e5e5e5;margin:6px 0;}}
.tt-mols{{display:flex;flex-wrap:wrap;gap:5px;margin-top:5px;}}
.tt-mol{{display:flex;flex-direction:column;align-items:center;width:66px;}}
.tt-mol img{{width:62px;height:62px;object-fit:contain;border:1px solid #e8e8e8;border-radius:3px;background:#fafafa;}}
.tt-mol-name{{font-size:8px;text-align:center;line-height:1.3;margin-top:2px;word-break:break-word;color:#444;}}
.tt-mol-lvl{{font-size:7.5px;color:#777;text-align:center;}}
.tt-mol-ph{{width:62px;height:62px;background:#f0f0f0;border:1px solid #e0e0e0;border-radius:3px;flex:none;}}
.tt-mol-wide{{display:flex;flex-direction:row;align-items:flex-start;gap:7px;width:100%;margin-bottom:4px;}}
.tt-mol-wide img{{width:62px;height:62px;object-fit:contain;border:1px solid #e8e8e8;border-radius:3px;background:#fafafa;flex:none;}}
.tt-mol-wide .tt-mol-ph{{border:1px solid #e8e8e8;background:#fafafa;}}
.tt-mol-info{{flex:1;min-width:0;padding-top:1px;}}
.tt-mol-info .tt-mol-name{{text-align:left;}}
.tt-mol-info .tt-mol-lvl{{text-align:left;}}
.tt-mol-src{{font-size:8px;color:#555;margin-top:3px;word-break:break-word;line-height:1.4;}}
/* Legend: fixed to viewport, never zoomed */
#leg{{
  position:fixed;bottom:14px;right:14px;
  background:white;border:1px solid #ccc;border-radius:4px;
  padding:8px 12px;font-size:10px;
  box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:50;
  max-height:90vh;overflow-y:auto;
  pointer-events:all;
}}
#leg b{{display:block;margin:6px 0 2px}}
.lr{{display:flex;align-items:center;gap:5px;margin:1px 0}}
.sw{{width:13px;height:9px;border:1px solid #ddd;flex:none}}
#zoom-hint{{
  position:fixed;top:10px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,.55);color:white;
  padding:4px 12px;border-radius:12px;font-size:11px;
  pointer-events:none;opacity:1;transition:opacity 1s;
}}
/* ── Fullscreen button ── */
#fullscreen-btn{{
  position:fixed;top:14px;right:14px;
  background:white;border:1px solid #ccc;border-radius:4px;
  padding:5px 10px;font-size:11px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:51;
}}
#fullscreen-btn:hover{{background:#e8e8e8;}}
/* ── Filter panel ── */
#filter-panel{{
  position:fixed;top:14px;left:14px;
  background:white;border:1px solid #ccc;border-radius:4px;
  padding:8px 10px;font-size:11px;
  box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:50;
  min-width:170px;max-width:220px;
  max-height:82vh;overflow-y:auto;
  pointer-events:all;
}}
#filter-panel b{{display:block;margin-bottom:5px;}}
#f-level{{width:100%;margin-bottom:5px;font-size:11px;}}
#f-val-wrap{{display:none;}}
#f-values{{width:100%;font-size:10px;margin-bottom:5px;}}
#f-mols{{width:100%;font-size:10px;margin-bottom:5px;}}
#f-match-levels{{width:100%;font-size:10px;margin-bottom:5px;}}
.f-btn-row{{display:flex;gap:4px;flex-wrap:wrap;}}
.f-btn{{flex:1;padding:3px 4px;font-size:10px;cursor:pointer;
  border:1px solid #bbb;border-radius:3px;background:#f5f5f5;}}
.f-btn:hover{{background:#e8e8e8;}}
#f-btn-show{{border-color:#3a9e3a;color:#2a6e2a;}}
#f-btn-excl{{border-color:#cc5500;color:#993300;}}
#f-btn-mol-show{{border-color:#3a9e3a;color:#2a6e2a;}}
#f-btn-mol-excl{{border-color:#cc5500;color:#993300;}}
#f-btn-ml-show{{border-color:#3a9e3a;color:#2a6e2a;}}
#f-btn-ml-excl{{border-color:#cc5500;color:#993300;}}
#f-hits-label{{display:flex;align-items:center;gap:5px;margin-bottom:6px;cursor:pointer;font-weight:normal;}}
.f-sep{{border:none;border-top:1px solid #e0e0e0;margin:7px 0 5px;}}
/* ── PhyloPic column ── */
#phylo-col{{position:relative;flex:none;overflow:hidden;border-right:1px solid #ddd;}}
.phylo-entry{{position:absolute;left:2px;right:2px;display:flex;align-items:center;gap:4px;overflow:hidden;}}
.phylo-entry img{{flex-shrink:0;object-fit:contain;opacity:0.85;}}
.phylo-label{{flex:1;overflow:hidden;font-size:9px;line-height:1.3;color:#333;word-break:break-word;}}
.phylo-label b{{display:block;font-size:9px;font-weight:600;}}
.phylo-label span{{display:block;font-size:8px;color:#888;}}
/* ── Export / print buttons ── */
#export-btn{{
  position:fixed;top:14px;right:130px;
  background:white;border:1px solid #ccc;border-radius:4px;
  padding:5px 10px;font-size:11px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:51;
}}
#export-btn:hover{{background:#e8e8e8;}}
#print-btn{{
  position:fixed;top:14px;right:220px;
  background:white;border:1px solid #ccc;border-radius:4px;
  padding:5px 10px;font-size:11px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:51;
}}
#print-btn:hover{{background:#e8e8e8;}}
#save-html-btn{{
  position:fixed;top:14px;right:310px;
  background:white;border:1px solid #ccc;border-radius:4px;
  padding:5px 10px;font-size:11px;cursor:pointer;
  box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:51;
}}
#save-html-btn:hover{{background:#e8e8e8;}}
@media print{{
  #filter-panel,#fullscreen-btn,#export-btn,#print-btn,#save-html-btn,#zoom-hint,#tooltip,#dot-tip{{display:none!important;}}
  body{{overflow:visible!important;}}
  #outer{{overflow:visible!important;height:auto!important;width:auto!important;}}
  #canvas{{transform:none!important;position:static!important;}}
}}
</style>
</head>
<body>
<div id="outer">
  <div id="canvas">
    <div id="header-row">
      <div id="tree-spacer" style="width:{tree_and_label_w}px;height:{header_h}px;"></div>
      <div id="phylo-spacer" style="width:{phylo_w}px;flex:none;background:#f5f5f5;border-bottom:1px solid #ccc;border-right:1px solid #ddd;"></div>
      {header_svg}
      <div id="hm2-hdr-spacer" style="width:{spacer_w}px;flex:none;background:#f5f5f5;border-bottom:1px solid #ccc;"></div>
      <div id="hm2-header-div" style="flex:none;">{header_svg2}</div>
    </div>
    <div id="col-markers"></div>
    <div id="content-row">
      <div id="tree-col">{tree_svg}</div>
      <div id="phylo-col" style="width:{phylo_w}px;height:{img_h}px;"></div>
      <div id="hm-wrap" style="position:relative;display:inline-block;flex:none;">
        <img id="hm" src="data:image/png;base64,{png_b64}"
             width="{img_w}" height="{img_h}" style="display:block;">
        <canvas id="hm-canvas" width="{img_w}" height="{img_h}"
                style="display:none;position:absolute;top:0;left:0;"></canvas>
      </div>
      <div id="hm2-content-spacer" style="width:{spacer_w}px;flex:none;background:white;"></div>
      <div id="hm2-wrap" style="position:relative;display:inline-block;flex:none;">
        <img id="hm2" src="data:image/png;base64,{png2_b64}"
             width="{img_w}" height="{img_h}" style="display:block;">
        <canvas id="hm2-canvas" width="{img_w}" height="{img_h}"
                style="display:none;position:absolute;top:0;left:0;"></canvas>
      </div>
    </div>
    <div id="row-marker"></div>
    <div id="row-marker2"></div>
    <div id="mol-markers"></div>
  </div>
</div>
<div id="tooltip"></div>
{legend}
<div id="filter-panel">
  <b>Filter rows</b>
  <label id="f-hits-label">
    <input type="checkbox" id="f-hits-only"> Has molecule hits
  </label>
  <select id="f-level">
    <option value="">— select level —</option>
  </select>
  <div id="f-val-wrap">
    <select id="f-values" multiple size="6"></select>
    <div class="f-btn-row">
      <button class="f-btn" id="f-btn-show">Show only</button>
      <button class="f-btn" id="f-btn-excl">Exclude</button>
      <button class="f-btn" id="f-btn-clear">Clear</button>
    </div>
  </div>
  <hr class="f-sep">
  <b>Match level</b>
  <select id="f-match-levels" multiple size="5"></select>
  <div class="f-btn-row">
    <button class="f-btn" id="f-btn-ml-show">Show only</button>
    <button class="f-btn" id="f-btn-ml-excl">Exclude</button>
    <button class="f-btn" id="f-btn-ml-clear">Clear</button>
  </div>
  <hr class="f-sep">
  <b>Molecules</b>
  <select id="f-mols" multiple size="5"></select>
  <div class="f-btn-row">
    <button class="f-btn" id="f-btn-mol-show">Show only</button>
    <button class="f-btn" id="f-btn-mol-excl">Exclude</button>
    <button class="f-btn" id="f-btn-mol-clear">Clear</button>
  </div>
  <hr class="f-sep">
  <b>Colour strips</b>
  <div style="display:grid;grid-template-columns:auto 1fr;align-items:center;gap:4px 6px;">
    <label style="font-weight:normal;white-space:nowrap;">Strip 1</label>
    <select id="f-strip1" style="font-size:10px;"></select>
    <label style="font-weight:normal;white-space:nowrap;">Strip 2</label>
    <select id="f-strip2" style="font-size:10px;"></select>
  </div>
  <hr class="f-sep">
  <label style="display:flex;align-items:center;gap:6px;font-weight:normal;">
    <span style="white-space:nowrap;">Label size</span>
    <input type="range" id="label-size" min="0.25" max="4" step="0.25" value="1" style="flex:1;min-width:50px;">
  </label>
  <div id="hm2-section" style="display:none;">
    <hr class="f-sep">
    <label style="display:flex;align-items:center;gap:6px;font-weight:normal;">
      <input type="checkbox" id="hm2-toggle">
      <span>Show clustered heatmap</span>
    </label>
  </div>
</div>
<div id="dot-tip"></div>
<div id="zoom-hint">Scroll to zoom &nbsp;·&nbsp; Drag to pan</div>
<button id="fullscreen-btn" title="Toggle fullscreen">&#x26F6; Fullscreen</button>
<button id="print-btn" title="Print / Save as PDF">&#x1F5A8; Print</button>
<button id="export-btn" title="Export current view as SVG">&#x2B07; SVG</button>
<button id="save-html-btn" title="Save interactive HTML with current view">&#x1F4BE; HTML</button>
<script>
(function(){{
  const fsBtn = document.getElementById('fullscreen-btn');
  fsBtn.addEventListener('click', function() {{
    if (!document.fullscreenElement) {{
      document.documentElement.requestFullscreen().catch(function() {{}});
    }} else {{
      document.exitFullscreen();
    }}
  }});
  document.addEventListener('fullscreenchange', function() {{
    fsBtn.textContent = document.fullscreenElement ? '✕ Exit Fullscreen' : '⛶ Fullscreen';
  }});
}})();
</script>
<script>
(function(){{
  const outer  = document.getElementById('outer');
  const canvas = document.getElementById('canvas');
  const hint   = document.getElementById('zoom-hint');

  let scale = 1, panX = 0, panY = 0;
  let dragging = false, startMX, startMY, startPX, startPY;

  function applyTransform() {{
    canvas.style.transform =
      'translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')';
  }}

  const CONTENT_H   = {content_h};
  const LABEL_COL_W = {label_col_w};   // base width of the leaf-label column
  let   labelColW   = LABEL_COL_W;     // current width (updated by slider)
  let   CONTENT_W   = {content_w};     // recomputed when label col resizes

  function fitToWindow() {{
    const vw = outer.clientWidth  || window.innerWidth;
    scale = (vw * 0.98) / CONTENT_W;
    panX  = (vw - CONTENT_W * scale) / 2;
    panY  = 0;
    applyTransform();
  }}

  if (document.readyState === 'complete') {{
    fitToWindow();
    renderPhyloPic(null);
    renderDefaultDots();
  }} else {{
    window.addEventListener('load', function() {{ fitToWindow(); renderPhyloPic(null); renderDefaultDots(); }});
  }}

  outer.addEventListener('wheel', function(e) {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const rect   = outer.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    panX = mx - (mx - panX) * factor;
    panY = my - (my - panY) * factor;
    scale *= factor;
    applyTransform();
  }}, {{ passive: false }});

  outer.addEventListener('mousedown', function(e) {{
    if (e.target.closest('#leg')) return;
    dragging = true;
    startMX  = e.clientX;  startMY = e.clientY;
    startPX  = panX;        startPY = panY;
    outer.classList.add('dragging');
  }});

  window.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    panX = startPX + (e.clientX - startMX);
    panY = startPY + (e.clientY - startMY);
    applyTransform();
  }});

  window.addEventListener('mouseup', function() {{
    dragging = false;
    outer.classList.remove('dragging');
  }});

  setTimeout(function() {{
    hint.style.opacity = '0';
    setTimeout(function() {{ hint.remove(); }}, 1100);
  }}, 3000);

  // ── Data ─────────────────────────────────────────────────────────────────
  const LEAVES    = {leaves_json};
  const PAIRS     = {pairs_json};
  const LEAF_HITS = {leaf_hits_json};   // [[pair_idx, masst|null, wd|null], ...]  per leaf
  const MOL_IMGS  = {mol_imgs_json};    // base64 PNG per pair index (popup-sized)
  const STRIP1_W  = {strip1_w};
  const STRIP2_W  = {strip2_w};
  const STRIP1_LABEL = {strip1_label_json};
  const STRIP2_LABEL = {strip2_label_json};
  const CELL_W    = {cell_w};
  const N_LEAVES  = {n_leaves};
  const TREE_W    = {tree_w};
  const HDR_H     = {header_h};
  const IMG_H     = {img_h};
  const IMG_W     = {img_w};
  const ROW_H     = IMG_H / N_LEAVES;

  // ── Filter / re-render data ─────────────────────────────────────────────
  const TREE_DATA      = {tree_data_json};
  const NAME_ARR       = {name_arr_json};
  const STRIP1_COLORS  = {strip1_colors_json};
  const STRIP1_DEF     = {strip1_def_json};
  const STRIP2_COLORS  = {strip2_colors_json};
  const STRIP2_DEF     = {strip2_def_json};
  const STRIP2_RARE    = new Set({strip2_rare_json});
  const ALL_STRIP_MAPS = {all_strip_maps_json};
  const STRIP1_INIT_SH = {strip1_init_sh_json};
  const STRIP2_INIT_SH = {strip2_init_sh_json};
  const MASST_COLORS   = {masst_colors_json};
  const WD_COLORS      = {wd_colors_json};

  // Active strip shorthand keys — updated interactively via the colour-strip selectors
  let currentStrip1Sh = STRIP1_INIT_SH;
  let currentStrip2Sh = STRIP2_INIT_SH;

  const SPACER_W    = {spacer_w};
  const CLUST_ORDER = {clust_order_json};   // clustered row i → original leaf index
  const CLUST_RANK  = new Array(N_LEAVES);  // original leaf index → clustered row i
  CLUST_ORDER.forEach(function(orig, ci) {{ CLUST_RANK[orig] = ci; }});
  const PHYLO_W        = {phylo_w};
  const COL_GAP        = {col_gap};
  const STRIP_GAP      = {strip_gap};
  const PHYLOPIC       = {phylopic_json};
  let   userPhyloPic   = [];
  window._mtSetUserPhyloPic = function(groups) {{
    userPhyloPic = groups || [];
    renderPhyloPic(activeLeafIndices);
  }};
  let   HM2_X0         = TREE_W + labelColW + PHYLO_W + IMG_W + SPACER_W;
  const LEAF_HIT_COLORS = {leaf_hit_colors_json};  // ott → fill hex for hit leaves

  // Normalized x position of each leaf tip in the tree (0..1 → multiply by TREE_W for SVG px)
  const LEAF_X = new Float64Array(N_LEAVES);
  (function buildLeafX(node) {{
    if ('i' in node) {{ LEAF_X[node.i] = node.x; return; }}
    for (const ch of node.ch) buildLeafX(ch);
  }})(TREE_DATA);

  // Mutable view state (updated by filter, reset by clear)
  let activeLeafIndices    = null;       // null = full view
  let activeRowH           = ROW_H;
  let activeExcludedMols   = new Set();  // set of pair indices to hide
  let activeKept2          = null;       // null = full clust view; array = filtered kept2
  let activeHm2RowH        = ROW_H;
  let activeLeafViewPos    = null;       // Map: leaf index → tree-row position in current filtered view
  let activeHighlightLeafSet = new Set(); // leaf indices highlighted by mol click
  let labelSizeScale       = 1.0;        // multiplier from the label-size slider
  let _activeMolHighlight  = false;      // true while mol-specific dots are shown

  // Allow the save-HTML restore script to push saved state back into this closure.
  let activeHiddenMatchLevels = new Set();   // match levels whose MASST coloring is suppressed

  window._mtSetActive = function(li, rh, em, k2, rh2, lvp) {{
    activeLeafIndices  = li;
    activeRowH         = rh;
    activeExcludedMols = new Set(em || []);
    activeKept2        = k2;
    activeHm2RowH      = rh2;
    activeLeafViewPos  = lvp ? new Map(lvp) : null;
  }};

  const tooltip    = document.getElementById('tooltip');
  const dotTip     = document.getElementById('dot-tip');
  const rowMarker  = document.getElementById('row-marker');
  const rowMarker2 = document.getElementById('row-marker2');
  const molMarkers = document.getElementById('mol-markers');
  const colMarkers = document.getElementById('col-markers');
  function updateRowMarkerBounds() {{
    rowMarker.style.left   = '0px';
    rowMarker.style.width  = HM2_X0 + 'px';
    rowMarker2.style.left  = HM2_X0 + 'px';
    rowMarker2.style.width = IMG_W  + 'px';
  }}
  updateRowMarkerBounds();

  function clearMolHitDots() {{
    _activeMolHighlight = false;
    dotTip.style.display = 'none';
    const svg = document.getElementById('tree-svg');
    if (svg) svg.querySelectorAll('.mol-hit-dot,.mol-hit-bg').forEach(function(el) {{ el.remove(); }});
    if (activeHighlightLeafSet.size > 0) {{
      activeHighlightLeafSet = new Set();
      const _hmCvs = document.getElementById('hm-canvas');
      if (_hmCvs && _hmCvs.style.display !== 'none') {{
        drawHeatmapCanvas(_hmCvs, activeLeafIndices || Array.from({{length:N_LEAVES}},function(_,i){{return i;}}));
      }}
      const _hm2Cvs = document.getElementById('hm2-canvas');
      if (_hm2Cvs && _hm2Cvs.style.display !== 'none') {{
        drawHeatmapCanvas(_hm2Cvs, activeKept2 || CLUST_ORDER);
      }}
    }}
  }}

  // Default dots: one dot per leaf that has any match across all non-excluded molecules.
  // Priority: orange (MASST+WD) > red shades (MASST only) > blue (WD only).
  function renderDefaultDots() {{
    const treeSvg = document.getElementById('tree-svg');
    if (!treeSvg) return;
    treeSvg.querySelectorAll('.mol-hit-dot').forEach(function(el) {{ el.remove(); }});
    const r = 4;
    const _ML_ORDER = ['subspecies','species','genus','family','order','class','phylum','kingdom'];
    const _ML_HEX   = {{'subspecies':'#ff0000','species':'#cc0000','genus':'#e05050',
                        'family':'#eb8080','order':'#f0a0a0','class':'#f5c0c0',
                        'phylum':'#fae0e0','kingdom':'#fdf0f0'}};
    for (let i = 0; i < N_LEAVES; i++) {{
      const y = getLeafTreeY(i);
      if (y === null) continue;
      const hits = LEAF_HITS[i] || [];
      let hasMasst = false, hasWd = false, bestMasstLv = null;
      for (let h = 0; h < hits.length; h++) {{
        if (activeExcludedMols.has(hits[h][0])) continue;
        if (hits[h][1] && !activeHiddenMatchLevels.has(hits[h][1])) {{
          hasMasst = true;
          const lv = hits[h][1];
          if (!bestMasstLv || _ML_ORDER.indexOf(lv) < _ML_ORDER.indexOf(bestMasstLv))
            bestMasstLv = lv;
        }}
        if (hits[h][2]) hasWd = true;
      }}
      if (!hasMasst && !hasWd) continue;
      const fill = (hasMasst && hasWd) ? '#e07000'
                 : hasMasst            ? (_ML_HEX[bestMasstLv] || '#cc0000')
                 :                       '#0000cc';
      const cx = LEAF_X[i] * TREE_W;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx',        cx.toFixed(1));
      circle.setAttribute('cy',        y.toFixed(1));
      circle.setAttribute('r',         r);
      circle.setAttribute('fill',      fill);
      circle.setAttribute('class',     'mol-hit-dot');
      circle.setAttribute('data-leaf', i);
      treeSvg.insertBefore(circle, treeSvg.firstChild);
    }}
  }}

  function clearSelection() {{
    tooltip.style.display    = 'none';
    rowMarker.style.display  = 'none';
    rowMarker2.style.display = 'none';
    molMarkers.innerHTML     = '';
    colMarkers.innerHTML     = '';
    clearMolHitDots();
    renderDefaultDots();
  }}

  function getLeafTreeY(i) {{
    let k;
    if (activeLeafViewPos) {{
      k = activeLeafViewPos.get(i);
      if (k === undefined) return null;
    }} else {{
      k = i;
    }}
    return (k + 0.5) * activeRowH;
  }}

  function renderPhyloPic(keptIndices) {{
    const col = document.getElementById('phylo-col');
    if (!col) return;
    col.innerHTML = '';
    const _allGroups = PHYLOPIC.concat(userPhyloPic);
    if (!_allGroups.length) return;

    const keptSet = keptIndices ? new Set(keptIndices) : null;
    const PAD  = 4;                          // min gap between silhouettes (px)
    const SZ   = Math.min(PHYLO_W - 4, 44); // fixed equal size for every icon
    const HALF = SZ / 2;
    const colH = parseFloat(col.style.height) || IMG_H;
    const BKT  = PHYLO_W - 3;               // x of bracket line for broadest level
    // Horizontal indent per level: more specific = further inside (smaller x).
    const _LVINDENT = {{
      'kingdom':0,'phylum':1,'class':2,'order':3,'family':4,'genus':5,'species':6,'subspecies':7,
      'ncbikingdom':0,'ncbiphylum':1,'ncbiclass':2,'ncbiorder':3,
      'ncbifamily':4,'ncbigenus':5,'ncbispecies':6
    }};
    const INDENT_STEP = 4;

    // ── Step 1: compute ideal centre for each visible group ──────────────
    const items = [];
    _allGroups.forEach(function(group) {{
      const vis = group.leaves.filter(function(i) {{
        return !keptSet || keptSet.has(i);
      }});
      if (!vis.length) return;
      const ys = vis.map(function(i) {{ return getLeafTreeY(i); }})
                    .filter(function(y) {{ return y !== null; }});
      if (!ys.length) return;
      const yMin = Math.min.apply(null, ys);
      const yMax = Math.max.apply(null, ys);
      // Nearest actual group-member Y to the range centre — used as the connector
      // target so the line always ends on a real group leaf, not an interloper.
      const _midY = (yMin + yMax) / 2;
      const nearestMidY = ys.reduce(function(best, y) {{
        return Math.abs(y - _midY) < Math.abs(best - _midY) ? y : best;
      }}, ys[0]);
      items.push({{ group: group, yMin: yMin, yMax: yMax, targetY: _midY, visCount: vis.length, nearestMidY: nearestMidY }});
    }});
    if (!items.length) return;
    items.sort(function(a, b) {{ return a.targetY - b.targetY; }});

    // ── Step 2: packed layout — forward pass (push down), then pull up ───
    const tops = new Array(items.length);
    let cursor = 0;
    for (let i = 0; i < items.length; i++) {{
      tops[i] = Math.max(items[i].targetY - HALF, cursor);
      cursor  = tops[i] + SZ + PAD;
    }}
    // If entries overflow the column bottom, pull up from the last item
    if (cursor - PAD > colH) {{
      cursor = colH;
      for (let i = items.length - 1; i >= 0; i--) {{
        tops[i] = Math.min(tops[i], cursor - SZ);
        cursor  = tops[i] - PAD;
      }}
    }}
    // Mark items whose entry-div centre falls within the column — anything
    // outside is clipped by overflow:hidden while the SVG (overflow:visible)
    // still draws the bracket, producing a visible bracket with no label.
    // Additionally suppress image-less groups whose packed position is more than
    // 3×SZ away from their bracket midpoint: without an image the text label is
    // too small for the user to associate with the distant bracket.
    const visible = items.map(function(item, idx) {{
      if (item.group.user) return true;  // user-added: always show
      const c = tops[idx] + HALF;
      if (c < 0 || c > colH) return false;
      if (!item.group.img && item.visCount > 3) {{
        const bktMid = (item.yMin + item.yMax) / 2;
        if (Math.abs(c - bktMid) > 3 * SZ) return false;
      }}
      return true;
    }});
    // Second pass: suppress any item whose Y range is fully contained within
    // a broader visible bracket — covers paraphylogenetic cases where an organism
    // has a NaN intermediate rank yet lands visually inside a labelled bracket.
    // User-added items are never suppressed by this pass.
    for (let i = 0; i < items.length; i++) {{
      if (!visible[i] || items[i].group.user) continue;
      const yMinI = items[i].yMin, yMaxI = items[i].yMax;
      for (let j = 0; j < items.length; j++) {{
        if (i === j || !visible[j] || items[j].visCount <= 3) continue;
        const yMinJ = items[j].yMin, yMaxJ = items[j].yMax;
        if (yMaxJ - yMinJ > yMaxI - yMinI && yMinI >= yMinJ && yMaxI <= yMaxJ) {{
          visible[i] = false; break;
        }}
      }}
    }}

    // ── Step 3: SVG helper ────────────────────────────────────────────────
    function svgEl(parent, tag, attrs) {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      parent.appendChild(el);
    }}

    // ── Step 4: render entry divs ─────────────────────────────────────────
    items.forEach(function(item, idx) {{
      if (!visible[idx]) return;
      const {{ group }} = item;
      const top = tops[idx];

      const entry = document.createElement('div');
      entry.className    = 'phylo-entry';
      entry.style.top    = top.toFixed(1) + 'px';
      entry.style.height = SZ + 'px';
      let hasContent = false;

      if (group.img || group.imgSrc) {{
        const el = document.createElement('img');
        el.src          = group.imgSrc || ('data:image/png;base64,' + group.img);
        el.alt          = group.label;
        el.title        = group.label + (group.level ? ' (' + group.level + ')' : '');
        el.style.width  = SZ + 'px';
        el.style.height = SZ + 'px';
        entry.appendChild(el);
        hasContent = true;
      }}
      const lbl = document.createElement('div');
      lbl.className = 'phylo-label';
      const safeName  = group.label.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const safeLevel = (group.level || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      lbl.innerHTML = '<b>' + safeName + '</b>'
                    + (safeLevel ? '<span>' + safeLevel + '</span>' : '');
      entry.appendChild(lbl);
      col.appendChild(entry);
    }});

    // ── Step 5: SVG overlay — bracket + connector for every group ─────────
    // Appended after entries so it renders on top (DOM order = paint order).
    const ov = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    ov.setAttribute('width',  PHYLO_W);
    ov.setAttribute('height', colH);
    ov.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;overflow:visible;';
    col.appendChild(ov);

    items.forEach(function(item, idx) {{
      if (!visible[idx]) return;
      const {{ group, yMin, yMax, visCount, nearestMidY }} = item;
      const top      = tops[idx];
      const center   = top + HALF;
      const midTgt   = nearestMidY;
      const imgEdge  = (group.img || group.imgSrc) ? SZ + 3 : 3;

      // Per-level bracket indent: more specific levels sit further inside (smaller x).
      const _lvIdx = _LVINDENT[(group.level || '').toLowerCase()];
      const BKT_i  = BKT - (_lvIdx !== undefined ? _lvIdx : 0) * INDENT_STEP;

      // Groups with ≤3 visible leaves get a connector-only rendering — no
      // vertical bracket line and no end ticks (too cramped to show a range).
      // Larger groups get the full bracket; species-level uses a dashed line.
      const isSmallGroup = visCount <= 3;
      if (!isSmallGroup) {{
        const isLeafLevel = (group.level === 'Species');
        const bktAttrs = {{ x1: BKT_i, y1: yMin.toFixed(1), x2: BKT_i, y2: yMax.toFixed(1),
                            stroke: '#999', 'stroke-width': '1.5' }};
        if (isLeafLevel) bktAttrs['stroke-dasharray'] = '3,2';
        svgEl(ov, 'line', bktAttrs);
        if (!isLeafLevel) {{
          [yMin, yMax].forEach(function(ty) {{
            svgEl(ov, 'line', {{ x1: BKT_i - 4, y1: ty.toFixed(1), x2: BKT_i + 1, y2: ty.toFixed(1),
                                 stroke: '#999', 'stroke-width': '1.5' }});
          }});
        }}
      }}
      // Connector line always drawn.
      // Solid when the group has a bracket (range of leaves); dashed for small
      // groups (≤3 leaves, no bracket) where we point to individual positions.
      const cAttrs = {{ x1: imgEdge, y1: center.toFixed(1),
                        x2: BKT_i - 2, y2: midTgt.toFixed(1),
                        stroke: '#999', 'stroke-width': '1' }};
      if (isSmallGroup) cAttrs['stroke-dasharray'] = '3,2';
      svgEl(ov, 'line', cAttrs);
    }});
  }}

  function highlightMolHits(pi) {{
    clearMolHitDots();
    _activeMolHighlight = true;
    const treeSvg = document.getElementById('tree-svg');
    if (!treeSvg) return;
    const r = 5;
    const _ML_ORDER = ['subspecies','species','genus','family','order','class','phylum','kingdom'];
    const _ML_HEX   = {{'subspecies':'#ff0000','species':'#cc0000','genus':'#e05050',
                        'family':'#eb8080','order':'#f0a0a0','class':'#f5c0c0',
                        'phylum':'#fae0e0','kingdom':'#fdf0f0'}};
    const svgW = TREE_W + labelColW;

    // Build highlight set first so canvas redraw can use it
    activeHighlightLeafSet = new Set();
    for (let i = 0; i < N_LEAVES; i++) {{
      const hits = LEAF_HITS[i] || [];
      for (let h = 0; h < hits.length; h++) {{
        if (hits[h][0] === pi && (hits[h][1] || hits[h][2])) {{
          activeHighlightLeafSet.add(i); break;
        }}
      }}
    }}

    // Force hm1 canvas activation with highlight tint
    const _hmCvs = document.getElementById('hm-canvas');
    const _hmImg = document.getElementById('hm');
    drawHeatmapCanvas(_hmCvs, activeLeafIndices || Array.from({{length:N_LEAVES}},function(_,i){{return i;}}));
    _hmCvs.style.display    = 'block';
    _hmImg.style.visibility = 'hidden';

    // Force hm2 canvas if it is visible
    const _hm2Wrap = document.getElementById('hm2-wrap');
    if (_hm2Wrap && _hm2Wrap.style.display !== 'none') {{
      const _hm2Cvs = document.getElementById('hm2-canvas');
      const _hm2Img = document.getElementById('hm2');
      drawHeatmapCanvas(_hm2Cvs, activeKept2 || CLUST_ORDER);
      _hm2Cvs.style.display    = 'block';
      _hm2Img.style.visibility = 'hidden';
    }}

    // Add SVG elements — inserted before firstChild so they sit behind tree lines/labels.
    // Dots inserted first → become firstChild; then bg rects inserted → become firstChild,
    // pushing dots behind tree content but in front of rects. Final order:
    //   [bg rects] [dots] [path] [text...]
    for (let i = 0; i < N_LEAVES; i++) {{
      const hits = LEAF_HITS[i] || [];
      let hasMasst = false, hasWd = false, bestMasstLv = null;
      for (let h = 0; h < hits.length; h++) {{
        if (hits[h][0] !== pi) continue;
        if (hits[h][1] && !activeHiddenMatchLevels.has(hits[h][1])) {{
          hasMasst = true;
          const lv = hits[h][1];
          if (!bestMasstLv || _ML_ORDER.indexOf(lv) < _ML_ORDER.indexOf(bestMasstLv))
            bestMasstLv = lv;
        }}
        if (hits[h][2]) hasWd = true;
      }}
      if (!hasMasst && !hasWd) continue;
      const y = getLeafTreeY(i);
      if (y === null) continue;

      // Dot (inserted before firstChild → ends up after bg rects, before tree path)
      const fill = (hasMasst && hasWd) ? '#e07000'
                 : hasMasst            ? (_ML_HEX[bestMasstLv] || '#cc0000')
                 :                       '#0000cc';
      const cx = LEAF_X[i] * TREE_W;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx',       cx.toFixed(1));
      circle.setAttribute('cy',       y.toFixed(1));
      circle.setAttribute('r',        r);
      circle.setAttribute('fill',     fill);
      circle.setAttribute('class',    'mol-hit-dot');
      circle.setAttribute('data-leaf', i);
      treeSvg.insertBefore(circle, treeSvg.firstChild);
    }}

    // Background row highlight rects — inserted last → end up first (before dots)
    for (let i = 0; i < N_LEAVES; i++) {{
      if (!activeHighlightLeafSet.has(i)) continue;
      const y = getLeafTreeY(i);
      if (y === null) continue;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x',      '0');
      rect.setAttribute('y',      (y - activeRowH / 2).toFixed(1));
      rect.setAttribute('width',  svgW);
      rect.setAttribute('height', Math.max(1, activeRowH).toFixed(1));
      rect.setAttribute('fill',   'rgba(255,240,180,0.7)');
      rect.setAttribute('class',  'mol-hit-bg');
      treeSvg.insertBefore(rect, treeSvg.firstChild);
    }}
  }}

  function placeTooltip(x, y, html) {{
    tooltip.innerHTML    = html;
    tooltip.style.display = 'block';
    const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    const vw = window.innerWidth,   vh = window.innerHeight;
    tooltip.style.left = (x + 16 + tw > vw ? x - tw - 8 : x + 16) + 'px';
    tooltip.style.top  = (y + 16 + th > vh ? y - th - 8 : y + 16) + 'px';
  }}

  function markRow(row) {{
    if (!_activeMolHighlight) clearMolHitDots();
    molMarkers.innerHTML = '';
    let viewRow = row;
    if (activeLeafIndices) {{
      viewRow = activeLeafIndices.indexOf(row);
      if (viewRow < 0) {{
        rowMarker.style.display = 'none'; rowMarker2.style.display = 'none';
        if (!_activeMolHighlight) renderDefaultDots();
        return;
      }}
    }}
    // hm1 marker
    rowMarker.style.display = 'block';
    rowMarker.style.top     = (HDR_H + viewRow * activeRowH) + 'px';
    rowMarker.style.height  = Math.max(3, activeRowH) + 'px';
    // hm2 marker — only when hm2 is visible
    const hm2WrapEl = document.getElementById('hm2-wrap');
    const hm2Shown  = hm2WrapEl && hm2WrapEl.style.display !== 'none';
    if (hm2Shown) {{
      if (activeKept2) {{
        const pos2 = activeKept2.indexOf(row);
        if (pos2 >= 0) {{
          rowMarker2.style.display = 'block';
          rowMarker2.style.top     = (HDR_H + pos2 * activeHm2RowH) + 'px';
          rowMarker2.style.height  = Math.max(3, activeHm2RowH) + 'px';
        }} else {{
          rowMarker2.style.display = 'none';
        }}
      }} else {{
        const clustRow = CLUST_RANK[row];
        if (clustRow !== undefined) {{
          rowMarker2.style.display = 'block';
          rowMarker2.style.top     = (HDR_H + clustRow * ROW_H) + 'px';
          rowMarker2.style.height  = Math.max(3, ROW_H) + 'px';
        }}
      }}
    }} else {{
      rowMarker2.style.display = 'none';
    }}
    if (!_activeMolHighlight) renderDefaultDots();
  }}

  function leafLineHtml(leaf, hits) {{
    let html = '';
    for (let i = 0; i < NCBI_LEVELS.length; i++) {{
      const sh = NCBI_LEVELS[i][0], lbl = NCBI_LEVELS[i][1];
      const val = leaf[sh];
      if (val) html += '<b>' + lbl + ':</b> ' + val + '<br>';
    }}
    // MASST / Wikidata status
    const hitArr = hits || [];
    const masstLevels = [];
    const wdLevels    = [];
    for (let i = 0; i < hitArr.length; i++) {{
      if (hitArr[i][1] && masstLevels.indexOf(hitArr[i][1]) < 0) masstLevels.push(hitArr[i][1]);
      if (hitArr[i][2] && wdLevels.indexOf(hitArr[i][2])    < 0) wdLevels.push(hitArr[i][2]);
    }}
    html += '<b>MASST:</b> '    + (masstLevels.length ? masstLevels.join(', ') : '—') + '<br>';
    html += '<b>Wikidata:</b> ' + (wdLevels.length    ? wdLevels.join(', ')    : '—');
    return html;
  }}

  function molCardHtml(pi, mv, wv, tt) {{
    const p = PAIRS[pi];
    const lvl = (mv ? 'MASST: ' + mv : '') + (mv && wv ? ' | ' : '') + (wv ? 'WD: ' + wv : '');
    const imgTag = MOL_IMGS[pi]
      ? '<img src="data:image/png;base64,' + MOL_IMGS[pi] + '">'
      : '<div class="tt-mol-ph"></div>';
    if (tt) {{
      return '<div class="tt-mol-wide">'
        + imgTag
        + '<div class="tt-mol-info">'
          + '<div class="tt-mol-name">' + p.l + '</div>'
          + '<div class="tt-mol-lvl">'  + lvl + '</div>'
          + '<div class="tt-mol-src"><b>Sources:</b> ' + tt + '</div>'
        + '</div>'
        + '</div>';
    }}
    return '<div class="tt-mol">'
      + imgTag
      + '<div class="tt-mol-name">' + p.l + '</div>'
      + '<div class="tt-mol-lvl">'  + lvl + '</div>'
      + '</div>';
  }}

  // Header click → show all leaves (in current view) that have this molecule
  function showMolPopup(pi, x, y) {{
    rowMarker.style.display = 'none';
    molMarkers.innerHTML    = '';
    colMarkers.innerHTML    = '';
    // Vertical column markers — one for hm1, one for hm2.
    // Appended to #col-markers which sits before #content-row in the DOM,
    // so they paint behind the heatmap and tree.
    function addColMarker(xLeft) {{
      const c = document.createElement('div');
      c.className = 'col-m';
      c.style.left   = xLeft + 'px';
      c.style.width  = (CELL_W * 2 + COL_GAP) + 'px';
      c.style.top    = HDR_H + 'px';
      c.style.height = IMG_H + 'px';
      colMarkers.appendChild(c);
    }}
    addColMarker(TREE_W + labelColW + PHYLO_W + STRIP1_W + STRIP_GAP + STRIP2_W + STRIP_GAP + pi * (CELL_W * 2 + COL_GAP));
    const _hm2WrapEl = document.getElementById('hm2-wrap');
    if (_hm2WrapEl && _hm2WrapEl.style.display !== 'none') {{
      addColMarker(HM2_X0 + STRIP1_W + STRIP_GAP + STRIP2_W + STRIP_GAP + pi * (CELL_W * 2 + COL_GAP));
    }}
    const p   = PAIRS[pi];
    const img = MOL_IMGS[pi]
      ? '<img src="data:image/png;base64,' + MOL_IMGS[pi] + '" style="width:62px;height:62px;object-fit:contain;">'
      : '';
    const viewIndices = activeLeafIndices || Array.from({{length: N_LEAVES}}, function(_, i) {{ return i; }});

    // --- Collect matching leaves ---
    const entries = [];
    for (let k = 0; k < viewIndices.length; k++) {{
      const i    = viewIndices[k];
      const hits = LEAF_HITS[i] || [];
      for (let h = 0; h < hits.length; h++) {{
        if (hits[h][0] !== pi) continue;
        const lf = LEAVES[i] || {{}};
        entries.push({{
          king: lf.k || '', cls: lf.c || '', ord: lf.o || '',
          name: lf.s || lf.g || lf.f || '—',
          mv: hits[h][1] || '', wv: hits[h][2] || '',
          viewRow: k, leafIdx: i,
        }});
        break;
      }}
    }}

    // Row highlighting is handled by highlightMolHits() via SVG rects (tree)
    // and canvas tinting (heatmap), called below.

    // --- Sort by kingdom → sub-group → name, then build HTML with headers ---
    // For Plantae use order as sub-header; for all others use class.
    function subGroupKey(e) {{
      return (e.king || '').toLowerCase() === 'plantae' ? (e.ord || '') : (e.cls || '');
    }}
    function subGroupLabel(e) {{
      return (e.king || '').toLowerCase() === 'plantae' ? (e.ord || 'Unknown order') : (e.cls || 'Unknown class');
    }}
    entries.sort(function(a, b) {{
      return (a.king || '').localeCompare(b.king || '')
          || subGroupKey(a).localeCompare(subGroupKey(b))
          || a.name.localeCompare(b.name);
    }});
    const leafLines = [];
    let lastKing = null, lastSub = null;
    entries.forEach(function(e) {{
      if (e.king !== lastKing) {{
        if (lastKing !== null) leafLines.push('<div style="height:3px;"></div>');
        leafLines.push('<div style="font-size:10px;font-weight:700;padding:3px 0 1px;'
          + 'border-top:1px solid #ddd;color:#222;">'
          + (e.king || 'Unknown kingdom') + '</div>');
        lastKing = e.king; lastSub = null;
      }}
      const sg = subGroupKey(e);
      if (sg !== lastSub) {{
        leafLines.push('<div style="font-size:9px;font-weight:600;color:#555;'
          + 'padding:1px 0 1px 6px;">'
          + subGroupLabel(e) + '</div>');
        lastSub = sg;
      }}
      const lvl = (e.mv ? 'MASST:' + e.mv : '') + (e.mv && e.wv ? ' | ' : '') + (e.wv ? 'WD:' + e.wv : '');
      leafLines.push('<div style="font-size:10px;padding:1px 0 1px 14px;">'
        + e.name
        + (lvl ? ' <span style="color:#888;font-size:9px;">(' + lvl + ')</span>' : '')
        + '</div>');
    }});
    highlightMolHits(pi);
    placeTooltip(x, y,
      '<b>' + p.l + '</b>'
      + (img ? '<div style="margin:5px 0;">' + img + '</div>' : '')
      + '<div class="tt-sep"></div>'
      + (leafLines.length
          ? '<b>' + leafLines.length + ' matching leaves:</b>'
            + '<div style="max-height:220px;overflow-y:auto;margin-top:3px;">'
            + leafLines.join('') + '</div>'
          : '<i style="color:#999">No hits in current view</i>')
    );
  }}

  // Tree-area or leaf-label click → show ALL hit molecules with sources
  function showLeafPopup(row, x, y) {{
    const leaf = LEAVES[row] || {{}};
    const hits = LEAF_HITS[row] || [];
    let molsHtml = '';
    for (let i = 0; i < hits.length; i++) {{
      molsHtml += molCardHtml(hits[i][0], hits[i][1], hits[i][2], hits[i][3] || '');
    }}
    placeTooltip(x, y,
      leafLineHtml(leaf, hits)
      + (hits.length
          ? '<div class="tt-sep"></div><b>Hit molecules (' + hits.length + '):</b>'
            + '<div class="tt-mols">' + molsHtml + '</div>'
          : ''));
    markRow(row);
  }}

  // Heatmap tile click → leaf lineage + that one molecule with structure
  function showTilePopup(row, hmX, x, y) {{
    const leaf = LEAVES[row] || {{}};
    let colHtml = '', molHtml = '';
    if (hmX < STRIP1_W) {{
      const _s1lbl = (ALL_STRIP_MAPS[currentStrip1Sh] || {{}}).label || STRIP1_LABEL;
      colHtml = '<b>Strip:</b> ' + _s1lbl;
    }} else if (hmX < STRIP1_W + STRIP_GAP + STRIP2_W) {{
      const _s2lbl = (ALL_STRIP_MAPS[currentStrip2Sh] || {{}}).label || STRIP2_LABEL;
      colHtml = '<b>Strip:</b> ' + _s2lbl;
    }} else {{
      const rel  = hmX - STRIP1_W - STRIP_GAP - STRIP2_W - STRIP_GAP;
      const pi   = Math.floor(rel / (CELL_W * 2 + COL_GAP));
      const isWd = (rel % (CELL_W * 2 + COL_GAP)) >= CELL_W;
      if (pi >= 0 && pi < PAIRS.length) {{
        colHtml = '<b>Molecule:</b> ' + PAIRS[pi].l
                + '<br><b>Source:</b> ' + (isWd ? 'Wikidata' : 'MASST');
        const _tHits = LEAF_HITS[row] || [];
        const _tHit  = _tHits.find(function(h) {{ return h[0] === pi; }});
        const _tt    = (!isWd && _tHit && _tHit[3]) ? _tHit[3] : '';
        molHtml = '<div class="tt-sep"></div>'
                + '<div class="tt-mols">' + molCardHtml(pi, isWd ? null : '?', isWd ? '?' : null, _tt) + '</div>';
      }}
    }}
    const tileHits = LEAF_HITS[row] || [];
    placeTooltip(x, y,
      leafLineHtml(leaf, tileHits)
      + (colHtml ? '<div class="tt-sep"></div>' + colHtml + molHtml : ''));
    markRow(row);
  }}

  // distinguish drag from click
  let mouseDownX = 0, mouseDownY = 0;
  outer.addEventListener('mousedown', function(e) {{
    mouseDownX = e.clientX; mouseDownY = e.clientY;
  }}, true);

  outer.addEventListener('click', function(e) {{
    if (Math.abs(e.clientX - mouseDownX) > 4 || Math.abs(e.clientY - mouseDownY) > 4) return;
    if (e.target.closest('#tooltip') || e.target.closest('#leg')) return;
    const rect = outer.getBoundingClientRect();
    const cx   = (e.clientX - rect.left - panX) / scale;
    const cy   = (e.clientY - rect.top  - panY) / scale;
    const hmX  = cx - TREE_W - labelColW - PHYLO_W;
    const hmY  = cy - HDR_H;
    // Click in header row → molecule column popup (hm1 or hm2)
    if (hmY < 0 && hmY > -HDR_H) {{
      const hm2X_hdr = cx - HM2_X0;
      if (hm2X_hdr >= 0 && hm2X_hdr <= IMG_W) {{
        const rel2 = hm2X_hdr - STRIP1_W - STRIP_GAP - STRIP2_W - STRIP_GAP;
        if (rel2 >= 0) {{
          const pi = Math.floor(rel2 / (CELL_W * 2 + COL_GAP));
          if (pi >= 0 && pi < PAIRS.length) {{ showMolPopup(pi, e.clientX, e.clientY); return; }}
        }}
        return;
      }}
      if (hmX >= 0 && hmX <= IMG_W) {{
        const rel = hmX - STRIP1_W - STRIP_GAP - STRIP2_W - STRIP_GAP;
        if (rel >= 0) {{
          const pi = Math.floor(rel / (CELL_W * 2 + COL_GAP));
          if (pi >= 0 && pi < PAIRS.length) {{ showMolPopup(pi, e.clientX, e.clientY); return; }}
        }}
      }}
      return;
    }}
    // Click in hm2 (clustered heatmap)?
    const hm2X = cx - HM2_X0;
    if (hm2X >= 0 && hm2X <= IMG_W && hmY >= 0 && hmY <= IMG_H) {{
      let row2;
      if (activeKept2) {{
        const nKept2 = activeKept2.length;
        const ki2 = Math.min(nKept2 - 1, Math.floor(hmY / IMG_H * nKept2));
        row2 = activeKept2[ki2];
      }} else {{
        const clustRow = Math.min(N_LEAVES - 1, Math.floor(hmY / IMG_H * N_LEAVES));
        row2 = CLUST_ORDER[clustRow];
      }}
      if (hm2X < STRIP1_W + STRIP_GAP + STRIP2_W) {{
        showLeafPopup(row2, e.clientX, e.clientY);
      }} else {{
        const rel2 = hm2X - STRIP1_W - STRIP_GAP - STRIP2_W - STRIP_GAP;
        const pi2  = Math.floor(rel2 / (CELL_W * 2 + COL_GAP));
        if (pi2 >= 0 && pi2 < PAIRS.length) highlightMolHits(pi2);
        showTilePopup(row2, hm2X, e.clientX, e.clientY);
      }}
      return;
    }}
    if (hmY < 0 || hmY > IMG_H) {{ clearSelection(); return; }}
    const nView     = activeLeafIndices ? activeLeafIndices.length : N_LEAVES;
    const rowInView = Math.min(nView - 1, Math.floor(hmY / IMG_H * nView));
    const row       = activeLeafIndices ? activeLeafIndices[rowInView] : rowInView;
    if (hmX < 0) {{
      // tree panel → show all molecules for this leaf
      showLeafPopup(row, e.clientX, e.clientY);
    }} else if (hmX <= IMG_W) {{
      // heatmap → show leaf info + clicked molecule
      showTilePopup(row, hmX, e.clientX, e.clientY);
    }} else {{
      clearSelection();
    }}
  }});

  document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') clearSelection();
  }});

  // Hover tooltip for green leaf-hit dots
  outer.addEventListener('mousemove', function(e) {{
    const dot = e.target && e.target.closest && e.target.closest('.mol-hit-dot');
    if (dot) {{
      const li   = parseInt(dot.getAttribute('data-leaf'), 10);
      const leaf = LEAVES[li] || {{}};
      dotTip.innerHTML = leafLineHtml(leaf, LEAF_HITS[li] || []);
      dotTip.style.display = 'block';
      const tw = dotTip.offsetWidth, th = dotTip.offsetHeight;
      const vw = window.innerWidth,   vh = window.innerHeight;
      dotTip.style.left = (e.clientX + 14 + tw > vw ? e.clientX - tw - 8 : e.clientX + 14) + 'px';
      dotTip.style.top  = (e.clientY + 14 + th > vh ? e.clientY - th - 8 : e.clientY + 14) + 'px';
    }} else {{
      dotTip.style.display = 'none';
    }}
  }});
  outer.addEventListener('mouseleave', function() {{ dotTip.style.display = 'none'; }});

  // ── Interactive PhyloPic addition (double-click) ─────────────────────────
  async function fetchAndAddPhyloPic(lookupName, label, level, leafIndices) {{
    const base = 'https://api.phylopic.org';
    const pdLicenses = [
      'https://creativecommons.org/publicdomain/zero/1.0/',
      'https://creativecommons.org/licenses/publicdomain/'
    ];
    const msg = document.createElement('div');
    msg.style.cssText = 'position:fixed;bottom:16px;left:50%;transform:translateX(-50%);'
      + 'background:#333;color:#fff;padding:8px 18px;border-radius:20px;font-size:13px;'
      + 'z-index:9999;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.4);';
    msg.textContent = 'Looking up PhyloPic for “' + label + '”…';
    document.body.appendChild(msg);
    try {{
      const rootResp = await fetch(base, {{headers:{{Accept:'application/json'}}}});
      const root = await rootResp.json();
      const build = root.build || 538;
      const namesToTry = [lookupName];
      const parts = lookupName.trim().split(/\\s+/);
      if (parts.length > 1) namesToTry.push(parts[0]);
      for (const attemptName of namesToTry) {{
        // Exact autocomplete match only
        const acResp = await fetch(
          base + '/autocomplete?query=' + encodeURIComponent(attemptName),
          {{headers:{{Accept:'application/json'}}}}
        );
        const acData = await acResp.json();
        const exact = (acData.matches || []).find(function(m) {{
          return m.toLowerCase() === attemptName.toLowerCase();
        }});
        const lookup = exact || attemptName.toLowerCase();
        // Find node UUID
        const ndResp = await fetch(
          base + '/nodes?filter_name=' + encodeURIComponent(lookup) + '&build=' + build + '&page=0',
          {{headers:{{Accept:'application/json'}}}}
        );
        const ndData = await ndResp.json();
        const nodeHrefs = ((ndData._links || {{}}).items) || [];
        if (!nodeHrefs.length) continue;
        const nodeUuid = nodeHrefs[0].href.split('/nodes/').pop().split('?')[0];
        // Verify node carries the queried name
        const niResp = await fetch(
          base + '/nodes/' + nodeUuid + '?build=' + build,
          {{headers:{{Accept:'application/json'}}}}
        );
        const niData = await niResp.json();
        const nodeNames = new Set(
          (niData.names || []).flatMap(function(nl) {{
            return nl.map(function(n) {{ return n.text.toLowerCase(); }});
          }})
        );
        if (!nodeNames.has(attemptName.toLowerCase())) continue;
        // Find CC0/PDM image
        let imgUuid = null;
        for (const lic of pdLicenses) {{
          const imResp = await fetch(
            base + '/images?filter_node=' + nodeUuid + '&build=' + build
              + '&page=0&filter_license=' + encodeURIComponent(lic),
            {{headers:{{Accept:'application/json'}}}}
          );
          const imData = await imResp.json();
          const imgItems = ((imData._links || {{}}).items) || [];
          if (imgItems.length) {{
            imgUuid = imgItems[0].href.split('/images/').pop().split('?')[0];
            break;
          }}
        }}
        if (!imgUuid) continue;
        // Get thumbnail closest to 40 px
        const metaResp = await fetch(
          base + '/images/' + imgUuid + '?build=' + build,
          {{headers:{{Accept:'application/json'}}}}
        );
        const meta = await metaResp.json();
        const thumbs = ((meta._links || {{}}).thumbnailFiles) || [];
        if (!thumbs.length) continue;
        const SZ  = 40;
        const best = thumbs.reduce(function(b, t) {{
          return Math.abs(parseInt(t.sizes) - SZ) < Math.abs(parseInt(b.sizes) - SZ) ? t : b;
        }});
        // Fetch image: try base64 (offline-capable) and fall back to direct URL
        // if the image CDN blocks cross-origin fetch (no CORS headers).
        let imgSrc = null;
        try {{
          const imgResp = await fetch(best.href);
          const blob    = await imgResp.blob();
          imgSrc = await new Promise(function(resolve) {{
            const reader = new FileReader();
            reader.onload = function() {{ resolve(reader.result); }};  // full data: URI
            reader.readAsDataURL(blob);
          }});
        }} catch(_) {{
          imgSrc = best.href;  // CORS blocked — use URL (requires internet when displayed)
        }}
        // Add/replace in userPhyloPic and re-render
        userPhyloPic = userPhyloPic.filter(function(g) {{ return g.label !== label; }});
        userPhyloPic.push({{
          label:       label,
          lookup_name: lookupName,
          level:       level,
          leaves:      leafIndices,
          imgSrc:      imgSrc,
          user:        true
        }});
        msg.textContent = '✓ Added PhyloPic for “' + label + '”';
        setTimeout(function() {{ msg.remove(); }}, 1500);
        renderPhyloPic(activeLeafIndices);
        return;
      }}
      msg.textContent = 'No PhyloPic silhouette found for “' + label + '”';
      setTimeout(function() {{ msg.remove(); }}, 2500);
    }} catch(err) {{
      msg.textContent = 'PhyloPic lookup failed: ' + err.message;
      setTimeout(function() {{ msg.remove(); }}, 2500);
    }}
  }}

  outer.addEventListener('dblclick', function(e) {{
    if (e.target.closest('#tooltip') || e.target.closest('#leg')) return;
    const rect2 = outer.getBoundingClientRect();
    const cx2   = (e.clientX - rect2.left - panX) / scale;
    const cy2   = (e.clientY - rect2.top  - panY) / scale;
    const hmX2  = cx2 - TREE_W - labelColW - PHYLO_W;
    const hmY2  = cy2 - HDR_H;
    if (hmY2 < 0 || hmY2 > IMG_H) return;
    const nView2     = activeLeafIndices ? activeLeafIndices.length : N_LEAVES;
    const rowInView2 = Math.min(nView2 - 1, Math.floor(hmY2 / IMG_H * nView2));
    if (rowInView2 < 0) return;
    const leafIdx2 = activeLeafIndices ? activeLeafIndices[rowInView2] : rowInView2;
    const leaf2    = LEAVES[leafIdx2] || {{}};
    if (hmX2 < 0) {{
      // Tree / leaf-label area: use finest available taxonomy for this leaf
      let name2 = null, level2 = null;
      for (let i = NCBI_LEVELS.length - 1; i >= 0; i--) {{
        const sh2 = NCBI_LEVELS[i][0], lbl2 = NCBI_LEVELS[i][1];
        if (leaf2[sh2]) {{ name2 = leaf2[sh2]; level2 = lbl2; break; }}
      }}
      if (!name2) return;
      fetchAndAddPhyloPic(name2, name2, level2, [leafIdx2]);
    }} else if (hmX2 < STRIP1_W) {{
      // Strip 1 column: all visible leaves sharing this strip value
      const sv1 = leaf2[currentStrip1Sh];
      if (!sv1) return;
      const sl1  = (ALL_STRIP_MAPS[currentStrip1Sh] || {{}}).label || STRIP1_LABEL;
      const vs1  = activeLeafIndices ? new Set(activeLeafIndices) : null;
      const leaves1 = [];
      for (let i = 0; i < N_LEAVES; i++) {{
        if (vs1 && !vs1.has(i)) continue;
        if ((LEAVES[i] || {{}})[currentStrip1Sh] === sv1) leaves1.push(i);
      }}
      fetchAndAddPhyloPic(sv1, sv1, sl1, leaves1);
    }} else if (hmX2 < STRIP1_W + STRIP_GAP + STRIP2_W) {{
      // Strip 2 column: all visible leaves sharing this strip value
      const sv2 = leaf2[currentStrip2Sh];
      if (!sv2) return;
      const sl2  = (ALL_STRIP_MAPS[currentStrip2Sh] || {{}}).label || STRIP2_LABEL;
      const vs2  = activeLeafIndices ? new Set(activeLeafIndices) : null;
      const leaves2 = [];
      for (let i = 0; i < N_LEAVES; i++) {{
        if (vs2 && !vs2.has(i)) continue;
        if ((LEAVES[i] || {{}})[currentStrip2Sh] === sv2) leaves2.push(i);
      }}
      fetchAndAddPhyloPic(sv2, sv2, sl2, leaves2);
    }}
  }});

  // ── Filter panel ─────────────────────────────────────────────────────────
  const NCBI_LEVELS = {ncbi_levels_json};
  (function() {{
    const fLevelSel      = document.getElementById('f-level');
    const fValuesSel     = document.getElementById('f-values');
    const fValWrap       = document.getElementById('f-val-wrap');
    const fHitsOnly      = document.getElementById('f-hits-only');
    const fMolsSel       = document.getElementById('f-mols');
    const fMatchLevelsSel = document.getElementById('f-match-levels');
    const treeSvgEl      = document.getElementById('tree-svg');
    const hmStatic       = document.getElementById('hm');
    const hmCanvas       = document.getElementById('hm-canvas');
    const initTreeHTML   = treeSvgEl.innerHTML;

    // Composable filter state
    const fState = {{
      hitsOnly: false,
      ncbiMode: null, ncbiKey: null, ncbiVals: null,
      excludedMols: new Set(),
      matchLevelMode: null, matchLevelVals: null
    }};

    function isAnyFilterActive() {{
      return fState.hitsOnly || fState.ncbiMode !== null ||
             fState.excludedMols.size > 0 || fState.matchLevelMode !== null;
    }}

    function updateLegend(keptIndices) {{
      const view = keptIndices || Array.from({{length: N_LEAVES}}, function(_, i) {{ return i; }});
      const actMasst = new Set(), actWd = new Set(), actS1 = new Set(), actS2 = new Set();
      view.forEach(function(i) {{
        const leaf = LEAVES[i] || {{}};
        const v1 = (leaf[currentStrip1Sh] || '').toLowerCase();
        const v2 = (leaf[currentStrip2Sh] || '').toLowerCase();
        if (v1) actS1.add(v1);
        if (v2) actS2.add(v2);
        (LEAF_HITS[i] || []).forEach(function(h) {{
          if (!activeExcludedMols.has(h[0])) {{
            if (h[1]) actMasst.add(h[1]);
            if (h[2]) actWd.add(h[2]);
          }}
        }});
      }});
      document.querySelectorAll('[data-leg-masst]').forEach(function(el) {{
        el.style.display = actMasst.has(el.dataset.legMasst) ? '' : 'none';
      }});
      document.querySelectorAll('[data-leg-wd]').forEach(function(el) {{
        el.style.display = actWd.has(el.dataset.legWd) ? '' : 'none';
      }});
      document.querySelectorAll('[data-leg-s1]').forEach(function(el) {{
        el.style.display = actS1.has(el.dataset.legS1) ? '' : 'none';
      }});
      document.querySelectorAll('[data-leg-s2]').forEach(function(el) {{
        const key = el.dataset.legS2;
        el.style.display = (key === '__other__' || actS2.has(key)) ? '' : 'none';
      }});
      // hide section headers when all their items are hidden
      ['masst','wd','s1','s2'].forEach(function(sec) {{
        const hdr = document.querySelector('[data-leg-hdr="' + sec + '"]');
        if (!hdr) return;
        const anyVisible = Array.from(document.querySelectorAll('[data-leg-' + sec + ']'))
          .some(function(el) {{ return el.style.display !== 'none'; }});
        hdr.style.display = anyVisible ? '' : 'none';
      }});
    }}

    function restoreDefaultView() {{
      activeExcludedMols      = new Set();
      activeHiddenMatchLevels = new Set();
      activeLeafIndices  = null;
      activeRowH         = ROW_H;
      activeKept2        = null;
      activeHm2RowH      = ROW_H;
      activeLeafViewPos  = null;
      clearSelection();
      treeSvgEl.innerHTML    = initTreeHTML;
      hmCanvas.style.display      = 'none';
      hmStatic.style.visibility   = '';
      document.getElementById('hm2-canvas').style.display     = 'none';
      document.getElementById('hm2').style.visibility         = '';
      updateLegend(null);
      renderPhyloPic(null);
      renderDefaultDots();
    }}

    function applyActiveFilters() {{
      // Sync module-level excluded mols
      activeExcludedMols = new Set(fState.excludedMols);

      // Match-level filter: hide MASST coloring for excluded levels instead of
      // removing the leaf row.  Compute the set of suppressed levels.
      if (fState.matchLevelMode === 'excl' && fState.matchLevelVals) {{
        activeHiddenMatchLevels = new Set(fState.matchLevelVals);
      }} else if (fState.matchLevelMode === 'show' && fState.matchLevelVals) {{
        // hide every level that is NOT in the selection
        const allLevels = new Set();
        for (let i = 0; i < N_LEAVES; i++) {{
          (LEAF_HITS[i] || []).forEach(function(h) {{ if (h[1]) allLevels.add(h[1]); }});
        }}
        activeHiddenMatchLevels = new Set(
          Array.from(allLevels).filter(function(lv) {{ return !fState.matchLevelVals.has(lv); }})
        );
      }} else {{
        activeHiddenMatchLevels = new Set();
      }}

      // Row filter: hitsOnly and NCBI taxonomy only (match-level no longer filters rows)
      const hasRowFilter = fState.hitsOnly || fState.ncbiMode !== null;
      const kept = [];
      for (let i = 0; i < N_LEAVES; i++) {{
        if (fState.hitsOnly) {{
          // A leaf counts as "has a hit" only if it has a visible MASST or WD hit
          const hasHit = (LEAF_HITS[i] || []).some(function(h) {{
            if (activeExcludedMols.has(h[0])) return false;
            return (h[1] && !activeHiddenMatchLevels.has(h[1])) || !!h[2];
          }});
          if (!hasHit) continue;
        }}
        if (fState.ncbiMode && fState.ncbiKey && fState.ncbiVals) {{
          const v = (LEAVES[i] || {{}})[fState.ncbiKey] || '';
          const inSel = fState.ncbiVals.has(v);
          if (fState.ncbiMode === 'show' && !inSel) continue;
          if (fState.ncbiMode === 'excl' &&  inSel) continue;
        }}
        kept.push(i);
      }}
      if (!kept.length) return;
      activeLeafIndices = hasRowFilter ? kept : null;
      activeRowH        = IMG_H / kept.length;
      clearSelection();
      renderFilteredView(kept);
      updateLegend(kept);
      renderPhyloPic(kept);
      renderDefaultDots();
    }}

    // Reset all controls to unchecked/blank (overrides browser form-state restore)
    fHitsOnly.checked = false;
    fState.hitsOnly   = false;

    // ── Populate controls ──────────────────────────────────────────────
    // Clear first so saved-HTML re-loads don't duplicate options.
    fLevelSel.innerHTML = '<option value="">— select level —</option>';
    NCBI_LEVELS.forEach(function(kv) {{
      const opt = document.createElement('option');
      opt.value = kv[0]; opt.textContent = kv[1];
      fLevelSel.appendChild(opt);
    }});
    fMolsSel.innerHTML = '';
    PAIRS.forEach(function(p, idx) {{
      const opt = document.createElement('option');
      opt.value = String(idx); opt.textContent = p.l;
      fMolsSel.appendChild(opt);
    }});
    // Populate match levels from values present in LEAF_HITS[i][*][1]
    (function() {{
      const ML_ORDER = ['subspecies','species','genus','family','order','class','phylum','kingdom'];
      const seen = {{}};
      LEAF_HITS.forEach(function(hits) {{
        hits.forEach(function(h) {{ if (h[1]) seen[h[1]] = 1; }});
      }});
      fMatchLevelsSel.innerHTML = '';
      ML_ORDER.forEach(function(lv) {{
        if (!seen[lv]) return;
        const opt = document.createElement('option');
        opt.value = lv; opt.textContent = lv;
        fMatchLevelsSel.appendChild(opt);
      }});
    }})();

    // ── Colour strip selectors ─────────────────────────────────────────
    (function() {{
      const fStrip1 = document.getElementById('f-strip1');
      const fStrip2 = document.getElementById('f-strip2');
      if (!fStrip1 || !fStrip2) return;

      // Populate with available NCBI levels
      function buildStripOpts(sel, initSh) {{
        sel.innerHTML = '';
        NCBI_LEVELS.forEach(function(kv) {{
          const sh = kv[0], lbl = kv[1];
          if (!ALL_STRIP_MAPS[sh]) return;
          const opt = document.createElement('option');
          opt.value = sh; opt.textContent = lbl;
          if (sh === initSh) opt.selected = true;
          sel.appendChild(opt);
        }});
      }}
      buildStripOpts(fStrip1, STRIP1_INIT_SH);
      buildStripOpts(fStrip2, STRIP2_INIT_SH);

      function buildStripLegendHtml(sh, sectionId) {{
        const map  = ALL_STRIP_MAPS[sh] || {{}};
        const hx   = map.hexColors || {{}};
        const rare = new Set(map.rare || []);
        const def  = map.defHex || '#aaaaaa';
        const lbl  = map.label || sh;
        const MAX  = 15;
        const keys = Object.keys(hx).sort();
        let html = '<b data-leg-hdr="' + sectionId + '">' + lbl + '</b>';
        const shown = keys.slice(0, MAX);
        shown.forEach(function(k) {{
          const cap = k.length < 20 ? k.charAt(0).toUpperCase() + k.slice(1) : k;
          html += '<div class="lr" data-leg-' + sectionId + '="' + k + '">' +
                  '<div class="sw" style="background:' + hx[k] + '"></div>' +
                  '<span>' + cap + '</span></div>';
        }});
        if (keys.length > MAX) {{
          html += '<div class="lr"><span style="color:#999">+' + (keys.length - MAX) + ' more…</span></div>';
        }}
        if (rare.size) {{
          html += '<div class="lr" data-leg-' + sectionId + '="__other__">' +
                  '<div class="sw" style="background:' + def + ';border:1px solid #aaa"></div>' +
                  '<span>Other (&lt;4 leaves)</span></div>';
        }}
        return html;
      }}

      function rebuildStripLegend() {{
        const s1El = document.getElementById('leg-s1-section');
        const s2El = document.getElementById('leg-s2-section');
        if (s1El) s1El.innerHTML = buildStripLegendHtml(currentStrip1Sh, 's1');
        if (s2El) s2El.innerHTML = buildStripLegendHtml(currentStrip2Sh, 's2');
        // Refresh visibility via updateLegend with whatever is currently shown
        updateLegend(activeLeafIndices);
      }}

      function onStripChange() {{
        currentStrip1Sh = fStrip1.value || STRIP1_INIT_SH;
        currentStrip2Sh = fStrip2.value || STRIP2_INIT_SH;
        rebuildStripLegend();
        const allIdx = Array.from({{length: N_LEAVES}}, function(_, i) {{ return i; }});
        // Always draw to canvas and hide the static PNG (strip differs from baked image)
        const hmCvs  = document.getElementById('hm-canvas');
        const hmStat = document.getElementById('hm');
        if (hmCvs) {{
          drawHeatmapCanvas(hmCvs, activeLeafIndices || allIdx);
          hmCvs.style.display     = 'block';
          if (hmStat) hmStat.style.visibility = 'hidden';
        }}
        const hm2Cvs  = document.getElementById('hm2-canvas');
        const hm2Stat = document.getElementById('hm2');
        if (hm2Cvs && hm2Cvs.style.display !== 'none') {{
          drawHeatmapCanvas(hm2Cvs, activeKept2 || allIdx);
          if (hm2Stat) hm2Stat.style.visibility = 'hidden';
        }}
      }}

      fStrip1.addEventListener('change', onStripChange);
      fStrip2.addEventListener('change', onStripChange);
    }})();

    // ── Row filters ────────────────────────────────────────────────────
    fHitsOnly.addEventListener('change', function() {{
      fState.hitsOnly = this.checked;
      if (isAnyFilterActive()) applyActiveFilters(); else restoreDefaultView();
    }});

    fLevelSel.addEventListener('change', function() {{
      const key = this.value;
      fValWrap.style.display = key ? 'block' : 'none';
      if (!key) {{
        fState.ncbiMode = null; fState.ncbiKey = null; fState.ncbiVals = null;
        if (!isAnyFilterActive()) restoreDefaultView();
        return;
      }}
      const seen = {{}};
      LEAVES.forEach(function(lf) {{ if (lf[key]) seen[lf[key]] = 1; }});
      fValuesSel.innerHTML = '';
      Object.keys(seen).sort().forEach(function(v) {{
        const opt = document.createElement('option');
        opt.value = v; opt.textContent = v;
        fValuesSel.appendChild(opt);
      }});
    }});

    function applyNcbi(mode) {{
      const key = fLevelSel.value;
      if (!key) return;
      const sel = new Set(
        Array.from(fValuesSel.selectedOptions).map(function(o) {{ return o.value; }})
      );
      if (!sel.size) return;
      fState.ncbiMode = mode; fState.ncbiKey = key; fState.ncbiVals = sel;
      applyActiveFilters();
    }}

    function clearRows() {{
      fState.hitsOnly = false; fState.ncbiMode = null; fState.ncbiKey = null; fState.ncbiVals = null;
      fState.matchLevelMode = null; fState.matchLevelVals = null;
      fHitsOnly.checked = false; fLevelSel.value = ''; fValWrap.style.display = 'none';
      fMatchLevelsSel.selectedIndex = -1;
      if (isAnyFilterActive()) applyActiveFilters(); else restoreDefaultView();
    }}

    // ── Molecule filters ───────────────────────────────────────────────
    function applyMolFilter(mode) {{
      const sel = new Set(
        Array.from(fMolsSel.selectedOptions).map(function(o) {{ return parseInt(o.value); }})
      );
      if (!sel.size) return;
      if (mode === 'excl') {{
        fState.excludedMols = sel;
      }} else {{
        const all = new Set(PAIRS.map(function(_, i) {{ return i; }}));
        fState.excludedMols = new Set([...all].filter(function(i) {{ return !sel.has(i); }}));
      }}
      applyActiveFilters();
    }}

    function clearMols() {{
      fState.excludedMols = new Set();
      fMolsSel.selectedIndex = -1;
      if (isAnyFilterActive()) applyActiveFilters(); else restoreDefaultView();
    }}

    function applyMatchLevel(mode) {{
      const sel = new Set(
        Array.from(fMatchLevelsSel.selectedOptions).map(function(o) {{ return o.value; }})
      );
      if (!sel.size) return;
      fState.matchLevelMode = mode; fState.matchLevelVals = sel;
      applyActiveFilters();
    }}

    function clearMatchLevels() {{
      fState.matchLevelMode = null; fState.matchLevelVals = null;
      fMatchLevelsSel.selectedIndex = -1;
      if (isAnyFilterActive()) applyActiveFilters(); else restoreDefaultView();
    }}

    document.getElementById('f-btn-show').addEventListener('click',  function() {{ applyNcbi('show'); }});
    document.getElementById('f-btn-excl').addEventListener('click',  function() {{ applyNcbi('excl'); }});
    document.getElementById('f-btn-clear').addEventListener('click', clearRows);
    document.getElementById('f-btn-mol-show').addEventListener('click',  function() {{ applyMolFilter('show'); }});
    document.getElementById('f-btn-mol-excl').addEventListener('click',  function() {{ applyMolFilter('excl'); }});
    document.getElementById('f-btn-mol-clear').addEventListener('click', clearMols);
    document.getElementById('f-btn-ml-show').addEventListener('click',  function() {{ applyMatchLevel('show'); }});
    document.getElementById('f-btn-ml-excl').addEventListener('click',  function() {{ applyMatchLevel('excl'); }});
    document.getElementById('f-btn-ml-clear').addEventListener('click', clearMatchLevels);
    document.getElementById('filter-panel').addEventListener('mousedown', function(e) {{ e.stopPropagation(); }});
    document.getElementById('filter-panel').addEventListener('click',     function(e) {{ e.stopPropagation(); }});
  }})();

  // ── Clustered heatmap toggle ─────────────────────────────────────────────
  (function() {{
    const hm2Section     = document.getElementById('hm2-section');
    const hm2Toggle      = document.getElementById('hm2-toggle');
    const hm2Wrap        = document.getElementById('hm2-wrap');
    const hm2HdrDiv      = document.getElementById('hm2-header-div');
    const hm2HdrSpacer   = document.getElementById('hm2-hdr-spacer');
    const hm2ContentSpc  = document.getElementById('hm2-content-spacer');

    function setHm2Visible(vis) {{
      const d = vis ? '' : 'none';
      hm2Wrap.style.display        = vis ? 'inline-block' : 'none';
      hm2HdrDiv.style.display      = d;
      hm2HdrSpacer.style.display   = d;
      hm2ContentSpc.style.display  = d;
      rowMarker2.style.display     = 'none';   // always reset; markRow will re-show if needed
    }}

    // Only offer the toggle when 2+ molecules are present
    if (PAIRS.length >= 2) {{
      hm2Section.style.display = '';
      setHm2Visible(false);   // hidden by default
      hm2Toggle.addEventListener('change', function() {{
        setHm2Visible(this.checked);
      }});
    }} else {{
      setHm2Visible(false);   // single molecule — permanently hidden
    }}
  }})();


  document.getElementById('label-size').addEventListener('input', function() {{
    labelSizeScale = parseFloat(this.value) || 1;
    labelColW      = Math.round(LABEL_COL_W * labelSizeScale);

    // Resize the SVG element (which already includes the label column in its viewBox)
    // and the matching header spacer so the heatmap shifts right accordingly.
    const svgEl = document.getElementById('tree-svg');
    if (svgEl) {{
      const newW = TREE_W + labelColW;
      svgEl.setAttribute('width', newW);
      svgEl.setAttribute('viewBox', '0 0 ' + newW + ' ' + IMG_H);
    }}
    const treeSpacer = document.getElementById('tree-spacer');
    if (treeSpacer) treeSpacer.style.width = (TREE_W + labelColW) + 'px';

    // Update all layout constants that depend on label column width
    HM2_X0    = TREE_W + labelColW + PHYLO_W + IMG_W + SPACER_W;
    CONTENT_W = TREE_W + labelColW + PHYLO_W + IMG_W + SPACER_W + IMG_W;
    updateRowMarkerBounds();

    // Scale every text element in the current tree SVG
    if (svgEl) {{
      svgEl.querySelectorAll('text').forEach(function(t) {{
        const base = parseFloat(t.getAttribute('data-base-fs') || t.getAttribute('font-size') || '4');
        if (!t.getAttribute('data-base-fs')) t.setAttribute('data-base-fs', String(base));
        t.setAttribute('font-size', Math.max(0.5, base * labelSizeScale).toFixed(2));
      }});
    }}

    fitToWindow();
  }});

  // ── Shared canvas heatmap renderer ───────────────────────────────────────
  function drawHeatmapCanvas(canvas, keptIndices) {{
    const nKept = keptIndices.length;
    const rowPx = IMG_H / nKept;
    canvas.width  = IMG_W;
    canvas.height = IMG_H;
    const ctx     = canvas.getContext('2d');
    const imgData = ctx.createImageData(IMG_W, IMG_H);
    const pix     = imgData.data;

    for (let ki = 0; ki < nKept; ki++) {{
      const li   = keptIndices[ki];
      const leaf = LEAVES[li] || {{}};
      const hits = LEAF_HITS[li] || [];
      const yStart = Math.round(ki * rowPx);
      const yEnd   = Math.min(IMG_H, Math.round((ki + 1) * rowPx));
      if (yStart >= yEnd) continue;

      const rowBuf = new Uint8Array(IMG_W * 4);
      if (activeHighlightLeafSet.has(li)) {{
        for (let p = 0; p < IMG_W * 4; p += 4) {{
          rowBuf[p]=255; rowBuf[p+1]=240; rowBuf[p+2]=180; rowBuf[p+3]=255;
        }}
      }} else {{
        rowBuf.fill(255);
      }}

      const _sm1 = ALL_STRIP_MAPS[currentStrip1Sh] || {{}};
      const _s1c = _sm1.colors || STRIP1_COLORS;
      const _s1d = _sm1.def || STRIP1_DEF;
      const _s1r = _sm1.rare ? new Set(_sm1.rare) : new Set();
      const s1Key = (leaf[currentStrip1Sh] || '').toLowerCase();
      const s1Rgb = (s1Key && !_s1r.has(s1Key)) ? (_s1c[s1Key] || _s1d) : _s1d;
      for (let x = 0; x < STRIP1_W; x++) {{
        rowBuf[x*4]=s1Rgb[0]; rowBuf[x*4+1]=s1Rgb[1]; rowBuf[x*4+2]=s1Rgb[2]; rowBuf[x*4+3]=255;
      }}
      const _sm2 = ALL_STRIP_MAPS[currentStrip2Sh] || {{}};
      const _s2c = _sm2.colors || STRIP2_COLORS;
      const _s2d = _sm2.def || STRIP2_DEF;
      const _s2r = _sm2.rare ? new Set(_sm2.rare) : STRIP2_RARE;
      const s2Key = (leaf[currentStrip2Sh] || '').toLowerCase();
      const s2Rgb = (!s2Key || _s2r.has(s2Key)) ? _s2d : (_s2c[s2Key] || _s2d);
      for (let x = STRIP1_W + STRIP_GAP; x < STRIP1_W + STRIP_GAP + STRIP2_W; x++) {{
        rowBuf[x*4]=s2Rgb[0]; rowBuf[x*4+1]=s2Rgb[1]; rowBuf[x*4+2]=s2Rgb[2]; rowBuf[x*4+3]=255;
      }}
      const hitMap = {{}};
      for (const h of hits) hitMap[h[0]] = h;
      for (let j = 0; j < PAIRS.length; j++) {{
        const x0 = STRIP1_W + STRIP_GAP + STRIP2_W + STRIP_GAP + j * (CELL_W * 2 + COL_GAP);
        if (activeExcludedMols.has(j)) {{
          for (let x = x0; x < x0 + CELL_W * 2; x++) {{
            rowBuf[x*4]=232; rowBuf[x*4+1]=232; rowBuf[x*4+2]=232; rowBuf[x*4+3]=255;
          }}
          continue;
        }}
        const h = hitMap[j];
        if (!h) continue;
        if (h[1] && !activeHiddenMatchLevels.has(h[1])) {{
          const mc = MASST_COLORS[h[1]];
          if (mc) for (let x = x0; x < x0 + CELL_W; x++) {{
            rowBuf[x*4]=mc[0]; rowBuf[x*4+1]=mc[1]; rowBuf[x*4+2]=mc[2]; rowBuf[x*4+3]=255;
          }}
        }}
        if (h[2]) {{
          const wc = WD_COLORS[h[2]];
          if (wc) for (let x = x0+CELL_W; x < x0+CELL_W*2; x++) {{
            rowBuf[x*4]=wc[0]; rowBuf[x*4+1]=wc[1]; rowBuf[x*4+2]=wc[2]; rowBuf[x*4+3]=255;
          }}
        }}
      }}
      for (let y = yStart; y < yEnd; y++) pix.set(rowBuf, y * IMG_W * 4);
    }}

    ctx.putImageData(imgData, 0, 0);
  }}

  // ── Filtered view renderer ────────────────────────────────────────────────
  function renderFilteredView(keptIndices) {{
    const nKept  = keptIndices.length;
    const rowPx  = IMG_H / nKept;
    const fontSz = Math.max(2, Math.min(6, rowPx * 0.7));
    const keptSet = new Set(keptIndices);

    // Post-order: assign y positions to kept leaves; propagate to internal nodes
    const leafNewY = new Map();
    const nodeYMap = new Map();
    let lc = 0;

    function computeY(node) {{
      if ('i' in node) {{
        if (!keptSet.has(node.i)) return null;
        leafNewY.set(node.i, lc++);
        return lc - 1;
      }}
      let mn = Infinity, mx = -Infinity, any = false;
      for (const ch of node.ch) {{
        const y = computeY(ch);
        if (y !== null) {{ if (y < mn) mn = y; if (y > mx) mx = y; any = true; }}
      }}
      if (!any) return null;
      const y = (mn + mx) / 2;
      nodeYMap.set(node.id, y);
      return y;
    }}
    computeY(TREE_DATA);
    activeLeafViewPos = leafNewY;  // Map: leaf index → tree-row position

    // Build SVG path segments (one per parent→child edge)
    const segs = [], txts = [];
    function buildSegs(node) {{
      if ('i' in node) return;
      if (!nodeYMap.has(node.id)) return;
      const px = (node.x * TREE_W).toFixed(2);
      const py = ((nodeYMap.get(node.id) + 0.5) * rowPx).toFixed(2);
      for (const ch of node.ch) {{
        let chX, chY;
        if ('i' in ch) {{
          if (!keptSet.has(ch.i)) continue;
          chX = ch.x; chY = leafNewY.get(ch.i);
          const nm = NAME_ARR[ch.i] || '';
          if (nm) {{
            const safe = nm.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
            const leaf = LEAVES[ch.i] || {{}};
            const ott  = leaf.ott || '';
            const hitColor = LEAF_HIT_COLORS[ott] || '#555555';
            const effectiveFsz = Math.max(0.5, fontSz * labelSizeScale).toFixed(2);
            txts.push('<text x="' + (TREE_W + 3).toFixed(1) + '" y="' + ((chY + 0.5) * rowPx).toFixed(1) +
              '" font-size="' + effectiveFsz + '" data-base-fs="' + fontSz.toFixed(1) + '"' +
              ' font-family="sans-serif" dominant-baseline="middle" fill="' + hitColor + '">' + safe + '</text>');
          }}
        }} else {{
          if (!nodeYMap.has(ch.id)) continue;
          chX = ch.x; chY = nodeYMap.get(ch.id);
        }}
        const cx = (chX * TREE_W).toFixed(2);
        const cy = ((chY + 0.5) * rowPx).toFixed(2);
        segs.push('M' + px + ',' + py + 'L' + px + ',' + cy + 'L' + cx + ',' + cy);
        buildSegs(ch);
      }}
    }}
    buildSegs(TREE_DATA);

    const treeSvgEl = document.getElementById('tree-svg');
    treeSvgEl.innerHTML =
      '<path d="' + segs.join(' ') + '" fill="none" stroke="#444444" stroke-width="0.4"/>' +
      txts.join('');

    // Draw hm1
    const hmStatic = document.getElementById('hm');
    const hmCanvas = document.getElementById('hm-canvas');
    drawHeatmapCanvas(hmCanvas, keptIndices);
    hmCanvas.style.display    = 'block';
    hmStatic.style.visibility = 'hidden';

    // Draw hm2: same leaves sorted by clustered rank
    const kept2 = keptIndices.slice().sort(function(a, b) {{
      return (CLUST_RANK[a] || 0) - (CLUST_RANK[b] || 0);
    }});
    activeKept2   = kept2;
    activeHm2RowH = IMG_H / kept2.length;
    const hm2Cvs = document.getElementById('hm2-canvas');
    drawHeatmapCanvas(hm2Cvs, kept2);
    hm2Cvs.style.display                             = 'block';
    document.getElementById('hm2').style.visibility = 'hidden';
  }}

  // ── SVG heatmap builder (vector rects, mirrors drawHeatmapCanvas logic) ──────
  function buildHeatmapSVG(leafOrder) {{
    const nKept = leafOrder.length;
    const rowH  = IMG_H / nKept;
    const p     = [];

    p.push('<rect x="0" y="0" width="' + IMG_W + '" height="' + IMG_H + '" fill="white"/>');

    const _sm1 = ALL_STRIP_MAPS[currentStrip1Sh] || {{}};
    const _s1c = _sm1.colors || STRIP1_COLORS;
    const _s1d = _sm1.def   || STRIP1_DEF;
    const _s1r = _sm1.rare  ? new Set(_sm1.rare) : new Set();
    const _sm2 = ALL_STRIP_MAPS[currentStrip2Sh] || {{}};
    const _s2c = _sm2.colors || STRIP2_COLORS;
    const _s2d = _sm2.def   || STRIP2_DEF;
    const _s2r = _sm2.rare  ? new Set(_sm2.rare) : STRIP2_RARE;

    function rgb(c) {{ return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }}
    function rect(x, y, w, h, fill) {{
      return '<rect x="' + x + '" y="' + y.toFixed(3) + '" width="' + w +
             '" height="' + h.toFixed(3) + '" fill="' + fill + '"/>';
    }}

    // Merge consecutive leaves with identical strip colors into one taller rect
    // to keep file size manageable for large trees.
    var s1Runs = [], s2Runs = [];  // array of run objects
    var s1Cur = null, s2Cur = null;

    for (let ki = 0; ki < nKept; ki++) {{
      const li   = leafOrder[ki];
      const leaf = LEAVES[li] || {{}};
      const hits = LEAF_HITS[li] || [];
      const y    = ki * rowH;
      const h    = rowH;

      // ── Strip 1 (run-length encoded) ──
      const s1Key  = (leaf[currentStrip1Sh] || '').toLowerCase();
      const s1Fill = rgb((s1Key && !_s1r.has(s1Key)) ? (_s1c[s1Key] || _s1d) : _s1d);
      if (s1Cur && s1Cur.fill === s1Fill) {{
        s1Cur.yEnd = y + h;
      }} else {{
        if (s1Cur) s1Runs.push(s1Cur);
        s1Cur = {{ fill: s1Fill, yStart: y, yEnd: y + h }};
      }}

      // ── Strip 2 (run-length encoded) ──
      const s2Key  = (leaf[currentStrip2Sh] || '').toLowerCase();
      const s2Fill = rgb((!s2Key || _s2r.has(s2Key)) ? _s2d : (_s2c[s2Key] || _s2d));
      if (s2Cur && s2Cur.fill === s2Fill) {{
        s2Cur.yEnd = y + h;
      }} else {{
        if (s2Cur) s2Runs.push(s2Cur);
        s2Cur = {{ fill: s2Fill, yStart: y, yEnd: y + h }};
      }}

      // ── Molecule columns (only emit colored cells) ──
      const hitMap = {{}};
      for (const hh of hits) hitMap[hh[0]] = hh;

      for (let j = 0; j < PAIRS.length; j++) {{
        const x0 = STRIP1_W + STRIP_GAP + STRIP2_W + STRIP_GAP + j * (CELL_W * 2 + COL_GAP);
        if (activeExcludedMols.has(j)) {{
          p.push(rect(x0, y, CELL_W * 2, h, '#e8e8e8'));
          continue;
        }}
        const hh = hitMap[j];
        if (!hh) continue;
        if (hh[1] && !activeHiddenMatchLevels.has(hh[1])) {{
          const mc = MASST_COLORS[hh[1]];
          if (mc) p.push(rect(x0, y, CELL_W, h, rgb(mc)));
        }}
        if (hh[2]) {{
          const wc = WD_COLORS[hh[2]];
          if (wc) p.push(rect(x0 + CELL_W, y, CELL_W, h, rgb(wc)));
        }}
      }}
    }}
    if (s1Cur) s1Runs.push(s1Cur);
    if (s2Cur) s2Runs.push(s2Cur);

    // Emit strip runs
    const s2x = STRIP1_W + STRIP_GAP;
    for (const r of s1Runs) p.push(rect(0,   r.yStart, STRIP1_W, r.yEnd - r.yStart, r.fill));
    for (const r of s2Runs) p.push(rect(s2x, r.yStart, STRIP2_W, r.yEnd - r.yStart, r.fill));

    return p.join('');
  }}

  // ── SVG export ───────────────────────────────────────────────────────────────
  function exportAsSVG() {{
    const totalW = Math.round(CONTENT_W);
    const totalH = HDR_H + IMG_H;
    const ser    = new XMLSerializer();

    function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

    const p = [];
    p.push('<?xml version="1.0" encoding="UTF-8"?>');
    p.push('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"');
    p.push('  width="' + totalW + '" height="' + totalH + '" viewBox="0 0 ' + totalW + ' ' + totalH + '">');
    p.push('<rect width="' + totalW + '" height="' + totalH + '" fill="white"/>');

    // Tree SVG (offset to below the header)
    const treeSvgEl = document.getElementById('tree-svg');
    if (treeSvgEl) {{
      const treeVB    = treeSvgEl.getAttribute('viewBox') || ('0 0 ' + (TREE_W + labelColW) + ' ' + IMG_H);
      const treeInner = Array.from(treeSvgEl.childNodes)
          .map(function(n) {{ return n.nodeType === 1 ? ser.serializeToString(n) : ''; }}).join('');
      p.push('<svg x="0" y="' + HDR_H + '" width="' + (TREE_W + labelColW) + '" height="' + IMG_H + '" viewBox="' + esc(treeVB) + '">');
      p.push(treeInner);
      p.push('</svg>');
    }}

    // PhyloPic column
    const phyloColEl = document.getElementById('phylo-col');
    const phyloX = TREE_W + labelColW;
    if (phyloColEl) {{
      p.push('<g transform="translate(' + phyloX + ',' + HDR_H + ')">');
      Array.from(phyloColEl.querySelectorAll('.phylo-entry')).forEach(function(entry) {{
        const top = parseFloat(entry.style.top)    || 0;
        const h   = parseFloat(entry.style.height) || 20;
        const sz  = Math.min(h, PHYLO_W - 4);
        const img = entry.querySelector('img');
        if (img) {{
          p.push('<image x="2" y="' + top.toFixed(1) + '" width="' + sz.toFixed(1) + '" height="' + sz.toFixed(1) + '" href="' + img.src + '"/>');
        }}
        const lbl = entry.querySelector('.phylo-label b');
        if (lbl && h >= 16) {{
          const tx = img ? (sz + 5).toFixed(1) : '4';
          p.push('<text x="' + tx + '" y="' + (top + h / 2).toFixed(1) + '" font-size="9" dominant-baseline="middle" font-family="sans-serif">' + esc(lbl.textContent) + '</text>');
        }}
      }});
      // Include bracket/connector SVG overlay
      const ovSvgEl = phyloColEl.querySelector('svg');
      if (ovSvgEl) {{
        const ovInner = Array.from(ovSvgEl.childNodes)
            .map(function(n) {{ return n.nodeType === 1 ? ser.serializeToString(n) : ''; }}).join('');
        p.push(ovInner);
      }}
      p.push('</g>');
    }}

    // HM1 & HM2 — vector rects (one per leaf per colored cell)
    const hmX       = phyloX + PHYLO_W;
    const hm1Order  = activeLeafIndices || Array.from({{length: N_LEAVES}}, function(_, i) {{ return i; }});
    const hm2Order  = activeKept2 || CLUST_ORDER;
    p.push('<svg x="' + hmX   + '" y="' + HDR_H + '" width="' + IMG_W + '" height="' + IMG_H + '" overflow="hidden">');
    p.push(buildHeatmapSVG(hm1Order));
    p.push('</svg>');
    p.push('<svg x="' + HM2_X0 + '" y="' + HDR_H + '" width="' + IMG_W + '" height="' + IMG_H + '" overflow="hidden">');
    p.push(buildHeatmapSVG(hm2Order));
    p.push('</svg>');

    // Column headers (each header SVG is positioned at its heatmap x-offset)
    Array.from(document.querySelectorAll('#header-row svg')).forEach(function(hsvg, idx) {{
      const xPos  = (idx === 0) ? hmX : HM2_X0;
      const w     = parseFloat(hsvg.getAttribute('width')  || IMG_W);
      const h     = parseFloat(hsvg.getAttribute('height') || HDR_H);
      const vb    = hsvg.getAttribute('viewBox') || ('0 0 ' + w + ' ' + h);
      const inner = Array.from(hsvg.childNodes)
          .map(function(n) {{ return n.nodeType === 1 ? ser.serializeToString(n) : ''; }}).join('');
      p.push('<svg x="' + xPos + '" y="0" width="' + w + '" height="' + h + '" viewBox="' + esc(vb) + '" overflow="visible">');
      p.push(inner);
      p.push('</svg>');
    }});

    // Separator line between tree and heatmap
    p.push('<line x1="' + hmX + '" y1="0" x2="' + hmX + '" y2="' + totalH + '" stroke="#bbbbbb" stroke-width="2"/>');

    // Legend (visible items only)
    const legEl = document.getElementById('leg');
    if (legEl) {{
      const items = [];
      Array.from(legEl.children).forEach(function(child) {{
        if (child.style && child.style.display === 'none') return;
        if (child.tagName === 'B') {{
          items.push({{ type: 'hdr', text: child.textContent }});
        }} else if (child.classList && child.classList.contains('lr') && child.style.display !== 'none') {{
          const sw = child.querySelector('.sw');
          const sp = child.querySelector('span');
          if (sw && sp) items.push({{ type: 'item', color: sw.style.background, text: sp.textContent }});
        }}
      }});
      const LW = 175, LPX = 10, LPY = 8, LIH = 14;
      let legH = LPY * 2;
      items.forEach(function(it) {{ legH += (it.type === 'hdr' ? LIH + 4 : LIH); }});
      const legX = 14, legY = totalH - legH;
      p.push('<rect x="' + legX + '" y="' + legY + '" width="' + LW + '" height="' + legH + '" fill="white" stroke="#cccccc" stroke-width="1" rx="4"/>');
      let ly = legY + LPY;
      items.forEach(function(it) {{
        if (it.type === 'hdr') {{
          ly += 4;
          p.push('<text x="' + (legX + LPX) + '" y="' + (ly + 10) + '" font-size="10" font-weight="bold" font-family="sans-serif">' + esc(it.text) + '</text>');
          ly += LIH + 2;
        }} else {{
          p.push('<rect x="' + (legX + LPX) + '" y="' + (ly + 2) + '" width="13" height="9" fill="' + esc(it.color) + '" stroke="#ddd" stroke-width="0.5"/>');
          p.push('<text x="' + (legX + LPX + 18) + '" y="' + (ly + 10) + '" font-size="10" font-family="sans-serif">' + esc(it.text) + '</text>');
          ly += LIH;
        }}
      }});
    }}

    p.push('</svg>');
    const blob = new Blob([p.join('')], {{ type: 'image/svg+xml;charset=utf-8' }});
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'metabotree_heatmap.svg';
    document.body.appendChild(a);
    a.click();
    setTimeout(function() {{ document.body.removeChild(a); URL.revokeObjectURL(url); }}, 500);
  }}

  // ── Save as HTML ─────────────────────────────────────────────────────────
  function saveAsHTML() {{
    const hmCvs  = document.getElementById('hm-canvas');
    const hmImg  = document.getElementById('hm');
    const hm2Cvs = document.getElementById('hm2-canvas');
    const hm2Img = document.getElementById('hm2');

    // Is a canvas-filtered view currently active?
    const hmIsCanvas  = hmCvs  && hmCvs.style.display  !== 'none';
    const hm2IsCanvas = hm2Cvs && hm2Cvs.style.display !== 'none';

    // Save originals so we can restore the live DOM after serialization.
    const origHmSrc      = hmImg  ? hmImg.src               : '';
    const origHm2Src     = hm2Img ? hm2Img.src              : '';
    const origHmVis      = hmImg  ? hmImg.style.visibility   : '';
    const origHm2Vis     = hm2Img ? hm2Img.style.visibility  : '';
    const origHmCvsDsp   = hmCvs  ? hmCvs.style.display      : '';
    const origHm2CvsDsp  = hm2Cvs ? hm2Cvs.style.display     : '';

    // If canvas is active, bake its pixels into the <img> and make it visible
    // (canvas pixel data is not serialised by outerHTML).
    if (hmIsCanvas) {{
      hmImg.src                = hmCvs.toDataURL('image/png');
      hmImg.style.visibility   = '';
      hmCvs.style.display      = 'none';
    }}
    if (hm2IsCanvas) {{
      hm2Img.src               = hm2Cvs.toDataURL('image/png');
      hm2Img.style.visibility  = '';
      hm2Cvs.style.display     = 'none';
    }}

    // Collect filter control state to embed in the restore script.
    function getSelected(id) {{
      const s = document.getElementById(id);
      return s ? Array.from(s.selectedOptions).map(function(o) {{ return o.value; }}) : [];
    }}
    function getVal(id)     {{ const e = document.getElementById(id); return e ? e.value   : ''; }}
    function getChecked(id) {{ const e = document.getElementById(id); return e ? e.checked : false; }}

    const savedState = {{
      panX: panX, panY: panY, scale: scale,
      labelSize:    getVal('label-size'),
      hitsOnly:     getChecked('f-hits-only'),
      fLevel:       getVal('f-level'),
      fValues:      getSelected('f-values'),
      fMatchLevels: getSelected('f-match-levels'),
      fMols:        getSelected('f-mols'),
      hm2Checked:   getChecked('hm2-toggle'),
      activeLeafIndices:  activeLeafIndices,
      activeRowH:         activeRowH,
      activeExcludedMols: Array.from(activeExcludedMols),
      activeKept2:        activeKept2,
      activeHm2RowH:      activeHm2RowH,
      activeLeafViewPos:  activeLeafViewPos ? Array.from(activeLeafViewPos.entries()) : null,
      userPhyloPic:       userPhyloPic
    }};

    // Build the restore code (pure JS, no <script> tags — those are added below).
    // IMPORTANT: no literal newlines inside JS string literals (raw newline = SyntaxError).
    const restoreCode =
      '(function(){{' +
        'var S=' + JSON.stringify(savedState) + ';' +
        'function rs(id,vals){{' +
          'var s=document.getElementById(id);if(!s)return;' +
          'var vs=new Set(vals);' +
          'Array.from(s.options).forEach(function(o){{o.selected=vs.has(o.value);}});' +
        '}}' +
        'function after(){{' +
          'if(typeof window._mtSetActive==="function"){{window._mtSetActive(S.activeLeafIndices,S.activeRowH,S.activeExcludedMols,S.activeKept2,S.activeHm2RowH,S.activeLeafViewPos);}}' +
          'if(typeof window._mtSetUserPhyloPic==="function"){{window._mtSetUserPhyloPic(S.userPhyloPic||[]);}}' +
          'var cv=document.getElementById("canvas");var el;' +
          'el=document.getElementById("label-size");' +
          'if(el){{el.value=S.labelSize;el.dispatchEvent(new Event("input"));}}' +
          'if(cv)cv.style.transform="translate("+S.panX+"px,"+S.panY+"px) scale("+S.scale+")";' +
          'el=document.getElementById("f-hits-only");if(el)el.checked=S.hitsOnly;' +
          'el=document.getElementById("f-level");' +
          'if(el){{el.value=S.fLevel;if(S.fLevel)el.dispatchEvent(new Event("change"));}}' +
          'rs("f-values",S.fValues);' +
          'rs("f-match-levels",S.fMatchLevels);rs("f-mols",S.fMols);' +
          'el=document.getElementById("hm2-toggle");' +
          'if(el&&S.hm2Checked!==el.checked){{el.checked=S.hm2Checked;el.dispatchEvent(new Event("change"));}}' +
        '}}' +
        'if(document.readyState==="loading"){{document.addEventListener("DOMContentLoaded",after);}}else{{after();}}' +
      '}})();';

    // Serialise the (possibly patched) document.
    const html = document.documentElement.outerHTML;

    // Restore live DOM before anything is redrawn.
    if (hmIsCanvas) {{
      hmImg.src              = origHmSrc;
      hmImg.style.visibility = origHmVis;
      hmCvs.style.display    = origHmCvsDsp;
    }}
    if (hm2IsCanvas) {{
      hm2Img.src               = origHm2Src;
      hm2Img.style.visibility  = origHm2Vis;
      hm2Cvs.style.display     = origHm2CvsDsp;
    }}

    // Inject restore script before the LAST </body> tag (not any in JS code).
    // Use lastIndexOf so we hit the real closing tag, not occurrences inside JS strings.
    // Use a function replacement to avoid '$' special characters in JSON values.
    var cbIdx = html.lastIndexOf('<' + '/body>');
    var injected = cbIdx < 0 ? html
      : html.slice(0, cbIdx) + '<script>' + restoreCode + '<\\/script><' + '/body>' + html.slice(cbIdx + 7);
    const finalHtml = '<!DOCTYPE html>' + injected;

    const blob = new Blob([finalHtml], {{ type: 'text/html;charset=utf-8' }});
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'metabotree_heatmap.html';
    document.body.appendChild(a);
    a.click();
    setTimeout(function() {{ document.body.removeChild(a); URL.revokeObjectURL(url); }}, 500);
  }}

  (function() {{
    const eb = document.getElementById('export-btn');
    if (eb) eb.addEventListener('click', exportAsSVG);
    const pb = document.getElementById('print-btn');
    if (pb) pb.addEventListener('click', function() {{ window.print(); }});
    const sb = document.getElementById('save-html-btn');
    if (sb) sb.addEventListener('click', saveAsHTML);
  }})();

}})();
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate MetaboTree HTML heatmap")
    ap.add_argument("--metadata",      required=True,  help="merged_metadata.tsv")
    ap.add_argument("--tree",          required=True,  help="labelled_supertree_subset_prepped.nwk")
    ap.add_argument("--molecules",     default=None,
                    help="structuremasst_input_unique.tsv: the SMILES source, and "
                         "the order the molecules appear in on the tree")
    ap.add_argument("--output",        default="tree_heatmap.html")
    ap.add_argument("--row-height",    type=float, default=0.45, dest="row_h",    metavar="PX")
    ap.add_argument("--cell-width",    type=int,   default=10,   dest="cell_w",   metavar="PX")
    ap.add_argument("--tree-width",    type=int,   default=2000, dest="tree_w",   metavar="PX")
    ap.add_argument("--strip1",        default="NCBIKingdom", dest="strip1",
                    help="Column for the first (leftmost) lineage colour strip (default: NCBIKingdom)")
    ap.add_argument("--strip2",        default="NCBIClass",   dest="strip2",
                    help="Column for the second lineage colour strip (default: NCBIClass)")
    ap.add_argument("--kingdom-width", type=int,   default=15,   dest="strip1_w", metavar="PX")
    ap.add_argument("--class-width",   type=int,   default=15,   dest="strip2_w", metavar="PX")
    ap.add_argument("--header-height", type=int,   default=200,  dest="header_h", metavar="PX")
    ap.add_argument("--min-hits",      type=int,   default=0,    dest="min_hits",
                    help="Prune leaves with fewer hits (0 = keep all)")
    ap.add_argument("--kingdom",       default=None,
                    help="Filter to one NCBIKingdom, e.g. --kingdom animalia")
    ap.add_argument("--min-specificity", default="", dest="min_specificity",
                    choices=["", "subspecies", "species", "genus", "family", "order", "class", "phylum", "kingdom"],
                    help="Only retain flexible MASST matches at this taxonomic level or finer.")
    args = ap.parse_args()

    # ── metadata ──────────────────────────────────────────────────────────────
    print("Loading metadata …")
    df_full = pd.read_csv(args.metadata, sep="\t", low_memory=False, index_col=0)
    print(f"  {len(df_full):,} rows loaded")

    if args.min_specificity:
        _min_fin = _TAX_FINENESS.get(args.min_specificity.lower(), 0)
        _flex_cols = [c for c in df_full.columns if c.startswith("masstFlexibleMatch_")]
        for _col in _flex_cols:
            df_full[_col] = df_full[_col].where(
                df_full[_col].map(
                    lambda v: _TAX_FINENESS.get(str(v).strip().lower(), -1) >= _min_fin
                              if pd.notna(v) else False
                ),
                other=pd.NA,
            )
        print(f"  Applied min_specificity={args.min_specificity!r} to {len(_flex_cols)} columns")

    # keep_ids for the full tree (all kingdoms) — used to set reference height
    hit_cols_all = [c for c in df_full.columns
                    if c.startswith("masstFlexibleMatch_") or c.startswith("wd_")]
    if args.min_hits > 0 and hit_cols_all:
        keep_ids_all = {_normalize_ott(str(x))
                        for x in df_full.index[df_full[hit_cols_all].notna().any(axis=1)]}
    else:
        keep_ids_all = {_normalize_ott(str(x)) for x in df_full.index}

    # kingdom-filtered working df
    if args.kingdom:
        df = df_full[df_full["NCBIKingdom"].str.lower() == args.kingdom.lower()].copy()
        print(f"  Filtered to NCBIKingdom={args.kingdom}: {len(df):,} rows")
        keep_ids_kingdom = {_normalize_ott(str(x)) for x in df.index}
    else:
        df = df_full
        keep_ids_kingdom = None

    pairs = _get_pairs(df_full)   # pair list from full dataset so all molecules appear
    print(f"  {len(pairs)} molecule pairs found")

    ordered = _order_from_molecules(pairs, args.molecules)
    if ordered is not None:
        pairs = ordered
        print("  Molecules ordered as listed in the input file")
    else:
        # no input file to take an order from - fall back to grouping molecules
        # that hit the same animals
        print("  Reordering molecules by Animalia MASST similarity …")
        pairs = _reorder_by_animalia(df_full, pairs)

    strip1_hex_map, strip1_rare, strip1_def_hex = _build_strip_colormap(df, args.strip1)
    strip2_hex_map, strip2_rare, strip2_def_hex = _build_strip_colormap(df, args.strip2)
    print(f"  Strip1 ({args.strip1}): {len(strip1_hex_map)} values")
    print(f"  Strip2 ({args.strip2}): {len(strip2_hex_map)} common + {len(strip2_rare)} rare")

    # kingdoms present — used only for legend filtering when strip1 is NCBIKingdom
    if args.strip1 == "NCBIKingdom" and "NCBIKingdom" in df.columns:
        present_kingdoms = {str(v).lower() for v in df["NCBIKingdom"].dropna().unique()}
    else:
        present_kingdoms = None

    # Leaf label map: use the finest available taxonomic name for each OTT.
    # Build from coarse→fine so finer names overwrite coarser ones.
    _FINEST_NAME_COLS = [
        "NCBIKingdom", "NCBIPhylum", "NCBIClass", "NCBIOrder",
        "NCBIFamily", "NCBIGenus", "NCBISpecies",
    ]
    name_map: dict[str, str] = {}
    for _col in _FINEST_NAME_COLS:
        if _col not in df.columns:
            continue
        for _raw_ott, _raw_v in df[_col].items():
            _v = str(_raw_v).strip()
            if _v and _v.lower() not in ("nan", "na", "none", ""):
                name_map[_normalize_ott(str(_raw_ott))] = _v

    # Disambiguate any duplicate labels by appending the OTT ID
    _label_count: dict[str, int] = {}
    for _lbl in name_map.values():
        _label_count[_lbl] = _label_count.get(_lbl, 0) + 1
    _dupes = {_lbl for _lbl, _cnt in _label_count.items() if _cnt > 1}
    if _dupes:
        name_map = {
            ott: (f"{lbl} [{ott}]" if lbl in _dupes else lbl)
            for ott, lbl in name_map.items()
        }

    # ── molecule structure images ──────────────────────────────────────────────
    print("Loading SMILES for molecule images …")
    smiles_map = _load_smiles_map(args.metadata, args.molecules)

    # Deduplicate pairs that share the same SMILES
    pairs, label_overrides = _dedup_pairs_by_smiles(pairs, smiles_map)
    if label_overrides:
        print(f"  Deduplicated {len(label_overrides)} molecule group(s) with identical SMILES:")
        for suf, lbl in label_overrides.items():
            print(f"    {lbl}")

    # ── Auto-size column width and header based on number of pairs ───────────
    n_pairs = len(pairs)
    if args.cell_w == 10 and n_pairs > 0:
        # Target ~600 px total data width; stride = 2*cell_w + COL_GAP(2)
        auto_cw = max(10, min(60, (600 // n_pairs - 2) // 2))
        args.cell_w = max(10, (auto_cw // 5) * 5 or auto_cw)
        print(f"  Auto cell width: {args.cell_w} px  ({n_pairs} pairs)")

    font_sz = max(9, min(18, args.cell_w))
    max_label_chars = max(
        (len(label_overrides.get(suf) or _display(fc or wc or suf))
         for suf, fc, wc in pairs),
        default=10,
    )
    min_hdr = int(max_label_chars * font_sz * 0.62) + 70
    args.header_h = max(150, min_hdr)

    mol_img_w  = args.cell_w * 2
    mol_img_h  = max(10, args.header_h // 3)   # matches img_h in _header_svg
    mol_imgs   = _render_mol_images(smiles_map, pairs, mol_img_w, mol_img_h)

    # ── tree ──────────────────────────────────────────────────────────────────
    print("Parsing tree …")
    tree  = Phylo.read(args.tree, "newick")
    n_raw = len(tree.get_terminals())
    print(f"  {n_raw:,} leaves in raw tree")

    # First prune: retain all hit leaves PLUS (when a kingdom is requested) all
    # kingdom leaves regardless of hits, so the default view shows the full kingdom.
    keep_ids_for_tree = keep_ids_all | keep_ids_kingdom if keep_ids_kingdom else keep_ids_all
    print("  Pruning tree …")
    tree = _prune(tree, keep_ids_for_tree)
    n_all_leaves = len(tree.get_terminals())
    print(f"  {n_all_leaves:,} leaves retained (reference for height)")

    # Second prune: further filter to the requested kingdom
    if keep_ids_kingdom is not None:
        tree = _prune(tree, keep_ids_kingdom)
        print(f"  {len(tree.get_terminals()):,} leaves after kingdom filter")

    print("  Ladderizing …")
    tree = _ladderize(tree)

    print("  Computing layout …")
    leaf_names, x_map, y_map, max_x = _layout(tree)
    n_leaves = len(leaf_names)
    print(f"  {n_leaves:,} leaves · max depth {max_x:.4f}")

    # display_h is based on n_all_leaves so kingdom views fill the same vertical space.
    # Enforce a minimum of tree_w // 3 so that small trees (e.g. a plant-only tree
    # with few leaves) are not rendered as an extremely flat strip — the tree width
    # is fixed at 2000 px but the height would otherwise be only n_leaves * 0.45 px.
    display_h = max(1, round(n_all_leaves * args.row_h))
    min_display_h = args.tree_w // 2
    if display_h < min_display_h:
        display_h = min_display_h
        print(f"  Display height: {display_h} px  (raised to tree_w/2 minimum; {n_all_leaves} ref leaves)")
    else:
        print(f"  Display height: {display_h} px  ({n_all_leaves} ref leaves × {args.row_h} px/leaf)")

    # ── Layout constants ──────────────────────────────────────────────────────
    SPACER_W  = 40
    PHYLO_W   = 120
    COL_GAP   = 4
    STRIP_GAP = 3

    # ── heatmap PNG ───────────────────────────────────────────────────────────
    _png_strip_kwargs = dict(
        strip1_w=args.strip1_w, strip2_w=args.strip2_w,
        strip1_col=args.strip1, strip2_col=args.strip2,
        strip1_hex_map=strip1_hex_map, strip1_rare=strip1_rare, strip1_def_hex=strip1_def_hex,
        strip2_hex_map=strip2_hex_map, strip2_rare=strip2_rare, strip2_def_hex=strip2_def_hex,
        col_gap=COL_GAP, strip_gap=STRIP_GAP,
    )
    print("Building PNG heatmap …")
    png_bytes, img_w, img_h = _build_png(
        leaf_names, df, pairs,
        row_h=args.row_h, cell_w=args.cell_w,
        display_h=display_h,
        **_png_strip_kwargs,
    )
    png_b64 = base64.b64encode(png_bytes).decode()
    print(f"  {img_w}×{img_h} px  ({len(png_bytes) // 1024} KB PNG)")

    # ── leaf metadata + hit data for JS click handler ─────────────────────────
    # All available NCBI taxonomy levels (ordered coarse→fine)
    _NCBI_LEVEL_DEFS = [
        ("NCBIKingdom", "k", "Kingdom"),
        ("NCBIPhylum",  "p", "Phylum"),
        ("NCBIClass",   "c", "Class"),
        ("NCBIOrder",   "o", "Order"),
        ("NCBIFamily",  "f", "Family"),
        ("NCBIGenus",   "g", "Genus"),
        ("NCBISpecies", "s", "Species"),
    ]
    avail_levels  = [(col, sh, lbl) for col, sh, lbl in _NCBI_LEVEL_DEFS if col in df.columns]
    ncbi_dicts_js = {sh: df[col].to_dict() for col, sh, _ in avail_levels}

    # Strip column dictionaries for JS (keyed by OTT)
    _s1_dict_js = df[args.strip1].to_dict() if args.strip1 in df.columns else {}
    _s2_dict_js = df[args.strip2].to_dict() if args.strip2 in df.columns else {}

    leaf_meta = [
        dict(
            {sh: str(ncbi_dicts_js[sh].get(ott, "") or "") for _, sh, _ in avail_levels},
            ott=ott,   # OTT ID needed for LEAF_HIT_COLORS lookup in JS
            s1=str(_s1_dict_js.get(ott, "") or ""),
            s2=str(_s2_dict_js.get(ott, "") or ""),
        )
        for ott in leaf_names
    ]
    leaves_json      = json.dumps(leaf_meta, ensure_ascii=False)
    ncbi_levels_json = json.dumps([[sh, lbl] for _, sh, lbl in avail_levels], ensure_ascii=False)
    pairs_json       = json.dumps(
        [{"l": label_overrides.get(suf) or _display(fc or wc or suf)} for suf, fc, wc in pairs],
        ensure_ascii=False,
    )

    # sparse hit list per leaf: [[pair_idx, masst_level|null, wd_level|null, top_taxa|""], ...]
    ms_dicts_js = {fc: df[fc].to_dict() for _, fc, _ in pairs if fc and fc in df.columns}
    wd_dicts_js = {wc: df[wc].to_dict() for _, _, wc in pairs if wc and wc in df.columns}
    tt_dicts_js = {suf: df[f"masstTopTaxa_{suf}"].to_dict()
                   for suf, fc, _ in pairs if fc and f"masstTopTaxa_{suf}" in df.columns}
    leaf_hits = []
    for ott in leaf_names:
        hits = []
        for j, (suf, fc, wc) in enumerate(pairs):
            mv = _val(ms_dicts_js.get(fc, {}).get(ott)) if fc else None
            wv = _val(wd_dicts_js.get(wc, {}).get(ott)) if wc else None
            if mv or wv:
                _tt_raw = tt_dicts_js.get(suf, {}).get(ott) if suf in tt_dicts_js else None
                tt = ("" if (_tt_raw is None or str(_tt_raw) in ("nan", "None", "NA", "<NA>", ""))
                      else str(_tt_raw))
                hits.append([j, mv, wv, tt])
        leaf_hits.append(hits)
    leaf_hits_json = json.dumps(leaf_hits, ensure_ascii=False)

    # Per-leaf label fill color: orange=both, blue=WD-only,
    # MASST-only → finest match level color from _MASST_HEX
    _MASST_LEVEL_ORDER = ["subspecies", "species", "genus", "family",
                          "order", "class", "phylum", "kingdom"]
    leaf_hit_colors: dict[str, str] = {}
    for ott, hits in zip(leaf_names, leaf_hits):
        has_masst = any(h[1] for h in hits)
        has_wd    = any(h[2] for h in hits)
        if has_masst and has_wd:
            leaf_hit_colors[ott] = "#cc6600"
        elif has_masst:
            best = min(
                (h[1] for h in hits if h[1]),
                key=lambda lv: _MASST_LEVEL_ORDER.index(lv)
                               if lv in _MASST_LEVEL_ORDER else 99,
            )
            leaf_hit_colors[ott] = _MASST_HEX.get(best, "#cc0000")
        elif has_wd:
            leaf_hit_colors[ott] = "#0000cc"

    # popup-sized molecule images (square, separate from header renders)
    print("Rendering popup molecule images …")
    mol_imgs_popup = _render_mol_images(smiles_map, pairs,
                                        width=60, height=60, render_scale=4)
    mol_imgs_json  = json.dumps(
        [mol_imgs_popup.get(suf, "") for suf, _, _ in pairs],
        ensure_ascii=False,
    )

    # ── Label column width ────────────────────────────────────────────────────
    # Estimate space needed for leaf labels at the base font size so the
    # heatmap starts far enough right that labels don't overlap it.
    _row_px_tree = img_h / n_leaves if n_leaves > 0 else 1.0
    _font_sz_tree = max(2.0, min(6.0, _row_px_tree * 0.7))
    _max_label_chars = max((len(v) for v in name_map.values()), default=20) if name_map else 20
    LABEL_COL_W = max(80, int(_max_label_chars * _font_sz_tree * 0.62))

    # ── SVG tree ──────────────────────────────────────────────────────────────
    print("Building tree SVG …")
    tree_svg = _tree_svg(
        tree, x_map, y_map, max_x,
        args.tree_w, n_leaves, img_h,
        name_map=name_map,
        leaf_hit_colors=leaf_hit_colors,
        label_col_w=LABEL_COL_W,
    )

    # ── column headers ────────────────────────────────────────────────────────
    print("Building column headers …")
    header_svg = _header_svg(
        pairs, args.cell_w,
        strip1_w=args.strip1_w, strip2_w=args.strip2_w,
        strip1_label=args.strip1, strip2_label=args.strip2,
        header_h=args.header_h, img_w=img_w,
        mol_imgs=mol_imgs, label_overrides=label_overrides,
        col_gap=COL_GAP, font_sz=font_sz, strip_gap=STRIP_GAP,
    )
    header_svg2 = header_svg

    # ── Second heatmap — leaves clustered by MASST hit profile ───────────────
    print("Computing clustered leaf order …")
    clust_order = _cluster_leaf_order(leaf_names, df, pairs)
    clust_leaf_names = [leaf_names[i] for i in clust_order]
    print(f"  {len(clust_order)} leaves ordered")
    print("Building clustered PNG heatmap …")
    png2_bytes, _, _ = _build_png(
        clust_leaf_names, df, pairs,
        row_h=args.row_h, cell_w=args.cell_w,
        display_h=display_h,
        **_png_strip_kwargs,
    )
    png2_b64 = base64.b64encode(png2_bytes).decode()
    clust_order_json = json.dumps(clust_order)
    print(f"  {len(png2_bytes) // 1024} KB")

    # ── Tree JSON for client-side re-layout ──────────────────────────────────
    print("Building tree JSON for JS …")
    tree_data      = _tree_to_json(tree, x_map, max_x)
    tree_data_json = json.dumps(tree_data, ensure_ascii=False)
    name_arr_json  = json.dumps(
        [name_map.get(ott, "") for ott in leaf_names], ensure_ascii=False
    )
    print(f"  tree JSON: {len(tree_data_json) // 1024} KB")

    # ── Color maps for JS canvas rendering ───────────────────────────────────
    strip1_colors_json = json.dumps(
        {k.lower(): list(_h2rgb(v)) for k, v in strip1_hex_map.items()}
    )
    strip1_def_json    = json.dumps(list(_h2rgb(strip1_def_hex)))
    strip2_colors_json = json.dumps(
        {k.lower(): list(_h2rgb(v)) for k, v in strip2_hex_map.items()}
    )
    strip2_def_json    = json.dumps(list(_h2rgb(strip2_def_hex)))
    strip2_rare_json   = json.dumps(list(strip2_rare))
    strip1_label_json  = json.dumps(args.strip1)
    strip2_label_json  = json.dumps(args.strip2)

    # Precompute color maps for ALL available NCBI levels so the in-HTML dropdowns
    # can switch strips interactively without regenerating the file.
    _sh_by_col = {col: sh for col, sh, _ in avail_levels}
    _all_strip_maps: dict = {}
    for _col, _sh, _lbl in avail_levels:
        _hm, _rare, _def = _build_strip_colormap(df, _col)
        _all_strip_maps[_sh] = {
            "colors":    {k.lower(): list(_h2rgb(v)) for k, v in _hm.items()},
            "hexColors": {k.lower(): v for k, v in _hm.items()},
            "rare":      list(_rare),
            "def":       list(_h2rgb(_def)),
            "defHex":    _def,
            "label":     _col,
        }
    all_strip_maps_json   = json.dumps(_all_strip_maps, ensure_ascii=False)
    strip1_init_sh_json   = json.dumps(_sh_by_col.get(args.strip1, ""))
    strip2_init_sh_json   = json.dumps(_sh_by_col.get(args.strip2, ""))

    masst_colors_json    = json.dumps({k: list(v) for k, v in _MS_RGB.items()})
    wd_colors_json       = json.dumps({k: list(v) for k, v in _WD_RGB.items()})
    leaf_hit_colors_json = json.dumps(leaf_hit_colors, ensure_ascii=False)

    # ── PhyloPic silhouettes ─────────────────────────────────────────────────
    # Allow ~1 silhouette per 48 px of display height (44 px icon + 4 px gap),
    # so the column fills naturally without overcrowding.  Hard-cap at 80.
    _phys_n_max = max(20, img_h // 48)
    _n_max      = min(_phys_n_max, 80)
    print(f"Selecting PhyloPic groups (n_max={_n_max}) …")
    phylo_groups = _select_phylopic_groups(df, leaf_names, leaf_hits, n_max=_n_max)
    print(f"  {len(phylo_groups)} groups selected")
    sz = (PHYLO_W - 4) * 2
    for g in phylo_groups:
        print(f"  Fetching PhyloPic for '{g['lookup_name']}' …")
        img = _fetch_phylopic_b64(g["lookup_name"], size=sz)
        # If the specific lookup (e.g. a species name) has no PhyloPic entry,
        # fall back to the group label (e.g. the class name) which is more
        # likely to have a silhouette.
        if img is None and g["lookup_name"].lower() != g["label"].lower():
            print(f"    → falling back to group label '{g['label']}'")
            img = _fetch_phylopic_b64(g["label"], size=sz)
        g["img"] = img or ""

    # Deduplicate silhouettes — two groups that resolved to the same PhyloPic
    # image get the image cleared on the lower-priority duplicate so the visual
    # is distinct while the label + bracket still appear.
    _seen_imgs: set[str] = set()
    for g in phylo_groups:
        if g["img"]:
            if g["img"] in _seen_imgs:
                print(f"  Duplicate silhouette for '{g['label']}' — showing label only")
                g["img"] = ""
            else:
                _seen_imgs.add(g["img"])
    phylopic_json = json.dumps(
        [{"img": g["img"], "label": g["label"], "level": g.get("level", ""),
          "leaves": g["leaves"]}
         for g in phylo_groups if g["img"]],
        ensure_ascii=False,
    )

    # ── HTML ──────────────────────────────────────────────────────────────────
    print("Writing HTML …")
    html = _HTML.format(
        header_h         = args.header_h,
        tree_w           = args.tree_w,
        tree_and_label_w = args.tree_w + LABEL_COL_W,
        content_w        = args.tree_w + LABEL_COL_W + PHYLO_W + img_w + SPACER_W + img_w,
        label_col_w      = LABEL_COL_W,
        content_h  = args.header_h + img_h,
        spacer_w   = SPACER_W,
        png2_b64   = png2_b64,
        clust_order_json = clust_order_json,
        tree_svg   = tree_svg,
        header_svg = header_svg,
        header_svg2 = header_svg2,
        png_b64    = png_b64,
        img_w      = img_w,
        img_h      = img_h,
        n_leaves   = n_leaves,
        strip1_w   = args.strip1_w,
        strip2_w   = args.strip2_w,
        strip1_label_json = strip1_label_json,
        strip2_label_json = strip2_label_json,
        cell_w     = args.cell_w,
        leaves_json      = leaves_json,
        ncbi_levels_json = ncbi_levels_json,
        pairs_json       = pairs_json,
        leaf_hits_json   = leaf_hits_json,
        mol_imgs_json    = mol_imgs_json,
        tree_data_json   = tree_data_json,
        name_arr_json    = name_arr_json,
        strip1_colors_json = strip1_colors_json,
        strip1_def_json    = strip1_def_json,
        strip2_colors_json = strip2_colors_json,
        strip2_def_json    = strip2_def_json,
        strip2_rare_json   = strip2_rare_json,
        all_strip_maps_json  = all_strip_maps_json,
        strip1_init_sh_json  = strip1_init_sh_json,
        strip2_init_sh_json  = strip2_init_sh_json,
        masst_colors_json    = masst_colors_json,
        wd_colors_json       = wd_colors_json,
        leaf_hit_colors_json = leaf_hit_colors_json,
        phylo_w      = PHYLO_W,
        col_gap      = COL_GAP,
        strip_gap    = STRIP_GAP,
        phylopic_json = phylopic_json,
        legend         = _legend_html(
            strip1_hex_map=strip1_hex_map, strip1_label=args.strip1, strip1_rare=strip1_rare,
            strip2_hex_map=strip2_hex_map, strip2_label=args.strip2,
            strip2_rare=strip2_rare, strip2_def_hex=strip2_def_hex,
            present_kingdoms=present_kingdoms,
        ),
    )
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
