## Structure MASST

### Architecture

The web app is a Rust server (`src/`, axum) that serves the browser UI
(`static/`) and a JSON API. Chemistry and data work stays in Python: the server
starts one long-lived compute worker (`bin/compute_worker.py`) that calls the
existing `bin/` and `external/` modules (RDKit, masst_records, FASST, LifeMASST,
GeoMASST) over a JSON-lines pipe. Worker processes are forked from a forkserver
that preloads the ReDU table and the class labels, like the former Celery
worker did, so Redis and the separate worker container are no longer needed.

| Former (Streamlit)                              | Now                                                   |
|-------------------------------------------------|-------------------------------------------------------|
| `structureMASST.py`, `pages/*`                  | `static/*.html`, `static/app.js`, `static/pages.js`   |
| `bin/streamlit_search_ui.py`, `streamlit_prepare.py` | `static/search.js` + `src/workflows.rs`          |
| `st.session_state`                              | `src/session.rs` (`sessionoutput/<sid>/state.json`)   |
| Celery + Redis (`tasks.py`, `run_worker.sh`)    | `bin/compute_worker.py`, `bin/compute_ops.py`, `src/worker.rs` |
| spinners blocking the page                      | background jobs with live logs (`src/jobs.rs`)        |
| `streamlit_ketcher`                             | Ketcher standalone in an iframe (`static/ketcher/`)   |

Pages: `/` (StructureMASST), `/domainmasst`, `/geomasst`, `/lifemasst`.

Environment: `PORT` (5000), `STRUCTUREMASST_PYTHON` / `STRUCTUREMASST_NEXTFLOW`
(the conda env's binaries), `SESSION_DIR`, `WORKER_PROCESSES` (4),
`PRELOAD_REDU` (1), `SMARTS_API_KEY` (or `keys.env`), plus `PATH_TO_SQLITE`,
`LIFEMASST_SCRATCH`, `NF_WORK_DIR`, `NF_CONDA_CACHE` below.

Development: `make ketcher` (download the structure editor), `make test`,
`make run-local` (with `StructureMASST_env` active).

### Reference data for LifeMASST and GeoMASST

Both tools need reference data that a MASST search does not produce. **You no
longer need to prepare it by hand**: the first time somebody runs LifeMASST or
GeoMASST from the app, the missing pieces are fetched and built, with the log on
screen, before the run starts. Every run after that reuses them.

What that involves differs between the two:

- **GeoMASST** downloads nothing. Its basemap assets are committed to its
  repository, so preparing it only verifies them.
- **LifeMASST** already ships every tree it offers, so nothing downloads those
  either. It does fetch the ReDU, Wikidata lineage, Empress community and plant
  distribution tables - about 160 MB in total, once.

FASSTrecords is deliberately not covered: that database is installed by hand.

To do it up front instead - a headless install, or a docker image build:

```bash
python bin/dependency_prep.py --status      # what this checkout has and needs
python bin/dependency_prep.py lifemasst
python bin/dependency_prep.py geomasst
```

Both pages also carry a "reference data" panel showing the same thing, with a
button to build it without starting a search.

Two environment variables matter for a local (non-docker) checkout:

| variable | what for |
|---|---|
| `PATH_TO_SQLITE` | where `masst_records.sqlite` actually is |
| `LIFEMASST_SCRATCH` | somewhere off the repo volume for bulky intermediates (the WCVP tables are ~500 MB of CSV, read once) |
| `NF_WORK_DIR`, `NF_CONDA_CACHE` | move the Nextflow work directory and conda cache (several GB) off a small root volume |

`external/LifeMASST/data/README.md` documents exactly what ships in git versus
what is fetched, and why.

### Updating ReDU in the FASSTrecords database

`bin/update_redu_table.py` replaces `redu_table` in the FASSTrecords database
(`masst_records.sqlite`) with a fresh ReDU dump and links each ReDU row to the
file FASSTrecords indexed: by exact path, and for converted copies
(`ccms_peak/`, `gnps2_converted/`) of files ReDU lists under their original path
(`peak/`, `raw/`, `updates/`) with the same rule FASST mode uses. It is a manual
tool; neither the app nor its workflows run it.

```bash
curl -L -o redu_dump.tsv https://redu.gnps2.org/dump      # ~1.7 GB
python bin/update_redu_table.py --dump redu_dump.tsv --keep-unmatched --drop-empty --dry-run
python bin/update_redu_table.py --dump redu_dump.tsv --keep-unmatched --drop-empty
```

- The database path comes from `config.py` (`PATH_TO_SQLITE`, which the
  environment variable of that name overrides), or `--db`.
- `--keep-unmatched` keeps the ReDU rows of files FASSTrecords has not indexed
  (FASST mode needs them); `--dry-run` reports what would change and writes nothing.
- The previous table stays as `redu_table_bak_<date>`; renaming it back to
  `redu_table` undoes the update.
- With docker-compose the database is mounted read-only into the containers: run
  the script on the host against `database/masst_records.sqlite`, then restart the
  containers (the server preloads the ReDU table, and datasette opens the file as
  immutable).

### Links: session and search settings in the address

The address bar always describes the page, so it can be bookmarked or shared:

```
https://<host>/?session=<id>&smiles=CN1C%3DNC2C1C(%3DO)N(C(%3DO)N2C)C&searchtype=substructure&mode=FASST&cosine=0.8
```

- `session` reopens that session - its results and files - for as long as its
  folder in `sessionoutput/` exists. A session that is gone is ignored: a new one is
  started and the search settings from the link are filled in.
- Search settings: `name`, `smiles`, `class`, `usi` (structure input); `searchtype`,
  `tanimoto`, `formula`, `elements` (library search); `mode`, `cosine`, `peaks`,
  `rank`, `prectol`, `fragtol`, `modsearch`, `elimination`, `addition`, `analogues`,
  `modformula`, `modmass`, `modcondition` (raw data search); `source=search`
  (GeoMASST). A link fills the form in; it never starts a search by itself.
- Whoever has a link with a session in it can see and change that session.
- **↺ New session** in the menu starts an empty session (results and form); the
  previous session stays available under its address, which the new page links to.

### Searching from the LifeMASST and GeoMASST pages

Neither page requires a StructureMASST run first: both offer the StructureMASST
search controls directly — name, SMILES/SMARTS, class label or USI, a batch CSV,
and the same advanced options — and run the same search behind them. Those
controls live in `static/search.js` (keyed by a page prefix) and the search
behind them in `src/workflows.rs`, so all three pages share one implementation
instead of each growing its own subset of the options.
