## Structure MASST

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

### Searching from the LifeMASST and GeoMASST pages

Neither page requires a StructureMASST run first: both offer the StructureMASST
search controls directly — name, SMILES/SMARTS, class label or USI, a batch CSV,
and the same advanced options — and run the same search behind them. Those
widgets live in `bin/streamlit_search_ui.py` and are keyed by a caller-supplied
prefix, so the two pages share one implementation instead of each growing its
own subset of the options.
