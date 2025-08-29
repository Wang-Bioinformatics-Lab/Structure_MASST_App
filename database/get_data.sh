#!/bin/bash
set -euo pipefail

# Download redu.tsv
curl -L -o redu.tsv https://redu.gnps2.org/dump

# Convert to Feather
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("redu.tsv", sep="\t", dtype=str)
df.to_feather("redu.feather")
EOF
rm redu.tsv
echo "Conversion complete: redu.feather"
