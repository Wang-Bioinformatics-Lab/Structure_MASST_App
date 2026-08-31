import os

# Path to the local masst_records SQLite copy.
# Default is the in-container path (docker-compose mounts ./database/ -> /app/database:ro).
# Local (non-Docker) runs keep the DB elsewhere for storage reasons, so override with:
#   export PATH_TO_SQLITE=/your/path/masst_records.sqlite
# If the file does not exist, _get_fetcher() silently falls back to MASSTRECORDS_ENDPOINT.
PATH_TO_SQLITE = os.environ.get("PATH_TO_SQLITE", "/app/database/masst_records.sqlite")

MASSTRECORDS_ENDPOINT =  "https://masst-records.gnps2.org/masst_records"
MASSTRECORDS_TIMEOUT = 1000
MASSTRECORDS_ROWLIMIT = 1000000

# Set to True in Docker (workers preload ReDU table). False for local Streamlit runs.
# Docker overrides this via USE_CELERY=true environment variable in docker-compose.
USE_CELERY = False
