PATH_TO_SQLITE = "/home/yasin/projects/Structure_MASST_App/database/masst_records.sqlite"
MASSTRECORDS_ENDPOINT =  "https://masst-records.gnps2.org/masst_records"
MASSTRECORDS_TIMEOUT = 1000
MASSTRECORDS_ROWLIMIT = 1000000

# Set to True in Docker (workers preload ReDU table). False for local Streamlit runs.
# Docker overrides this via USE_CELERY=true environment variable in docker-compose.
USE_CELERY = False