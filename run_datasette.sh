#!/bin/bash

datasette -p 5234 -h 0.0.0.0 \
--setting max_returned_rows 1000000 \
--setting sql_time_limit_ms 60000 \
--setting max_csv_mb 0 --immutable /app/database/masst_records.sqlite