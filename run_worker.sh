#!/bin/bash

source activate StructureMASST_env
celery -A tasks worker --loglevel=INFO --concurrency=8