server-compose-build-nocache:
	docker-compose --compatibility build --no-cache

server-compose-interactive:
	docker-compose --compatibility build
	docker-compose --compatibility -f docker-compose.yml -f docker-compose-dev.yml up

server-compose:
	docker-compose --compatibility build
	docker-compose --compatibility -f docker-compose.yml -f docker-compose-dev.yml up -d

server-compose-production:
	docker-compose --compatibility build
	docker-compose --compatibility -f docker-compose.yml up -d

attach:
	docker exec -i -t structure-masst-server /bin/bash

# ---- local development (outside docker) ----
KETCHER_VERSION ?= 3.18.0

# Download the Ketcher structure editor into static/ketcher (git-ignored)
ketcher:
	mkdir -p static/ketcher
	wget -q https://github.com/epam/ketcher/releases/download/v$(KETCHER_VERSION)/ketcher-standalone-$(KETCHER_VERSION).zip -O static/ketcher.zip
	unzip -q -o static/ketcher.zip -d static/ketcher

build:
	cargo build --release

test:
	cargo test

# Run locally; needs the StructureMASST_env python (RDKit etc.) and nextflow on PATH
run-local: build
	STRUCTUREMASST_PYTHON=$$(which python) PORT=5000 ./target/release/structure-masst
