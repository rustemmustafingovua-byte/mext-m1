# MEXT-M1 — one CLI for the three predictive layers.
#
#   make build     release binaries
#   make run       start the daemon with a 2 GB cap and demo activity
#   make test      cargo test + pytest + AI selftest
#   make bench     page-cache vs compressed-pool benchmark
#   make doctor    environment diagnostics

RAM         ?= 2G
PORT        ?= 8765
WORKSPACE   ?= $(CURDIR)
CARGO       ?= cargo
MANIFEST    := core-daemon/Cargo.toml
RELEASE_BIN := core-daemon/target/release/mext
DEBUG_BIN   := core-daemon/target/debug/mext
PY          := .venv/bin/python

.DEFAULT_GOAL := help

.PHONY: help build run run-ai dev dev-once review test test-rust test-py selftest ci e2e e2e-ai bench doctor fmt fmt-check clippy clean install-ai service-install service-uninstall service-status

help:
	@echo "MEXT-M1"
	@echo "  make build      cargo build --release"
	@echo "  make run        start with cap=$(RAM) on port $(PORT) + --demo"
	@echo "  make run-ai     same, but with the MLX/RAG engine and a real model id"
	@echo "  make dev        watch loop: rebuild + restart debug daemon on edits"
	@echo "  make dev-once   debug build only"
	@echo "  make review     release build + cargo test + pytest + node check + selftest"
	@echo "  make ci         same checks as GitHub Actions (no daemon)"
	@echo "  make test       cargo test, pytest, AI selftest"
	@echo "  make e2e        boot a real daemon and verify CLI+HTTP end to end"
	@echo "  make e2e-ai     same, with MLX neural loader and LanceDB RAG"
	@echo "  make bench      RAM/latency benchmark"
	@echo "  make doctor     environment checks"
	@echo "  make service-install   install a launchd user service (see --dry-run)"
	@echo "  make service-status    launchctl state"
	@echo "  make service-uninstall remove the service"
	@echo "  make clean      remove build artefacts"
	@echo
	@echo "  variables: RAM=$(RAM) PORT=$(PORT) WORKSPACE=$(WORKSPACE)"

build:
	$(CARGO) build --release --manifest-path $(MANIFEST)

run: build
	$(RELEASE_BIN) start --workspace "$(WORKSPACE)" --ram-limit $(RAM) --port $(PORT) --demo

run-ai: build
	$(RELEASE_BIN) start --workspace "$(WORKSPACE)" --ram-limit $(RAM) --port $(PORT) \
		--model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

dev:
	bash scripts/dev.sh --workspace "$(WORKSPACE)" --ram-limit $(RAM) --port $(PORT)

dev-once:
	bash scripts/dev.sh --once

review:
	bash scripts/dev.sh --review

ci: fmt-check test-rust test-py
	node --check dashboard/app.js
	$(PY) -m mext_ai selftest

fmt-check:
	$(CARGO) fmt --manifest-path $(MANIFEST) --check
	$(PY) -m compileall -q ai-engine/mext_ai

service-install: build
	bash scripts/install-service.sh --workspace "$(WORKSPACE)" --ram-limit $(RAM) --port $(PORT)

service-status:
	bash scripts/install-service.sh --status

service-uninstall:
	bash scripts/install-service.sh --uninstall

install-ai:
	$(PY) -m pip install -e "ai-engine[dev,pdf]"

test: test-rust test-py selftest

test-rust:
	$(CARGO) test --manifest-path $(MANIFEST)

test-py:
	$(PY) -m pytest ai-engine -q
	$(PY) -m compileall -q ai-engine/mext_ai

selftest:
	$(PY) -m mext_ai selftest

e2e: build
	bash scripts/e2e.sh

e2e-ai: build
	bash scripts/e2e.sh --with-ai

bench: build
	bash scripts/bench.sh "$(WORKSPACE)"

doctor: build
	$(RELEASE_BIN) doctor

fmt:
	$(CARGO) fmt --manifest-path $(MANIFEST)
	$(PY) -m compileall -q ai-engine/mext_ai

clippy:
	$(CARGO) clippy --manifest-path $(MANIFEST) --all-targets -- -D warnings

clean:
	$(CARGO) clean --manifest-path $(MANIFEST)
	rm -rf ai-engine/build ai-engine/*.egg-info ai-engine/.pytest_cache
	find ai-engine -name __pycache__ -type d -prune -exec rm -rf {} +
