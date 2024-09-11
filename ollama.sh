#!/usr/bin/env bash
#
# Run ollama server with gpu support (metal) on macos.

set -euo pipefail

export OLLAMA_KEEP_ALIVE=10m
# default: ~/.ollama/models
export OLLAMA_MODELS=./docker/ollama/data/models
# default: false
export OLLAMA_NOHISTORY=true

function check() {
  command -v ollama >/dev/null 2>&1 || { echo >&2 "Command 'ollama' not found. Try 'brew install ollama'."; exit 1; }
}

function run_ollama() {
  ollama serve
}

check
run_ollama
