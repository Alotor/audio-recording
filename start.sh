#!/usr/bin/env bash
set -euo pipefail

uv run --extra cuda audio-capture record
