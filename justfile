# List all available recipes (this command)
@help:
    printf "Make-like commands to help you manage the project.\n"
    printf "Make sure you install uv.\n\n"
    printf "Usage: just [task]\n\n"
    just --list

# Install project dependencies
@install: (needs "uv")
    needs uv
    uv sync --all-extras

# Format source code and tests
@format: (needs "uv")
    uvx ruff format -- src tests notebooks
    uvx ruff check --select=I --fix -- src tests notebooks

alias fmt := format

@clean: (needs "uv")
    uvx nbstripout -- notebooks

# Run unit tests
@test: (needs "uv")
    uv run pytest --capture=no -- tests

@docstrings: (needs "uv")
    uvx interrogate src tests

# Run unit tests with coverage
@coverage format="text": (needs "uv")
    uv run coverage run \
      --data-file=.coverage.unit \
      --module pytest -- tests

    uv run coverage report \
      --format={{ format }} \
      --data-file=.coverage.unit

# Assert a command is available
[private]
needs +commands:
    #!/usr/bin/env zsh
    set -euo pipefail
    for cmd in "$@"; do
      if ! command -v $cmd &> /dev/null; then
        echo "$cmd binary not found. Did you forget to install it?"
        exit 1
      fi
    done
