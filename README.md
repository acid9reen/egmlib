# EGM Lib

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

---

EGM library for other projects


---

## Developer corner

### Pre-commit hooks

There are some pre-commit hooks available, to install them run
```bash
pre-commit install
```

### Task runner

There are some make recipes available:
* lint (use ruff and mypy to lint code)
* deps (generate and install (hopefully in venv) dependencies from
    pyproject.toml, install local package in editable mode)
* test (run all pytest tests)
