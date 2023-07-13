# Contributing

We want to make contributing to this project as easy and transparent as possible.

## Issues

Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue. The recommended issue format is:

---

#### To Reproduce

`How to reproduce the issue.`

#### Expected behavior

`Expected output.`

#### Environment

`Your environment.`

---

## Developer environment

Test project:

```bash
pytest .
```

Lint project:
Linting is required for PRs. Lint via ruff or use the provided pre commit hooks.

### Precommit hooks

Hooks can be installed from .pre-commit-config.yaml. For example:

1.  `$ pip install pre-commit`
2.  `$ pre-commit install`

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've changed APIs, update the documentation.
3. Ensure the test suite passes (`pytest`). These are also required for PRs.
4. Make sure your code lints (`ruff`). This is also required for PRs.

## Coding Style

We use
[![Code style: ruff](https://github.com/astral-sh/ruff)](https://github.com/astral-sh/ruff)
