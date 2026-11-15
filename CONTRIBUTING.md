# Contributing

Thanks for taking a look. This is one of my portfolio repositories: it documents a piece of research or engineering
work I did, rebuilt so anyone can run it on synthetic or public data.

## Getting started

```bash
git clone https://github.com/Avinash186200/volatility-regime-ml.git
cd volatility-regime-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

## How the repository is organised

- The maintained code is described in the [README](README.md); method notes are in [`docs/`](docs/README.md).
- [`original/`](original/README.md) is the original research code. Treat it as a read-only record.
- Tests must stay fast and offline: no network, no API keys, no local data files.

## Changes

- Open an issue first for anything larger than a small fix, so we can agree on the approach.
- Keep pull requests focused, with a test for any behaviour change, and update the docs and
  [CHANGELOG](CHANGELOG.md) in the same pull request.
- Never commit data files, credentials or `.env` files. Configuration that needs a secret reads it from an
  environment variable, with an example in `.env.example` where relevant.
- Results in the README must be reproducible from the code in this repository. Do not add real-market performance
  figures unless they come from the leak-free method and are labelled as preliminary.
