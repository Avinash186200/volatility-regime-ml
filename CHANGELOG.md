# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-11-15

### Changed
- Docs: training-grade README and walkthrough. Added what problem this solves, what you will learn, key concepts,
  a mermaid diagram of the pipeline, an accurate repo layout, and a clearer synthetic-data label on the results.

### Fixed
- README quick start: says what the `--csv` file must contain (it is not shipped), that `--ticker` needs network
  access, and where the `results/` files go.

## [1.1.0] - 2026-11-15

### Added
- `original/`: the original research code, sanitised (names, credentials, data and notebook outputs removed).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.

## [1.0.0] - 2026-11-14

### Added
- The maintained code: EWMA, GARCH(1,1), HAR-RV, log-HAR, a regime classifier with a regime-conditional HAR and two
  ensembles, a purged rolling-origin walk-forward, QLIKE/MSE with Diebold-Mariano tests, documentation, tests and CI.

## [0.1.0] - 2026-11-01

### Added
- First public release: the case-study README and licence.
