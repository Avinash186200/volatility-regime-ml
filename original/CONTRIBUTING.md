# Contributing

## Branch workflow

- Don't work directly on `main`. Create a branch named `intern/<your-name>` (e.g. `intern/your-name`)
  and do all your work there.
- Commit early and often — small commits are easier to review than one giant one at the end.
- Open a pull request into `main` whenever you have something worth a look, even if it's not
  finished. A draft PR is fine.
- Don't merge your own PR.

## Getting set up

1. Clone the repo and create your branch.
2. `pip install -r requirements.txt` inside a virtual environment.
3. Read `docs/project_brief.md` for the full project description.
4. Look at `src/models/baseline_har.py` before adding a new model — every model should share the
   same `fit` / `predict` interface in `src/models/base.py` so comparisons stay fair.

## Ground rules

- **Tune nothing on the test set.** The single most common way this kind of project quietly fails.
- **Same data, same split, same evaluation, for every model.** If one model gets an advantage in
  setup, the comparison between models means nothing.
- **A model that doesn't beat the HAR-RV baseline is a real, useful result** — say so plainly rather
  than tuning until it does.
- If you're not sure whether something counts as leakage, ask.
