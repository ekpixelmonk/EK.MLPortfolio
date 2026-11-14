# EK.MLPortfolio

Personal ML portfolio: flagship ML projects and a Model Zoo, served by ASP.NET Core MVC (.NET 10) with ONNX Runtime inference. See [SPEC.md](SPEC.md).

## Prerequisites
- .NET SDK 10 (pinned in `global.json`)
- [uv](https://docs.astral.sh/uv/) for the Python training projects under `ml/`

## Status
Phases 1–3 complete, Phase 5 under way: project catalog, themed homepage (light/dark) with domain/tag filters and search, generic project pages (Markdown write-up, chart gallery, form generated from `schema.json`, styled 404), and seven live Model Zoo projects (linear regression and random forest train on real Reliance prices from Yahoo Finance; the rest use synthetic data; logistic regression, linear regression, decision tree, random forest, SVM, KNN, naive Bayes) served in-process through ONNX Runtime at `POST /predict/{slug}`. The NSE regime classifier is a "Coming soon" placeholder. See the phase table in [SPEC.md](SPEC.md) and the log in [docs/ADR.md](docs/ADR.md).

## Configure
Owner name, tagline and social links are in `src/MLPortfolio.Web/appsettings.json` under `Site`. GitHub/LinkedIn links appear only once their URLs are set. `Site:RepoUrl` (and `RepoBranch`) is the monorepo URL: each live project shows "View the code on GitHub" under its form, linking to `ml/<slug>`; push the repo for the links to resolve. Projects are listed in `src/MLPortfolio.Web/App_Data/projects.json`.

## Train a model (Python)
Each project under `ml/<slug>/` is a self-contained uv project (Python 3.12). Training writes the ONNX model, `schema.json`, charts and the .NET parity fixtures into the web app:

```powershell
cd ml/logistic-regression
uv sync
uv run python train.py
```

If `uv sync` fails with a certificate error (for example behind a corporate proxy), set `$env:UV_NATIVE_TLS = "1"` to use the system certificate store. Each project's README has the details.

## Run locally
```powershell
dotnet run --project src/MLPortfolio.Web
```

## Test
```powershell
dotnet test MLPortfolio.slnx
```
