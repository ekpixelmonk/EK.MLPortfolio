# CLAUDE.md

Personal ML portfolio: ASP.NET Core MVC (.NET 10) serving ONNX models trained in Python (uv). Read [SPEC.md](SPEC.md) for requirements and [docs/ADR.md](docs/ADR.md) for decisions and the per-phase log.

## Commands (PowerShell, Windows)
```powershell
dotnet build MLPortfolio.slnx
dotnet test MLPortfolio.slnx
dotnet run --project src/MLPortfolio.Web --launch-profile http   # http://localhost:5215
```
Training (per project, Python 3.12 via uv): `cd ml/<slug>; uv sync; uv run python train.py`. If uv fails with `UnknownIssuer`, set `UV_NATIVE_TLS=1` (system certificates; verification stays on).
SDK is pinned in `global.json` (10.0.x). Do not add `Microsoft.AspNetCore.Mvc.Testing` without `--version 10.0.x` (latest is 11.x and breaks net10.0).

## Layout
- `src/MLPortfolio.Web`: the site. `Services/ProjectCatalog` loads and validates `App_Data/projects.json` at startup; `Controllers/HomeController` renders the homepage; views in `Views/`; theme tokens in `wwwroot/css/site.css`; `wwwroot/js/site.js` (theme toggle) and `home.js` (filter/search).
- Project page (`/projects/{slug}`): `Controllers/ProjectsController` + `Views/Projects/Details.cshtml`; form partial `Views/Shared/_TryModel.cshtml`; `wwwroot/js/project.js` handles presets and the `/predict/{slug}` call (the endpoint itself is Phase 3). Services: `ProjectContentStore` (Markdown → HTML, raw HTML disabled), `SchemaStore` (loads/validates `schema.json`, cached; invalid → no form + warning log). Unknown URLs go through `UseStatusCodePagesWithReExecute` to a styled 404.
- Prediction: `Controllers/PredictController` (`POST /predict/{slug}`) → `Services/InputValidator` (schema validation) → `IModelRunner`/`OnnxModelRunner` (cached `InferenceSession` per slug). ONNX inputs: one per schema input, named after it, shape `[1,1]` (float for numbers, string for selects). Rate limit policy `predict` in `Program.cs`.
- Parity: `ml/<slug>/train.py` writes `tests/fixtures/<slug>.json` (scikit-learn answers); `PredictTests` checks the .NET runner against it. Re-run `train.py` after changing a model, then `dotnet test`.
- `src/MLPortfolio.Web/App_Data`: `projects.json`, `content/<slug>.md`, `models/<slug>/{model.onnx, schema.json}`.
- `ml/<slug>/`: self-contained uv projects (`train.py`, `notebook.ipynb`, README, `uv.lock`). `train.py` is the single source of truth for the schema, ONNX export, charts and fixtures. Never import across folders or into the web app.
- Real-data models: `ml/linear-regression` (Reliance and Nifty 50) and `ml/random-forest` (adds India VIX) download via `yfinance` into git-ignored `ml/<slug>/data/` (`train.py --refresh` re-downloads; on Windows it builds a CA bundle from the system store). Each has its own copy of the download and `prepare()` audit code (ADR-025, ADR-026); keep the form ranges in `INPUTS` and the cleaning in step. The random forest's ONNX graph appends `exp()` and a multiply by the 60-day volatility input (the forest predicts a log ratio).
- `ml/nse-regime` (flagship, in progress): `data.py` (Nifty 200 and India VIX download into git-ignored `data/`) `labels.py` (four causal trend x volatility regimes, ADR-027) `features.py` (audit and seven features, ADR-028) and `train.py` (walk-forward evaluation, balanced logistic regression to ONNX, charts, fixtures; `--report` prints the page's training tables; ADR-029) and `eda.py` (12 EDA charts embedded inline in the write-up, ADR-030). It is live with `inferenceMode: "snapshot"` (ADR-031): no form and no `/predict`; `train.py` writes `models/nse-regime/snapshot.json` (the latest day's seven feature values) and `Services/RegimeReadingService` runs the ONNX model on it for the callout on the page. Re-run `train.py` (and `dotnet test`) after refreshing the data so the reading and the test fixture stay current.
- Adding a Zoo model: copy the closest `ml/<slug>/train.py` (classifiers: `decision-tree`, `svm`, `knn`, `naive-bayes`; regressors: `random-forest`, `linear-regression`), keep the ONNX input convention, run it, then add `projects.json` entry (chart captions are printed by `train.py`), `content/<slug>.md`, an executed notebook, a closing `## Conclusion` (with what would improve it) in the write-up, and the slug to `ClassificationSlugs`/`RegressionSlugs` in `PredictTests`. Classifier labels are chosen by arg-max of the probabilities (not the ONNX label output). Keep `scikit-learn<1.9` while an SVC uses `probability=True`.
- `tests/MLPortfolio.Tests`: xUnit unit tests plus `WebApplicationFactory<Program>` integration tests.

## Conventions
- Projects are metadata-driven: adding one is data (folder + `projects.json` entry), not new controller/view code.
- Homepage tag pills come from `IProjectCatalog.SharedTags` (tags on 2+ projects); one-off tags are searchable only.
- Flagship projects need a `domain` (closed set in `Models/ProjectMeta.cs` → `ProjectDomains`); domain pills on the homepage appear only for domains in use.
- Keep a project's catalog `status` as `coming-soon` until it is really built. `inferenceMode` is `onnx` (form and `/predict`), `snapshot` (no form; a stated reading from `snapshot.json`), `lookup` (reserved, not implemented) or `none`.
- Site owner info and social links live in `appsettings.json` (`Site`: `LinkedInUrl`, `TwitterUrl`, `MediumUrl`, `GitHubUrl`, `Email`); never invent URLs. They render as icon-only links via `Views/Shared/_SocialLinks.cshtml` in both the navbar and the footer, and only when set. Project code/notebook links are derived from `Site:RepoUrl` by `SiteOptions.CodeUrlFor/NotebookUrlFor` (live projects only; a project's `repoUrl` overrides).
- Colours and spacing go through the CSS tokens in `site.css`; keep light and dark in sync and WCAG AA.
- Razor HTML-encodes characters like `·` and `'` in some contexts; assert around them in tests. To test a different catalog, replace `IProjectCatalog` via `WebApplicationFactory.WithWebHostBuilder` (see `ProjectPageTests`).
- Only slugs found in the catalog may be used to build file paths (stores assume this).

## Workflow
- Don't write implementation plans; build directly once scope is agreed.
- After each implementation, update SPEC.md, CLAUDE.md, README.md and docs/ADR.md (new ADR for each real decision, a Phase Log entry per phase).
- Don't commit unless asked.
