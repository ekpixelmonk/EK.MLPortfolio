# ML Portfolio — Specification

**Owner:** Kartik · **Stack:** ASP.NET Core MVC (.NET 10) + Python (uv) for training · **Status:** Draft v1 · **Platform:** Windows 11 (PowerShell; no Ubuntu/bash assumptions)

---

## 1. Purpose

A personal ML portfolio website that showcases:

1. **Flagship ML projects**: in-depth showcase projects, currently finance/stock-market focused (e.g. NSE regime classifier) but open to any domain.
2. **Model Zoo**: one example project per popular ML model (synthetic or real data).

Each project is a card on the homepage. Clicking a card opens a project page with a summary, visualizations exported from the notebook, an input form that returns live model output, and a link to the GitHub repo containing notebooks/code.

## 2. Goals and Non-Goals

**Goals**
- Professional, fast, responsive design.
- Adding a new project requires only data/config (folder + JSON entry), no new controller or view code.
- Models run inside the MVC app via ONNX Runtime (no separate Python service by default).
- Every project is reproducible from its own repo.

**Non-Goals (v1)**
- User accounts, login, or saved predictions.
- Live market data feeds or trading functionality.
- Training models in the browser or on the server.
- Serving large transformer/LLM models.

## 3. Architecture Overview

- **Training side (Python):** each project lives in `ml/<slug>/` as a uv project. A notebook and `train.py` produce charts, `model.onnx`, and `schema.json`.
- **Serving side (.NET):** the MVC app loads `projects.json`, renders cards and pages, and runs inference through `OnnxModelRunner`.
- **Contract between the two:** `model.onnx` + `schema.json` (+ optional `lookup.json` for precomputed projects). Parity tests verify .NET output matches Python output.

```
MLPortfolio/
├── global.json             pins .NET SDK 10
├── MLPortfolio.slnx        (.NET 10 default solution format)
├── src/MLPortfolio.Web/
│   ├── Controllers/        HomeController, ProjectsController, PredictController
│   ├── Models/             ProjectMeta, FormSchema, PredictRequest/Response
│   ├── Services/           ProjectCatalog, IModelRunner, OnnxModelRunner, MarkdownRenderer
│   ├── Views/              Home/, Projects/, Shared/
│   ├── wwwroot/
│   │   ├── css/, js/, lib/
│   │   └── projects/<slug>/charts/
│   └── App_Data/
│       ├── projects.json
│       ├── content/<slug>.md
│       └── models/<slug>/{model.onnx, schema.json, lookup.json?}
├── ml/<slug>/              notebook, train.py, export_onnx.py, pyproject.toml, README.md
└── tests/MLPortfolio.Tests/
```

### 3.1 Repository strategy: monorepo now, split later

All code lives in this single repo initially. To keep any project extractable into its own repo later:

1. Each `ml/<slug>/` is fully self-contained (own `pyproject.toml`, `uv.lock`, notebook, README). No imports or relative paths across project folders or into the web app.
2. Shared assets (e.g. chart styling) are copied into each project or published as a small package, never imported across folders.
3. The trained `model.onnx` and `schema.json` are committed under `src/MLPortfolio.Web/App_Data/models/<slug>/`, so the site works regardless of where the training code lives.
4. Each project's README "how to run" must work from a clean clone of its own folder.
5. `repoUrl` and `notebookUrl` use deep links to the project folder; on a split, only these URLs change.

## 4. Functional Requirements

### 4.1 Homepage (`/`)
- Hero section: name, one-line positioning, links (GitHub, LinkedIn, Twitter).
- Two sections: **Flagship ML** and **Model Zoo**.
- Responsive card grid. Each card shows: title, one-line summary, tags, status badge (Live / Notebook only / Coming soon), thumbnail chart.
- Tag filter chips and text search (client-side).
- Card click navigates to `/projects/{slug}` (404 until Phase 2 builds the project page).
- Owner name, tagline, description and GitHub/LinkedIn URLs come from the `Site` section of `appsettings.json`; a social link is rendered only when its URL is set. Twitter is not included yet.
- Flagship projects have a **domain** (`finance`, `cybersecurity`; add more in `ProjectDomains`). It is shown as a label on the card.
- Filters: one pill active at a time (**All**, then a pill per domain *in use*, then tag pills) plus a text search over title, summary, domain and tags. Only domains used by a project get a pill, so Cybersecurity appears once a cybersecurity project exists. Search;  a section hides when it has no visible cards, and an empty state shows when nothing matches.
- Catalog entries stay `coming-soon` until their project is actually built, so a "Live" badge never points at a missing page.

### 4.2 Project page (`/projects/{slug}`)
Sections in order:
1. **Header:** title, category, tags, GitHub button.
2. **Summary:** problem, data, approach, results (rendered from `App_Data/content/<slug>.md` with Markdig).
3. **Visualizations:** gallery of charts from `wwwroot/projects/<slug>/charts/` (PNG/SVG, or Plotly JSON for interactive charts). Captions come from `projects.json`.
4. **Try the model:** form generated from `schema.json`; result panel shows prediction, probabilities/confidence, and a short interpretation string.
5. **Links:** GitHub repo, notebook (nbviewer/GitHub), related projects.
6. **Disclaimer** (projects with `"disclaimer": "finance"` in `projects.json`, independent of category): "Educational purposes only. Not investment advice."

Unknown slug (and any unknown URL) returns a styled 404 with the real 404 status code.

Implementation notes (Phase 2):
- The Markdown write-up is rendered with Markdig with raw HTML disabled; a missing write-up shows a short placeholder line.
- The gallery currently supports image files (PNG/SVG); Plotly JSON charts are deferred. The section is hidden when a project lists no charts.
- The form is shown only when `inferenceMode` is not `none` and a valid `schema.json` exists. An invalid or missing schema never breaks the page; it just hides the form (and logs a warning). While a project's status is not `live`, the form renders as a disabled preview with a notice.
- "Related projects" are other projects in the same domain or sharing tags, best matches first (max 3).
- Fields: `number` inputs (min, max, optional `step`) and `select` inputs; schema values and presets are validated when the schema loads.

### 4.3 Prediction endpoint (`POST /predict/{slug}`)
- Request: JSON object of field name → value.
- Validation against `schema.json` (type, min, max, allowed options, required). Invalid input returns 400 with per-field errors.
- Response: `{ prediction, label?, probabilities?, explanation?, modelVersion }`, where `probabilities` is an object mapping each label to a value between 0 and 1. A 400 body is `{ "errors": { "<field>": "message" } }`. The page script (`wwwroot/js/project.js`) already expects these shapes.
- Projects with `inferenceMode: "lookup"` resolve against `lookup.json` (precomputed outputs) instead of ONNX. *(Not implemented yet; arrives with the first project that needs it.)*

Implementation notes (Phase 3):
- Only projects that are `live` with `inferenceMode: "onnx"` and a valid schema are served. Anything else returns **404** with a JSON body `{ "error": "..." }` (not the HTML 404 page).
- **ONNX input convention** (shared with the Python training code): one ONNX input per schema input, named after it, shape `[1,1]`: `float` for `number`, `string` for `select`. Classifiers output a label and a probability tensor (ZipMap disabled); regressors output one float.
- For classification, `prediction` is the class index and `label` the class name. For regression, `prediction` is the value (4 decimals) and `label` is a display string such as `"+3.19%"` using the schema's `outputs.unit`.
- Unknown fields are rejected; missing required fields are errors; optional fields fall back to their schema default.
- A missing or broken model returns **503** `{ "error": "..." }` with no internal detail. Request bodies are limited to 16 KB and `/predict` is rate-limited per client IP (default 30 per minute, `RateLimiting:PredictPermitsPerMinute`); over the limit returns **429**.
- Logging records slug, outcome and latency only, never input values.
- Projects with `inferenceMode: "none"` hide the form.

### 4.4 Metadata-driven configuration

**`projects.json` entry**
```json
{
  "slug": "nse-regime",
  "title": "NSE Market Regime Classifier",
  "category": "flagship",
  "domain": "finance",
  "disclaimer": "finance",
  "summary": "Classifies NIFTY market regimes using trend and volatility features.",
  "tags": ["classification", "time series"],
  "status": "live",
  "inferenceMode": "onnx",
  "repoUrl": "https://github.com/<user>/<repo>",
  "notebookUrl": "https://github.com/<user>/<repo>/blob/main/notebook.ipynb",
  "thumbnail": "charts/regime-timeline.png",
  "charts": [{ "file": "regime-timeline.png", "caption": "Regimes over time" }],
  "order": 1
}
```

**`schema.json`**
```json
{
  "modelFile": "model.onnx",
  "modelVersion": "1.0.0",
  "inputs": [
    { "name": "volatility_20d", "label": "20-day volatility (%)", "type": "number", "min": 0, "max": 100, "default": 14.5, "help": "Annualised" },
    { "name": "trend_strength", "label": "Trend strength", "type": "number", "min": -1, "max": 1, "default": 0.2 },
    { "name": "gap_type", "label": "Gap", "type": "select", "options": ["none", "up", "down"], "default": "none" }
  ],
  "outputs": { "type": "classification", "labels": ["Calm Trend", "Choppy", "High Volatility"] },
  "presets": [{ "name": "Crisis-like", "values": { "volatility_20d": 45, "trend_strength": -0.7, "gap_type": "down" } }]
}
```
Presets render as one-click example buttons above the form.

## 5. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Performance | Home page under 1 s on broadband; prediction under 200 ms for tree/linear models; ONNX sessions created once and cached. |
| Model size | Each `model.onnx` ideally under 10 MB; use Git LFS above 50 MB. |
| Security | Server-side schema validation; request body limit (e.g. 16 KB); rate limiting on `/predict` (`AddRateLimiter`); HTTPS; security headers; no user data stored. |
| Accessibility | Semantic HTML, labelled form fields, keyboard navigable, WCAG AA contrast, alt text on charts. |
| SEO/Sharing | Per-page title/description, OpenGraph and Twitter card tags (thumbnail per project). |
| Reliability | Graceful errors if a model fails to load; the page still renders without the form. |
| Observability | Structured logging (Serilog optional); log slug, latency, validation failures; no input values at info level. |
| Browser support | Latest Chrome, Edge, Firefox, Safari; mobile responsive. |

## 6. Design Direction

- Clean, professional, data-focused. Dark and light themes with a toggle (respect `prefers-color-scheme`).
- One consistent type scale and spacing system; restrained palette with a single accent.
- Cards with hover lift, clear tag chips, thumbnails with consistent aspect ratio.
- Charts share one style (palette, fonts, sizes) across all projects. Export at 2x for crispness.
- UI library: **Bootstrap 5** (decided), with custom CSS variables for theming (light/dark).
- Visual direction: **"Editorial minimal"** (decided): warm off-white background, Georgia serif for headings, system sans for body, thin borders, a single indigo accent. Tokens live in `wwwroot/css/site.css`. Theme is set from `localStorage` or `prefers-color-scheme` before first paint.

## 7. Technology

- **.NET 10**, ASP.NET Core MVC, Razor views, `Microsoft.ML.OnnxRuntime`, Markdig, xUnit.
- **Python 3.12 + uv:** pandas, numpy, scikit-learn, XGBoost/LightGBM, PyTorch (deep learning examples only), plotly, matplotlib, `skl2onnx`, `onnxmltools`, `onnxruntime`, jupyter.
- **Front end:** vanilla JS (`fetch`), Plotly.js or Chart.js for interactive charts.
- **Hosting:** runs locally for now (`dotnet run`). Production hosting and custom domain are deferred and will be chosen later. **CI:** GitHub Actions (build, test); deploy step added once hosting is chosen.

## 8. Project Content Plan

> **v1 scope:** two pilot Model Zoo projects are built end to end (logistic regression, linear regression). Flagships (marked ⭐ below) appear as **Coming soon** placeholder cards until their phase starts. All other projects are backlog.

### Flagship ML (current backlog is finance/fintech; other domains welcome)
1. ⭐ NSE market regime classifier (flagship; NIFTY universe), placeholder in v1
2. Volatility forecasting (GARCH/ML comparison)
3. Next-day return direction classifier
4. Anomaly detection on price/volume
5. Credit risk scoring (synthetic fintech data)
6. Portfolio optimization demo (mean-variance with ML-estimated returns)

### Model Zoo (one per model)
Pilots (built): **Logistic regression** (borrower default, synthetic) and **Linear regression** (stock return vs market, synthetic). Both are finance-flavoured by design but are not labelled as finance projects; they carry the finance disclaimer. Full list:

Linear regression · Logistic regression · Decision tree · Random forest · Gradient boosting (XGBoost) · SVM · KNN · Naive Bayes · k-means · PCA · Gaussian mixture / DBSCAN · Small neural network (MLP) · Time-series model (ARIMA/Prophet-style or LSTM)

Each project's data is synthetic or openly licensed; the data source is stated on the page.

## 9. Per-Project Definition of Done

- [ ] `ml/<slug>/` is a uv project with a pinned `uv.lock` and README (problem, data, how to run), self-contained per §3.1.
- [ ] Notebook runs top to bottom from a clean environment.
- [ ] 3–4 charts exported with captions; one chosen as the thumbnail.
- [ ] `model.onnx` exported and verified with Python `onnxruntime`.
- [ ] `schema.json` written with sensible defaults, ranges, and at least 2 presets.
- [ ] `content/<slug>.md` written (problem, data, approach, results, limitations).
- [ ] Entry added to `projects.json`.
- [ ] Parity test added: .NET output equals Python output on 5+ reference inputs.
- [ ] Projects with `"disclaimer": "finance"` carry the disclaimer.
- [ ] GitHub repo link and notebook link verified.

## 10. Testing Strategy

- **Unit:** `ProjectCatalog` parsing, schema validation, input-to-tensor mapping.
- **Parity:** reference inputs/outputs from Python stored in `tests/fixtures/<slug>.json`.
- **Integration:** `WebApplicationFactory` tests for `/`, `/projects/{slug}`, `/predict/{slug}` (valid, invalid, unknown slug).
- **Manual:** responsive check, keyboard navigation, Lighthouse score target 90+ (performance, accessibility, SEO).

## 11. Milestones

| Phase | Scope | Estimate (part-time) |
|---|---|---|
| 0 | Solution scaffold, repo, `.gitignore`, theme decision | 0.5 day |
| 1 | Catalog service, homepage, shared layout, themes | 1 day ✅ done |
| 2 | Generic project page, Markdown content, chart gallery, dynamic form from schema | 1–2 days ✅ done |
| 3 | Two pilot Model Zoo projects end to end (logistic regression, linear regression), `PredictController`, parity tests | 2–3 days ✅ done |
| 3b | Flagship placeholders: "Coming soon" cards and pages (`status: "coming-soon"`, `inferenceMode: "none"`) for the NSE regime classifier and other flagships | 0.5 day ✅ done (delivered in Phases 1–2) |
| 4 | Flagship: NSE regime classifier (replaces its placeholder) | 3–5 days |
| 5 | Remaining Flagship and Model Zoo projects (repeat checklist) | ongoing |
| 6 | Dockerfile/CI, rate limiting, OpenGraph; deployment and custom domain once hosting is chosen | 1 day |

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Some models don't convert to ONNX | Keep preprocessing inside the sklearn pipeline; fall back to `lookup` mode; as a last resort add a FastAPI sidecar. |
| Feature engineering differs between Python and C# | Forms take already-computed features; parity tests catch drift. |
| Free-tier hosting cold starts | Show a loading state; keep the app lightweight. |
| Live-data expectations for finance projects | Use manual feature entry or historical-date lookup; state data cutoff on the page. |
| Scope creep across 20+ projects | Metadata-driven pages and the per-project checklist; ship pilot first. |

## 13. Open Questions

1. ~~Bootstrap 5 or Tailwind?~~ **Decided:** Bootstrap 5.
2. ~~Hosting target~~ **Deferred:** will be custom; run locally for now.
3. ~~Separate GitHub repo per project, or subfolders in one monorepo with deep links?~~ **Decided:** start as a monorepo (hybrid later). All projects live in this repo under `ml/<slug>/` and `repoUrl`/`notebookUrl` deep-link to the folder (e.g. `.../tree/main/ml/<slug>`). Flagship projects (e.g. NSE regime classifier) may later be split into their own repos (`git filter-repo --subdirectory-filter ml/<slug>` preserves history); only the `repoUrl` changes. See §3.1 for the rules that keep this split cheap.
4. ~~Custom domain name?~~ **Deferred:** custom domain later, with hosting.
5. ~~Which Model Zoo project is the pilot?~~ **Decided:** two pilots: logistic regression (classification output) and linear regression (numeric output), so both output types in `schema.json` are exercised. Flagships ship as placeholders in v1.
6. For the NSE regime classifier, form input via manual features or historical-date lookup (or both)?

## 14. Acceptance Criteria for v1

- Homepage lists the 2 live pilot Model Zoo projects and the flagship placeholder(s) (Coming soon), with working filters and search.
- Both pilot project pages have live forms returning correct model output (parity-tested).
- Placeholder pages render cleanly with no form and no broken links.
- All live projects link to their GitHub repo folder.
- App runs locally with `dotnet run`, with rate limiting and OpenGraph tags in place (public deployment is out of scope for v1).
- Lighthouse 90+ on Performance, Accessibility, and SEO for home and one project page (measured locally).
