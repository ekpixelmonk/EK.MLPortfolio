# Architecture Decision Records

Living technical record for ML Portfolio. Companion to [SPEC.md](../SPEC.md): the spec says *what* we build, this file records *why* we built it this way and *what actually happened* in each phase.

**How to maintain this file**
- Add a new ADR whenever a significant technical decision is made. Number sequentially; never delete an ADR. If a decision changes, mark the old one `Superseded by ADR-NNN` and add a new one.
- At the end of each phase, add an entry to the [Phase Log](#phase-log) (what was built, deviations from the spec, gotchas, follow-ups).

**Status values:** Proposed · Accepted · Superseded · Deprecated

## Index

| ADR | Title | Status | Phase |
|---|---|---|---|
| [001](#adr-001-monorepo-now-split-flagships-later) | Monorepo now, split flagships later | Accepted | 0 |
| [002](#adr-002-in-process-inference-with-onnx-runtime) | In-process inference with ONNX Runtime | Accepted | 0 |
| [003](#adr-003-metadata-driven-projects-and-onnx--schema-contract) | Metadata-driven projects and `model.onnx` + `schema.json` contract | Accepted | 0 |
| [004](#adr-004-bootstrap-5-for-the-ui) | Bootstrap 5 for the UI | Accepted | 0 |
| [005](#adr-005-net-10-sdk-pinned-slnx-solution-format) | .NET 10, SDK pinned via `global.json`, `.slnx` solution | Accepted | 0 |
| [006](#adr-006-local-first-hosting-deployment-deferred) | Local-first; hosting and domain deferred | Accepted | 0 |
| [007](#adr-007-v1-scope-two-pilots-plus-flagship-placeholders) | v1 scope: two pilots plus flagship placeholders | Accepted | 0 |
| [008](#adr-008-visual-direction-editorial-minimal-with-css-tokens) | Visual direction: "Editorial minimal" with CSS tokens | Accepted | 1 |
| [009](#adr-009-category-names-and-per-project-disclaimer) | Categories `flagship`/`zoo`; disclaimer is per project | Accepted | 1 |
| [010](#adr-010-fail-fast-immutable-projectcatalog) | Fail-fast, immutable `ProjectCatalog` | Accepted | 1 |
| [011](#adr-011-site-owner-settings-in-configuration) | Site owner settings in configuration | Accepted | 1 |
| [012](#adr-012-theme-before-first-paint-client-side-filtering-no-jquery) | Theme before first paint; client-side filtering; no jQuery | Accepted | 1 |
| [013](#adr-013-domain-classification-for-flagship-projects) | Domain classification for flagship projects | Accepted | 1 |
| [014](#adr-014-schema-driven-form-with-validated-cached-schemas) | Schema-driven form with validated, cached schemas | Accepted | 2 |
| [015](#adr-015-markdown-write-ups-with-raw-html-disabled) | Markdown write-ups with raw HTML disabled | Accepted | 2 |
| [016](#adr-016-styled-error-pages-via-status-code-re-execution) | Styled error pages via status-code re-execution | Accepted | 2 |
| [017](#adr-017-form-preview-for-projects-that-are-not-live) | Form preview for projects that are not live | Accepted | 2 |
| [018](#adr-018-onnx-input-convention-and-in-process-runner) | ONNX input convention and in-process runner | Accepted | 3 |
| [019](#adr-019-prediction-api-contract-limits-and-error-handling) | Prediction API contract, limits and error handling | Accepted | 3 |
| [020](#adr-020-train-py-as-the-single-source-of-truth-for-a-project) | `train.py` as the single source of truth for a project | Accepted | 3 |
| [021](#adr-021-finance-flavoured-synthetic-pilots) | Finance-flavoured synthetic pilots | Accepted | 3 |

---

## ADR-001: Monorepo now, split flagships later

- **Status:** Accepted · **Phase:** 0
- **Context:** The portfolio will hold 20+ projects. A repo per project means heavy overhead and awkward sharing of parity-test fixtures with the web app. A flagship (NSE regime classifier) may deserve its own repo for profile visibility.
- **Decision:** Everything lives in this one repo. Each `ml/<slug>/` is a fully self-contained uv project. `repoUrl`/`notebookUrl` deep-link to the folder. A project can later be extracted with `git filter-repo --subdirectory-filter ml/<slug>`, changing only its `repoUrl`.
- **Rules that keep the split cheap** (SPEC §3.1): no cross-folder imports; shared assets copied or packaged, not imported; `model.onnx` + `schema.json` committed under `App_Data/models/<slug>/`; each README "how to run" works from a clean clone of its own folder.
- **Consequences:** + one clone, one CI, easy parity tests. − repo grows with models/charts (Git LFS above 50 MB); only extracted projects get their own pinned repo on the GitHub profile.

## ADR-002: In-process inference with ONNX Runtime

- **Status:** Accepted · **Phase:** 0
- **Context:** Models are trained in Python but the site is ASP.NET Core. Options: a Python sidecar (FastAPI), or export to ONNX and run inside the web app.
- **Decision:** Export models to ONNX (`skl2onnx`, `onnxmltools`) and run them in the MVC app with `Microsoft.ML.OnnxRuntime`. No Python service by default. Sessions are created once and cached.
- **Fallbacks** (SPEC §12): keep preprocessing inside the sklearn pipeline; use `inferenceMode: "lookup"` (precomputed `lookup.json`) for models that don't convert; FastAPI sidecar only as a last resort.
- **Consequences:** + single deployable, fast (<200 ms target) predictions. − not every model converts; feature engineering must not diverge between Python and C#, so forms take already-computed features and parity tests guard drift.

## ADR-003: Metadata-driven projects and `model.onnx` + `schema.json` contract

- **Status:** Accepted · **Phase:** 0
- **Context:** Adding a project should not require new controllers or views.
- **Decision:** Pages are generated from data: `App_Data/projects.json` (catalog), `App_Data/content/<slug>.md` (write-up, rendered with Markdig), `App_Data/models/<slug>/{model.onnx, schema.json, lookup.json?}` (inference), `wwwroot/projects/<slug>/charts/` (visuals). The prediction form is generated from `schema.json`, and `POST /predict/{slug}` validates input server-side against it. `inferenceMode` is `onnx`, `lookup`, or `none` (hides the form).
- **Consequences:** + new project = folder + JSON entry. − `schema.json` must stay expressive enough (number, select, presets, output types); changes to its shape affect every project, so version it deliberately.

## ADR-004: Bootstrap 5 for the UI

- **Status:** Accepted · **Phase:** 0 (replaces open question 1)
- **Context:** Choice was Bootstrap 5 or Tailwind.
- **Decision:** Bootstrap 5.3 (the MVC template bundles 5.3.3) with custom CSS variables for theming. Dark/light via `data-bs-theme`, respecting `prefers-color-scheme`, with a manual toggle.
- **Consequences:** + no build pipeline, fast to ship, built-in dark mode. − more generic look by default; a restrained palette and single accent must be applied through CSS variables to stand out.

## ADR-005: .NET 10, SDK pinned via `global.json`, `.slnx` solution

- **Status:** Accepted · **Phase:** 0
- **Context:** The dev machine has .NET SDK 8, 9 and 10 installed. .NET 10 creates `.slnx` solutions by default. NuGet's latest `Microsoft.AspNetCore.Mvc.Testing` (11.0.0) targets net11.0 only.
- **Decision:**
  - Target `net10.0`. `global.json` pins SDK `10.0.204` with `rollForward: latestFeature`.
  - Use the `.slnx` format (`MLPortfolio.slnx`) instead of `.sln`.
  - Pin `Microsoft.AspNetCore.Mvc.Testing` to `10.0.15` (latest 10.x).
- **Consequences:** + reproducible builds across installed SDKs. − when upgrading to .NET 11 the test package pin must move with it; older tooling that predates `.slnx` can't open the solution.

## ADR-006: Local-first; hosting and domain deferred

- **Status:** Accepted · **Phase:** 0 (replaces open questions 2 and 4)
- **Decision:** v1 runs locally with `dotnet run`. Production hosting and a custom domain will be chosen later. CI (GitHub Actions) builds and tests now; a deploy step is added once hosting is chosen. Public-facing hardening (rate limiting, OpenGraph, security headers) is still built in v1 so deployment is not blocked later.
- **Consequences:** + no hosting cost or lock-in decisions yet. − Lighthouse and cold-start behaviour are only measurable locally until deployment.

## ADR-007: v1 scope: two pilots plus flagship placeholders

- **Status:** Accepted · **Phase:** 0 (replaces open question 5)
- **Context:** Scope creep across 20+ projects is the main risk (SPEC §12).
- **Decision:** v1 builds two Model Zoo pilots end to end: **logistic regression** (classification output) and **linear regression** (numeric output), so both output types in `schema.json` are exercised and parity-tested. Flagships (e.g. NSE regime classifier) ship as "Coming soon" placeholders (`status: "coming-soon"`, `inferenceMode: "none"`) in Phase 3b and are built for real in Phase 4.
- **Open:** NSE classifier input mode (manual features vs historical-date lookup) is deferred until Phase 4 (SPEC §13, Q6).

## ADR-008: Visual direction: "Editorial minimal" with CSS tokens

- **Status:** Accepted · **Phase:** 1
- **Context:** The site must read as built by a detail-oriented professional. Three directions were mocked up (editorial minimal, data-forward dark, soft modern); the owner chose the first.
- **Decision:** Warm off-white, Georgia serif headings, system sans body, thin borders, one indigo accent. All colours live as tokens under `[data-bs-theme="light|dark"]` in `site.css`, mapped onto Bootstrap's `--bs-*` variables. No web fonts. Cards lift on hover, with `prefers-reduced-motion` respected and a skip link plus visible focus rings.
- **Consequences:** + no font downloads, consistent theming for later pages and charts. − Georgia is a safe but not distinctive choice; the owner expects to refine the look iteratively.

## ADR-009: Category names and per-project disclaimer

- **Status:** Accepted · **Phase:** 1
- **Context:** The owner may build non-finance flagship projects, so a "Finance & Fintech ML" section name and a disclaimer tied to it were too narrow.
- **Decision:** Categories are `flagship` ("Flagship ML") and `zoo` ("Model Zoo"). The finance disclaimer is a separate `disclaimer: "finance"` field per project in `projects.json` (rendered from Phase 2). The catalog rejects unknown categories, statuses and inference modes.
- **Consequences:** + new domains need no code change. − the disclaimer must be set deliberately on each finance project (covered by the per-project checklist).

## ADR-010: Fail-fast, immutable `ProjectCatalog`

- **Status:** Accepted · **Phase:** 1
- **Context:** Every page depends on `projects.json`. A typo should not surface as a broken card or a runtime 500.
- **Decision:** `ProjectCatalog.LoadFile` parses and validates the whole file once at startup (required fields, slug format and uniqueness, allowed enum values; JSON comments and trailing commas allowed) and reports every problem in one `ProjectCatalogException`. The catalog is a singleton, immutable, sorted by `order` then title. Edits to `projects.json` need an app restart.
- **Consequences:** + bad data is caught immediately with a readable message; no per-request I/O. − no hot reload of the catalog; acceptable for a portfolio, revisit if editing live becomes useful.

## ADR-011: Site owner settings in configuration

- **Status:** Accepted · **Phase:** 1
- **Decision:** Owner name, tagline, description and GitHub/LinkedIn URLs are bound from the `Site` section of `appsettings.json` (`SiteOptions`). A social link renders only when its URL is non-empty, so the site never ships made-up URLs. Environment variables (`Site__GitHubUrl`) can override them in deployment.
- **Consequences:** + content changes need no code change. − the URLs are blank until the owner fills them in.

## ADR-012: Theme before first paint, client-side filtering, no jQuery

- **Status:** Accepted · **Phase:** 1
- **Decision:**
  - A small inline script in `<head>` sets `data-bs-theme` from `localStorage` or `prefers-color-scheme` before the stylesheet applies, avoiding a flash; the toggle persists the choice (storage access wrapped in try/catch).
  - Tag filter and search are vanilla JS over server-rendered cards (`data-tags`, `data-search`), with `aria-pressed` on chips and a polite live region announcing counts.
  - The template's jQuery and scoped-CSS references were removed; Bootstrap's bundle remains for later components.
- **Consequences:** + fast, no framework, works without a build step. − an inline script will need a CSP nonce or hash when security headers are added in Phase 6.

## ADR-013: Domain classification for flagship projects

- **Status:** Accepted · **Phase:** 1 (follow-up change)
- **Context:** Flagship projects can span subject areas (finance now, cybersecurity planned). Filtering by a narrow tag like "NSE" doesn't scale to that.
- **Decision:** `projects.json` gets a `domain` field. Allowed values are a closed set in `ProjectDomains` (`finance`, `cybersecurity`), required for flagship projects and optional for Model Zoo. The homepage shows a pill per domain *that a project actually uses*, ahead of the tag pills, and a domain label on the card. The old `NSE` tag was removed in favour of the Finance domain. `disclaimer` stays a separate field (ADR-009).
- **Consequences:** + a new domain is one line in `ProjectDomains` plus a project using it; no empty pills, so Cybersecurity stays hidden until it has a project. − domains are code, not data, so adding one needs a rebuild.

## ADR-014: Schema-driven form with validated, cached schemas

- **Status:** Accepted · **Phase:** 2
- **Context:** The input form must come from `schema.json` (ADR-003), and the same file will later validate `POST /predict`. A broken schema must not take a project page down.
- **Decision:** `SchemaStore` loads `App_Data/models/<slug>/schema.json` once per slug and caches it. It validates on load: identifier-style input names, unique names, known types (`number`, `select`), `min <= max`, positive `step`, select options present, defaults and preset values valid for their input, outputs type known, classification needs two or more labels. An invalid schema logs a warning and the page renders without a form. Only slugs found in the catalog ever reach the file system, which rules out path traversal. The form is rendered server-side (`_TryModel`); `project.js` only adds presets, local validation and the fetch call.
- **Consequences:** + the schema is checked once, in one place, and Phase 3 can reuse the same model for server-side validation. − the cache means schema edits need an app restart. The model only supports `number` and `select` inputs; new input types need schema, view and validation changes.

## ADR-015: Markdown write-ups with raw HTML disabled

- **Status:** Accepted · **Phase:** 2
- **Decision:** Write-ups are `App_Data/content/<slug>.md`, rendered with Markdig (advanced extensions: tables, etc.) with raw HTML turned off, then output with `Html.Raw`. Rendered HTML is cached per slug.
- **Consequences:** + a write-up can never inject markup or script, so `Html.Raw` is safe. − authors can't embed custom HTML; use images or Markdown, or revisit with a sanitiser if that becomes limiting.

## ADR-016: Styled error pages via status-code re-execution

- **Status:** Accepted · **Phase:** 2
- **Decision:** `UseStatusCodePagesWithReExecute("/error/{0}")` sends bare status codes to `HomeController.StatusCodePage`, which renders a styled `NotFound` view for 404 and the generic error view otherwise. The real status code is preserved. `ProjectsController` simply returns `NotFound()` for unknown slugs.
- **Consequences:** + one consistent 404 for unknown slugs and unknown URLs, correct status for crawlers. − API endpoints returning empty 404s (e.g. `/predict/{slug}` in Phase 3) will also be re-executed, so they should return a body or opt out.

## ADR-017: Form preview for projects that are not live

- **Status:** Accepted · **Phase:** 2
- **Context:** The pilots have a draft schema but no model yet, and the owner wanted to see the form working in the UI before Phase 3.
- **Decision:** When a project's status is not `live` but a valid schema exists, the form renders disabled with a notice that predictions are switched off. The Predict button and presets are disabled. A project whose `inferenceMode` is `none` never shows a form. The draft schema for logistic regression (exam pass/fail from study hours, attendance, previous score and tutoring) is marked as draft in `schema.json` and may change when the model is trained.
- **Consequences:** + visitors and the owner can see what the demo will look like without a misleading live result. − the draft schema's features are provisional; Phase 3 may change them.

## ADR-018: ONNX input convention and in-process runner

- **Status:** Accepted · **Phase:** 3
- **Context:** ADR-002 chose in-process ONNX Runtime. We need a mapping between schema inputs and model inputs that works for every future project.
- **Decision:** One ONNX input per schema input, named after it, shape `[1,1]`: `float` for `number`, `string` for `select`. On the Python side each column gets its own `initial_types` entry when converting with `skl2onnx` (ZipMap disabled for classifiers). `OnnxModelRunner` creates one `InferenceSession` per slug lazily, caches it, and disposes them on shutdown; a failed load is not cached, so fixing the file and retrying works. Classifiers: the probability tensor is found by name (contains "prob"), else the second output; regressors: the first output.
- **Consequences:** + no per-model C# code, and the schema alone describes the model. − tree ensembles or other converters that emit different output names or shapes may need small additions to the output reader; models needing array/vector inputs would need a new input type.

## ADR-019: Prediction API contract, limits and error handling

- **Status:** Accepted · **Phase:** 3
- **Decision:**
  - `POST /predict/{slug}` accepts a JSON object and returns `{ prediction, label?, probabilities?, explanation?, modelVersion }`. Classification: `prediction` is the class index, `probabilities` a label→value map. Regression: `prediction` is the value and `label` a display string such as `+3.19%` built from `outputs.unit`.
  - Validation is strict and server-side (`InputValidator`): unknown fields rejected, required fields enforced, ranges and options checked; all field errors are returned together in `{ "errors": { field: message } }` with status 400.
  - Non-live, unknown, `inferenceMode: none` or schema-less projects return 404 with a JSON `{ error }` body, so the status-code page re-execution from ADR-016 does not replace API errors with HTML. A missing/broken model returns 503 with a generic message.
  - 16 KB body limit (`RequestSizeLimit`), fixed-window rate limit per client IP (30/min by default, configurable via `RateLimiting:PredictPermitsPerMinute`), 429 on excess.
  - Logs carry slug, outcome and latency; input values are never logged.
- **Consequences:** + predictable for the page script and for any other client. − the rate limit keys on the remote IP, so behind a reverse proxy the forwarded-headers middleware must be configured at deployment or every user shares one bucket. The body-size limit is enforced by Kestrel, not by the in-memory test server, so it has no automated test. `lookup` mode is still unimplemented.

## ADR-020: `train.py` as the single source of truth for a project

- **Status:** Accepted · **Phase:** 3
- **Context:** The schema, the ONNX input names, the charts and the parity fixtures must never drift apart.
- **Decision:** Each `ml/<slug>/train.py` defines the inputs once and generates everything the site consumes: `model.onnx`, `schema.json` (with presets and ranges), `charts/*.png` (2x, shared style, plus `thumbnail.png`), and `tests/fixtures/<slug>.json` (scikit-learn's answers for the presets plus random in-range inputs). It asserts that onnxruntime matches scikit-learn on the whole test set *before* writing anything. The notebook reuses `train.py` so notebook and site cannot diverge. `train.py` takes `--repo-root` so a project can be extracted and still write into a checkout (ADR-001). The `.NET` parity test compares the runner against the fixtures (probabilities to 4 decimals, regression values to 3).
- **Consequences:** + one command regenerates and re-validates a project. − generated `schema.json`, fixtures and charts are committed artefacts that must be regenerated together; hand-editing `schema.json` would be overwritten. Projects duplicate a little code (chart style) to stay self-contained.

## ADR-021: Finance-flavoured synthetic pilots

- **Status:** Accepted · **Phase:** 3
- **Context:** The owner wanted both Model Zoo pilots to have a finance or stock-market problem statement, without labelling them as finance projects.
- **Decision:**
  - Logistic regression: classify a borrower as `Repaid` or `Default` from income, debt-to-income, credit history, late payments and employment (synthetic).
  - Linear regression: predict a stock's daily return from market return, sector return and volume ratio (synthetic, known beta ≈ 1.15).
  - Both stay in the `zoo` category with no `domain` (so no Finance pill), but set `"disclaimer": "finance"` so the "Educational purposes only. Not investment advice." note shows. Data is generated, never real, and the pages say so.
- **Consequences:** + the zoo reads as relevant to a finance audience while staying general. − the credit scenario overlaps in theme with the planned flagship "Credit risk scoring"; that flagship should use richer data and a different model.

---

## ADR-022: Zoo batch 1 conventions

- **Status:** accepted.
- **Context:** Five more Zoo models were added in one batch. They had to fit the existing schema (classification with labels, or regression with a unit) and the parity-test approach without new .NET code.
- **Decision:**
  - Only models whose output is a class or a number are built now: decision tree, random forest (regression), SVM, KNN (three classes) and naive Bayes. Gradient boosting (needs `onnxmltools`), k-means, PCA, GMM/DBSCAN, MLP and time series are deferred; the unsupervised ones need new output types in `schema.json`.
  - The predicted class is the arg-max of the ONNX probabilities, in `train.py` checks, fixtures and the .NET runner alike. SVC's own label output uses a different rule near 50% and disagreed with the arg-max for 0.24% of rows.
  - `scikit-learn` is pinned `<1.9`: `SVC(probability=True)` is deprecated from 1.9 and the replacement may not convert with skl2onnx.
  - Parity tolerance is per model: 1e-4 on probabilities by default, 2e-3 for the SVM (Platt scaling in float32). The .NET test skips the class check only when the top two probabilities are within 0.01.
  - Every model's write-up reports accuracy next to the majority-class baseline; where the model barely beats it (SVM) the page says so and leans on AUC.
- **Consequences:** adding a model is a `train.py`, a catalog entry, a write-up and a line in the test lists. Charts that extrapolate (the SVM probability surface) are limited to the range with plenty of training data.

## Phase Log

Record what was actually built, deviations from the spec, gotchas and follow-ups. Add one entry per phase when it completes.

### Phase 0: Scaffold (complete)

- **Built:** solution `MLPortfolio.slnx`; `src/MLPortfolio.Web` (ASP.NET Core MVC, net10.0) with Markdig 1.6.0 and Microsoft.ML.OnnxRuntime 1.32.0; `tests/MLPortfolio.Tests` (xUnit, references the web project, `Mvc.Testing` 10.0.15); empty `ml/`, `App_Data/{content,models}`, `Services/`, `wwwroot/projects/`, `tests/fixtures/`; `global.json`; Python entries in `.gitignore`; README with run/test commands.
- **Verification:** `dotnet build MLPortfolio.slnx` succeeds with 0 warnings and 0 errors. The app was run with `dotnet run --project src/MLPortfolio.Web --launch-profile http`: `/` and `/Home/Privacy` return 200, an unknown route returns 404, and the bundled Bootstrap CSS is served (200). There are no tests yet.
- **Deviations from the spec:** `.slnx` instead of `.sln` (ADR-005). Spec layout and §3 updated.
- **Gotchas:**
  - `dotnet add package Microsoft.AspNetCore.Mvc.Testing` without a version resolves to 11.0.0 and fails with NU1202 on net10.0; pin a 10.x version.
  - Naming the solution `MLPortfolio.sln` with `dotnet new sln` produces `.slnx` on .NET 10, so `dotnet sln MLPortfolio.sln add` fails. Use the `.slnx` name.
  - Git on Windows prints LF→CRLF warnings; harmless. A `.gitattributes` can set line endings explicitly if desired.
- **Follow-ups:** none blocking. Phase 1 next (ProjectCatalog, homepage, layout, themes).

### Phase 1: Catalog, homepage, layout, themes (complete)

- **Built:**
  - `ProjectCatalog` / `IProjectCatalog` (validated, immutable, singleton), `ProjectMeta`, `SiteOptions`, `HomeViewModel`.
  - `App_Data/projects.json` with three entries (NSE flagship plus the two pilots), all `coming-soon`, and placeholder SVG thumbnails under `wwwroot/projects/<slug>/charts/`.
  - Shared layout (skip link, header with Projects/GitHub/LinkedIn and theme toggle, footer), Home view with hero, tag chips, search and two card sections (`_ProjectSection` partial), restyled Error view.
  - `site.css` tokens for the editorial-minimal theme in light and dark; `site.js` (theme) and `home.js` (filter/search).
  - Per-page title, description and OpenGraph tags; `public partial class Program` for integration tests.
- **Verification:** `dotnet test` passes 35 tests (catalog parsing, validation, sorting and lookup; homepage content, assets and absent social links). Run locally and checked in Chrome: dark theme from the system setting, toggle switches and persists, tag filter, search and the empty state behave correctly.
- **Deviations from the plan:**
  - The two pilots are `coming-soon`, not `live`, until Phase 3 builds them, so a "Live" badge never links to a missing page.
  - Social links are hidden until configured, instead of showing placeholder URLs.
  - The template's Privacy page and `_Layout.cshtml.css` were removed.
- **Gotchas:** Razor HTML-encodes `·` in `<title>`, so tests match around it. A `SAME`/`same` duplicate-slug test first failed because the lowercase-slug rule fires before the duplicate check.
- **Follow-ups:**
  - Hero copy says "each with a live demo", which is not yet true; reword or wait for Phase 3.
  - Fill in `Site:GitHubUrl` and `Site:LinkedInUrl`.
  - Card links 404 until Phase 2.
  - Add a CSP when security headers arrive in Phase 6 (inline theme script).
  - Not yet checked: Lighthouse scores and the narrow-screen layout.
### Phase 2: Generic project page, content, gallery, dynamic form (complete)

- **Built:**
  - `/projects/{slug}`: header (status, domain, tags, GitHub button), Markdown summary, chart gallery with captions, "Try the model" form, links and related projects, finance disclaimer.
  - `ProjectsController`, `ProjectPageViewModel`, `Details` and `_TryModel` views.
  - `FormSchema` model, `SchemaStore` (validated, cached), `ProjectContentStore` (Markdig), `IProjectCatalog.Related`.
  - `wwwroot/js/project.js`: presets, local validation and the `/predict/{slug}` call, expecting the response shapes in SPEC §4.3.
  - Styled 404 and error pages; project-page CSS tokens.
  - Write-ups for the three catalog projects, and a draft `schema.json` for logistic regression.
- **Verification:** `dotnet test` passes 76 tests: schema validation (including every rejection case), Markdown rendering (raw HTML stripped), related-project ranking, and page tests for known and unknown slugs, case-insensitive slugs, a path-traversal slug, disabled form preview, a live project with a chart gallery and links, a project without content or charts, and the finance disclaimer. The running app returns 200 for the three project pages and 404 for unknown URLs.
- **Not verified:**
  - The page was not looked at in a browser this session (the Chrome extension was not connected).
  - Preset and submit behaviour in `project.js` is untested until Phase 3 provides an endpoint.
  - Mobile layout.
- **Deviations from the spec:** Plotly JSON charts are deferred (images only for now). Form shows as a disabled preview for non-live projects (ADR-017).
- **Gotchas:** Razor boolean attributes (`disabled`, `required`, `selected`) are omitted when false, which the form relies on. The chart gallery and live-form paths are tested by swapping `IProjectCatalog` in `WebApplicationFactory`, since no real project has charts yet.
- **Follow-ups:**
  - Phase 3 must add `POST /predict/{slug}` returning the SPEC §4.3 shapes, and mark the pilots `live`.
  - Return a body from `/predict` errors (ADR-016).
  - Confirm the draft logistic regression features before training.
  - Decide whether linear regression gets a draft schema too.
### Phase 3: Pilot projects, `PredictController`, parity tests (complete)

- **Built:**
  - Python: `ml/logistic-regression` and `ml/linear-regression` (uv, Python 3.12, committed `uv.lock`), each with `train.py`, an executed `notebook.ipynb` and a README.
  - Generated artefacts: `model.onnx`, `schema.json`, four charts plus a thumbnail per project, and parity fixtures in `tests/fixtures/`.
  - .NET: `InputValidator`, `IModelRunner`/`OnnxModelRunner`, `PredictController`, `PredictResponse`, a `predict` rate-limit policy, `RateLimiting` setting.
  - Both pilots are `live` in `projects.json` with chart captions; write-ups now contain real results.
- **Results:** logistic regression ROC AUC 0.81, accuracy 85.5% (majority-class baseline about 83%); linear regression R² 0.78, RMSE 0.81 pp. ONNX matches scikit-learn to within 1e-7 (classifier) and 5e-7 (regressor) on the held-out sets.
- **Verification:** `dotnet test` passes 111 tests, including parity of the .NET runner against the Python fixtures (8 reference inputs each), endpoint behaviour (200 for both output types, 400 with per-field errors, malformed and non-object bodies, 415, JSON 404s, 503 for a failing or missing model), validator rules, and the rate limit (200, 200, 429, 429; pages unaffected). The running app answers `/predict` correctly for both projects; each call takes under 1 ms inside the app.
- **Not verified:**
  - The "Try the model" page script has not been exercised in a browser (the Chrome extension was not connected during Phases 2–3), so presets, Predict, the probability bars and error display are untested end to end.
  - The notebooks were executed, but their outputs were not reviewed visually.
  - The 16 KB body limit.
  - Rate limiting behind a proxy.
- **Deviations from the spec:**
  - `lookup` inference mode deferred.
  - Per-project `repoUrl` left empty: no GitHub remote or owner URL exists yet, so the GitHub buttons and the "GitHub repo and notebook link verified" checklist item are open.
  - The pilots' draft schema from Phase 2 was replaced by the generated one (borrower default instead of exam pass/fail).
- **Gotchas:**
  - `uv sync` failed with `invalid peer certificate: UnknownIssuer` on this machine; `UV_NATIVE_TLS=1` (system certificate store, verification still on) fixes it.
  - Newer scikit-learn `RocCurveDisplay.from_predictions` rejects `color=`; the ROC curve is drawn manually.
  - `jupyter nbconvert --execute` needs a registered kernel, so `ipykernel install --user --name python3` was run once, which writes a kernel spec into the user's Jupyter data directory (outside the repo).
  - Razor/`ToString("P1")` with the invariant culture renders "97.9 %", so percentages are formatted manually.
  - The shared test factory would hit the per-IP rate limit across many requests, so `PredictTests` raises the limit and the rate-limit tests lower it per factory.
- **Follow-ups:**
  - Add a GitHub remote and set `repoUrl`/`notebookUrl` for the pilots.
  - Check the form end to end in a browser.
  - Use `ForwardedHeaders` and a CSP at deployment (Phase 6).
  - Decide whether chart PNGs should use Git LFS if they grow.

### Phase 3b: Flagship placeholders (complete, delivered earlier)

- The NSE regime classifier already has a "Coming soon" card (Phase 1) and a project page with write-up and finance disclaimer (Phase 2), with `inferenceMode: none` so no form is shown. Nothing further was needed; `/predict/nse-regime` returns a JSON 404.
### Phase 4: Flagship NSE regime classifier (not started)
### Phase 5: Remaining projects (in progress)

#### Zoo batch 1 (complete)

- **Built:** `ml/decision-tree`, `ml/random-forest`, `ml/svm`, `ml/knn`, `ml/naive-bayes` (uv projects with `train.py`, executed notebook, README), generated model/schema/charts/fixtures, `projects.json` entries (orders 3-7), write-ups with real results, test updates.
- **Results (held-out):**

| Model | Accuracy / R² | Baseline | ROC AUC |
|---|---|---|---|
| Decision tree (approval) | 93.2% | 55.7% (majority) | 0.93 |
| SVM (fraud, 15% positives) | 85.8% | 85.0% (majority) | 0.75 |
| KNN (3 classes) | 85.3% | 37.9% (majority) | 0.95 |
| Naive Bayes (churn, 35%) | 71.9% | 65.4% (majority) | 0.79 |
| Random forest (volatility) | R² 0.85, RMSE 2.57 | linear R² 0.80, RMSE 3.01 | n/a |

  ONNX matches scikit-learn to within 1.2e-6 on probabilities (2.2e-8 for tree, KNN) and 4e-6 on the forest's predictions; the forest's ONNX file is about 0.9 MB.
- **Verification:** `dotnet test` passes 138 tests: parity for all seven models against the Python fixtures, every preset of every model through `/predict`, homepage and page checks for the new projects. Charts for the tree, SVM, forest and naive Bayes were viewed and fixed (overlapping tree boxes, SVM surface artefact at large amounts, hidden overlapping histograms). The notebooks executed without errors.
- **Not verified:** the form script in a browser (still untested end to end); notebook outputs were not read cell by cell.
- **Gotchas:**
  - Initial synthetic data was off: SVM AUC 0.62 (signal too weak), KNN 98% (too easy), naive Bayes below the baseline. The generators were retuned until results were plausible and honest.
  - A running dev server locks `bin\...\MLPortfolio.Web.exe`, which breaks `dotnet test`; stop it first.
  - Git Bash heredocs with apostrophes failed in one tool call; write Markdown files with the editor tool.
- **Follow-ups:** gradient boosting (`onnxmltools`), then the unsupervised models once `schema.json` has cluster/projection outputs; GitHub remote and `repoUrl`; browser check of the form.
### Phase 6: Docker, CI, rate limiting, OpenGraph, deployment (not started)
