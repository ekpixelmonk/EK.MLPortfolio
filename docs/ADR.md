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

---

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

### Phase 1: Catalog, homepage, layout, themes (not started)
### Phase 2: Generic project page, content, gallery, dynamic form (not started)
### Phase 3: Pilot projects, `PredictController`, parity tests (not started)
### Phase 3b: Flagship placeholders (not started)
### Phase 4: Flagship NSE regime classifier (not started)
### Phase 5: Remaining projects (not started)
### Phase 6: Docker, CI, rate limiting, OpenGraph, deployment (not started)
