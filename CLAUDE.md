# CLAUDE.md

Personal ML portfolio: ASP.NET Core MVC (.NET 10) serving ONNX models trained in Python (uv). Read [SPEC.md](SPEC.md) for requirements and [docs/ADR.md](docs/ADR.md) for decisions and the per-phase log.

## Commands (PowerShell, Windows)
```powershell
dotnet build MLPortfolio.slnx
dotnet test MLPortfolio.slnx
dotnet run --project src/MLPortfolio.Web --launch-profile http   # http://localhost:5215
```
SDK is pinned in `global.json` (10.0.x). Do not add `Microsoft.AspNetCore.Mvc.Testing` without `--version 10.0.x` (latest is 11.x and breaks net10.0).

## Layout
- `src/MLPortfolio.Web`: the site. `Services/ProjectCatalog` loads and validates `App_Data/projects.json` at startup; `Controllers/HomeController` renders the homepage; views in `Views/`; theme tokens in `wwwroot/css/site.css`; `wwwroot/js/site.js` (theme toggle) and `home.js` (filter/search).
- `src/MLPortfolio.Web/App_Data`: `projects.json`, `content/<slug>.md`, `models/<slug>/{model.onnx, schema.json}`.
- `ml/<slug>/`: self-contained uv projects (notebook, `train.py`, export). Never import across folders or into the web app.
- `tests/MLPortfolio.Tests`: xUnit unit tests plus `WebApplicationFactory<Program>` integration tests.

## Conventions
- Projects are metadata-driven: adding one is data (folder + `projects.json` entry), not new controller/view code.
- Keep a project's catalog `status` as `coming-soon` until it is really built.
- Site owner info and social links live in `appsettings.json` (`Site`); never invent URLs.
- Colours and spacing go through the CSS tokens in `site.css`; keep light and dark in sync and WCAG AA.
- Razor HTML-encodes characters like `·`; assert around them in tests.

## Workflow
- Don't write implementation plans; build directly once scope is agreed.
- After each implementation, update SPEC.md, CLAUDE.md, README.md and docs/ADR.md (new ADR for each real decision, a Phase Log entry per phase).
- Don't commit unless asked.
