# EK.MLPortfolio

Personal ML portfolio: flagship ML projects and a Model Zoo, served by ASP.NET Core MVC (.NET 10) with ONNX Runtime inference. See [SPEC.md](SPEC.md).

## Prerequisites
- .NET SDK 10 (pinned in `global.json`)
- [uv](https://docs.astral.sh/uv/) for the Python training projects under `ml/`

## Status
Phases 1–2 complete: project catalog, themed homepage (light/dark) with domain/tag filters and search, and generic project pages (Markdown write-up, chart gallery, form generated from `schema.json`, styled 404). The prediction endpoint and real models arrive in Phase 3; see the phase table in [SPEC.md](SPEC.md) and the log in [docs/ADR.md](docs/ADR.md).

## Configure
Owner name, tagline and social links are in `src/MLPortfolio.Web/appsettings.json` under `Site`. GitHub/LinkedIn links appear only once their URLs are set. Projects are listed in `src/MLPortfolio.Web/App_Data/projects.json`.

## Run locally
```powershell
dotnet run --project src/MLPortfolio.Web
```

## Test
```powershell
dotnet test MLPortfolio.slnx
```
