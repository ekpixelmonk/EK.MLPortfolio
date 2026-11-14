# EK.MLPortfolio

Personal ML portfolio: finance/fintech ML projects and a Model Zoo, served by ASP.NET Core MVC (.NET 10) with ONNX Runtime inference. See [SPEC.md](SPEC.md).

## Prerequisites
- .NET SDK 10 (pinned in `global.json`)
- [uv](https://docs.astral.sh/uv/) for the Python training projects under `ml/`

## Status
Phase 1 complete: project catalog, themed homepage (light/dark), tag filter and search. Project pages and live models arrive in Phases 2–3; see the phase table in [SPEC.md](SPEC.md) and the log in [docs/ADR.md](docs/ADR.md).

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
