# EK.MLPortfolio

Personal ML portfolio: finance/fintech ML projects and a Model Zoo, served by ASP.NET Core MVC (.NET 10) with ONNX Runtime inference. See [SPEC.md](SPEC.md).

## Prerequisites
- .NET SDK 10 (pinned in `global.json`)
- [uv](https://docs.astral.sh/uv/) for the Python training projects under `ml/`

## Run locally
```powershell
dotnet run --project src/MLPortfolio.Web
```

## Test
```powershell
dotnet test MLPortfolio.slnx
```
