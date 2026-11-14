namespace MLPortfolio.Web.Models;

public sealed record HomeViewModel(
    SiteOptions Site,
    IReadOnlyList<ProjectMeta> Flagship,
    IReadOnlyList<ProjectMeta> Zoo,
    IReadOnlyList<string> Tags);
