namespace MLPortfolio.Web.Models;

public sealed record HomeViewModel(
    SiteOptions Site,
    IReadOnlyList<ProjectMeta> Flagship,
    IReadOnlyList<ProjectMeta> Zoo,
    IReadOnlyList<string> Domains,
    /// <summary>Tags shown as filter pills (see <see cref="Services.IProjectCatalog.SharedTags"/>).</summary>
    IReadOnlyList<string> Tags);
