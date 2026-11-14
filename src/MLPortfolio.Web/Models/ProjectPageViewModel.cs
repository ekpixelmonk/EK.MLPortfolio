namespace MLPortfolio.Web.Models;

public sealed record ProjectPageViewModel(
    ProjectMeta Project,
    string? ContentHtml,
    FormSchema? Schema,
    /// <summary>False while the project is not <c>live</c>: the form is shown disabled as a preview.</summary>
    bool FormEnabled,
    IReadOnlyList<ProjectMeta> Related);
