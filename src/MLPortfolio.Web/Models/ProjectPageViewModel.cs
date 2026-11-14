namespace MLPortfolio.Web.Models;

public sealed record ProjectPageViewModel(
    ProjectMeta Project,
    string? ContentHtml,
    FormSchema? Schema,
    /// <summary>False while the project is not <c>live</c>: the form is shown disabled as a preview.</summary>
    bool FormEnabled,
    IReadOnlyList<ProjectMeta> Related,
    /// <summary>Resolved GitHub links (see <see cref="SiteOptions"/>); null when not available.</summary>
    string? CodeUrl = null,
    string? NotebookUrl = null);
