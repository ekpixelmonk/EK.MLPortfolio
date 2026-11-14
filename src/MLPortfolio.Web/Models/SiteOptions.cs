namespace MLPortfolio.Web.Models;

/// <summary>Owner/branding settings, bound from the "Site" section of appsettings.json.</summary>
public sealed class SiteOptions
{
    public const string SectionName = "Site";

    public string OwnerName { get; set; } = "Kartik";
    public string Tagline { get; set; } = "Machine learning, engineered end to end.";
    public string Description { get; set; } = "Flagship ML projects and a Model Zoo, each with a live demo.";

    /// <summary>Optional. A link is rendered only when its URL is set.</summary>
    public string? GitHubUrl { get; set; }
    public string? LinkedInUrl { get; set; }

    /// <summary>
    /// Optional. The monorepo's GitHub URL. Live projects link to <c>ml/&lt;slug&gt;</c> inside it unless
    /// the project sets its own <c>repoUrl</c>.
    /// </summary>
    public string? RepoUrl { get; set; }

    /// <summary>Branch used for the derived code and notebook links.</summary>
    public string RepoBranch { get; set; } = "main";

    /// <summary>Resolves the code link for a project: its own <c>repoUrl</c>, else the monorepo folder, else none.</summary>
    public string? CodeUrlFor(ProjectMeta project)
    {
        if (!string.IsNullOrWhiteSpace(project.RepoUrl)) return project.RepoUrl;
        var root = NormalisedRepoUrl();
        return root is null || project.Status != ProjectStatuses.Live ? null : $"{root}/tree/{RepoBranch}/ml/{project.Slug}";
    }

    /// <summary>Resolves the notebook link: the project's own <c>notebookUrl</c>, else the monorepo notebook, else none.</summary>
    public string? NotebookUrlFor(ProjectMeta project)
    {
        if (!string.IsNullOrWhiteSpace(project.NotebookUrl)) return project.NotebookUrl;
        var root = NormalisedRepoUrl();
        return root is null || project.Status != ProjectStatuses.Live || !string.IsNullOrWhiteSpace(project.RepoUrl)
            ? null
            : $"{root}/blob/{RepoBranch}/ml/{project.Slug}/notebook.ipynb";
    }

    private string? NormalisedRepoUrl()
    {
        if (string.IsNullOrWhiteSpace(RepoUrl)) return null;
        var url = RepoUrl.Trim().TrimEnd('/');
        return url.EndsWith(".git", StringComparison.OrdinalIgnoreCase) ? url[..^4] : url;
    }
}
