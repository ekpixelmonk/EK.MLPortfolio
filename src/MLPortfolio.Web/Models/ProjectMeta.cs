namespace MLPortfolio.Web.Models;

/// <summary>One entry of <c>App_Data/projects.json</c>.</summary>
public sealed record ProjectMeta
{
    public string Slug { get; init; } = "";
    public string Title { get; init; } = "";
    public string Category { get; init; } = "";
    public string Summary { get; init; } = "";
    public IReadOnlyList<string> Tags { get; init; } = [];
    public string Status { get; init; } = "";
    public string InferenceMode { get; init; } = ProjectInferenceModes.None;
    public string? Disclaimer { get; init; }
    public string? RepoUrl { get; init; }
    public string? NotebookUrl { get; init; }
    public string? Thumbnail { get; init; }
    public IReadOnlyList<ProjectChart> Charts { get; init; } = [];
    public int Order { get; init; }

    public string StatusLabel => Status switch
    {
        ProjectStatuses.Live => "Live",
        ProjectStatuses.NotebookOnly => "Notebook only",
        _ => "Coming soon",
    };

    /// <summary>Web path of the thumbnail, or null when the project has none.</summary>
    public string? ThumbnailPath => string.IsNullOrWhiteSpace(Thumbnail) ? null : $"~/projects/{Slug}/{Thumbnail}";
}

public sealed record ProjectChart(string File, string Caption);

public static class ProjectCategories
{
    public const string Flagship = "flagship";
    public const string Zoo = "zoo";
    public static readonly IReadOnlySet<string> All = new HashSet<string> { Flagship, Zoo };
}

public static class ProjectStatuses
{
    public const string Live = "live";
    public const string NotebookOnly = "notebook-only";
    public const string ComingSoon = "coming-soon";
    public static readonly IReadOnlySet<string> All = new HashSet<string> { Live, NotebookOnly, ComingSoon };
}

public static class ProjectInferenceModes
{
    public const string Onnx = "onnx";
    public const string Lookup = "lookup";
    public const string None = "none";
    public static readonly IReadOnlySet<string> All = new HashSet<string> { Onnx, Lookup, None };
}
