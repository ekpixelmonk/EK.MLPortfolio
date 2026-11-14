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
    /// <summary>Subject area (see <see cref="ProjectDomains"/>). Required for flagship projects.</summary>
    public string? Domain { get; init; }
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

    public string? DomainLabel => ProjectDomains.LabelFor(Domain);

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

/// <summary>Subject areas a project can belong to. Add a value here to introduce a new domain.</summary>
public static class ProjectDomains
{
    public const string Finance = "finance";
    public const string Cybersecurity = "cybersecurity";

    private static readonly Dictionary<string, string> Labels = new()
    {
        [Finance] = "Finance",
        [Cybersecurity] = "Cybersecurity",
    };

    public static readonly IReadOnlySet<string> All = new HashSet<string>(Labels.Keys);

    public static string? LabelFor(string? domain) =>
        domain is not null && Labels.TryGetValue(domain, out var label) ? label : null;
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
    /// <summary>No form: the page states the model's reading of a stored snapshot of inputs (<c>snapshot.json</c>).</summary>
    public const string Snapshot = "snapshot";
    public const string None = "none";
    public static readonly IReadOnlySet<string> All = new HashSet<string> { Onnx, Lookup, Snapshot, None };
}
