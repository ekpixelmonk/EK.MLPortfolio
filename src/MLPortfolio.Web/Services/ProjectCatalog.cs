using System.Text.Json;
using MLPortfolio.Web.Models;

namespace MLPortfolio.Web.Services;

public interface IProjectCatalog
{
    /// <summary>All projects, sorted by <see cref="ProjectMeta.Order"/> then title.</summary>
    IReadOnlyList<ProjectMeta> All { get; }

    IReadOnlyList<ProjectMeta> InCategory(string category);

    /// <summary>Distinct tags across all projects, sorted case-insensitively.</summary>
    IReadOnlyList<string> Tags { get; }

    /// <summary>Domains that at least one project uses, sorted by display label. Unused domains are not listed.</summary>
    IReadOnlyList<string> Domains { get; }

    /// <summary>Case-insensitive slug lookup; null when unknown.</summary>
    ProjectMeta? Find(string slug);

    /// <summary>Other projects in the same domain or sharing a tag, best matches first (at most <paramref name="max"/>).</summary>
    IReadOnlyList<ProjectMeta> Related(ProjectMeta project, int max = 3);
}

public sealed class ProjectCatalogException(string message) : Exception(message);

/// <summary>Immutable catalog loaded once from <c>projects.json</c> and validated up front.</summary>
public sealed class ProjectCatalog : IProjectCatalog
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    private readonly Dictionary<string, ProjectMeta> _bySlug;

    private ProjectCatalog(IEnumerable<ProjectMeta> projects)
    {
        All = projects.OrderBy(p => p.Order).ThenBy(p => p.Title, StringComparer.OrdinalIgnoreCase).ToList();
        _bySlug = All.ToDictionary(p => p.Slug, StringComparer.OrdinalIgnoreCase);
        Tags = All.SelectMany(p => p.Tags).Distinct(StringComparer.OrdinalIgnoreCase)
                  .OrderBy(t => t, StringComparer.OrdinalIgnoreCase).ToList();
        Domains = All.Select(p => p.Domain).OfType<string>().Distinct()
                     .OrderBy(d => ProjectDomains.LabelFor(d), StringComparer.OrdinalIgnoreCase).ToList();
    }

    public IReadOnlyList<ProjectMeta> All { get; }
    public IReadOnlyList<string> Tags { get; }
    public IReadOnlyList<string> Domains { get; }

    public IReadOnlyList<ProjectMeta> InCategory(string category) =>
        All.Where(p => p.Category == category).ToList();

    public ProjectMeta? Find(string slug) => _bySlug.GetValueOrDefault(slug);

    public IReadOnlyList<ProjectMeta> Related(ProjectMeta project, int max = 3) =>
        All.Where(p => p.Slug != project.Slug)
           .Select(p => (Project: p, Score:
               (p.Domain is not null && p.Domain == project.Domain ? 2 : 0) +
               p.Tags.Intersect(project.Tags, StringComparer.OrdinalIgnoreCase).Count()))
           .Where(x => x.Score > 0)
           .OrderByDescending(x => x.Score)
           .Select(x => x.Project)
           .Take(max)
           .ToList();

    public static ProjectCatalog LoadFile(string path)
    {
        if (!File.Exists(path))
            throw new ProjectCatalogException($"Project catalog not found: {path}");
        return Parse(File.ReadAllText(path), path);
    }

    public static ProjectCatalog Parse(string json, string source = "projects.json")
    {
        List<ProjectMeta>? projects;
        try
        {
            projects = JsonSerializer.Deserialize<List<ProjectMeta>>(json, JsonOptions);
        }
        catch (JsonException ex)
        {
            throw new ProjectCatalogException($"{source}: invalid JSON. {ex.Message}");
        }

        if (projects is null)
            throw new ProjectCatalogException($"{source}: expected a JSON array of projects.");

        var errors = Validate(projects);
        if (errors.Count > 0)
            throw new ProjectCatalogException($"{source} is invalid:{Environment.NewLine}- " +
                                              string.Join(Environment.NewLine + "- ", errors));
        return new ProjectCatalog(projects);
    }

    private static List<string> Validate(List<ProjectMeta> projects)
    {
        var errors = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < projects.Count; i++)
        {
            var p = projects[i];
            var who = string.IsNullOrWhiteSpace(p.Slug) ? $"entry #{i + 1}" : $"'{p.Slug}'";

            if (string.IsNullOrWhiteSpace(p.Slug)) errors.Add($"{who}: slug is required.");
            else if (!IsValidSlug(p.Slug)) errors.Add($"{who}: slug must be lowercase letters, digits and hyphens.");
            else if (!seen.Add(p.Slug)) errors.Add($"{who}: duplicate slug.");

            if (string.IsNullOrWhiteSpace(p.Title)) errors.Add($"{who}: title is required.");
            if (string.IsNullOrWhiteSpace(p.Summary)) errors.Add($"{who}: summary is required.");
            if (!ProjectCategories.All.Contains(p.Category))
                errors.Add($"{who}: category '{p.Category}' must be one of {string.Join(", ", ProjectCategories.All)}.");
            if (p.Domain is not null && !ProjectDomains.All.Contains(p.Domain))
                errors.Add($"{who}: domain '{p.Domain}' must be one of {string.Join(", ", ProjectDomains.All)}.");
            if (p.Category == ProjectCategories.Flagship && string.IsNullOrWhiteSpace(p.Domain))
                errors.Add($"{who}: domain is required for flagship projects.");
            if (!ProjectStatuses.All.Contains(p.Status))
                errors.Add($"{who}: status '{p.Status}' must be one of {string.Join(", ", ProjectStatuses.All)}.");
            if (!ProjectInferenceModes.All.Contains(p.InferenceMode))
                errors.Add($"{who}: inferenceMode '{p.InferenceMode}' must be one of {string.Join(", ", ProjectInferenceModes.All)}.");
        }
        return errors;
    }

    private static bool IsValidSlug(string slug) =>
        slug.All(c => c is (>= 'a' and <= 'z') or (>= '0' and <= '9') or '-') && !slug.StartsWith('-') && !slug.EndsWith('-');
}
