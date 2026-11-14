using System.Collections.Concurrent;
using Markdig;

namespace MLPortfolio.Web.Services;

public interface IProjectContentStore
{
    /// <summary>HTML rendered from <c>App_Data/content/&lt;slug&gt;.md</c>, or null when there is no write-up.</summary>
    string? GetHtml(string slug);
}

public sealed class ProjectContentStore(IWebHostEnvironment env) : IProjectContentStore
{
    // Raw HTML in Markdown is disabled so a write-up can never inject markup or script.
    private static readonly MarkdownPipeline Pipeline =
        new MarkdownPipelineBuilder().UseAdvancedExtensions().DisableHtml().Build();

    private readonly ConcurrentDictionary<string, string?> _cache = new(StringComparer.OrdinalIgnoreCase);

    /// <remarks>The slug must already have been validated against the catalog; it is never taken from user input directly.</remarks>
    public string? GetHtml(string slug) => _cache.GetOrAdd(slug, Load);

    private string? Load(string slug)
    {
        var path = Path.Combine(env.ContentRootPath, "App_Data", "content", slug + ".md");
        return File.Exists(path) ? Render(File.ReadAllText(path)) : null;
    }

    public static string Render(string markdown) => Markdown.ToHtml(markdown, Pipeline);
}
