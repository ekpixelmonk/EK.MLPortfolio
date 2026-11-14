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
}
