using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Tests;

public class ProjectCatalogTests
{
    private static string Entry(
        string slug = "demo", string title = "Demo", string category = "zoo", string summary = "A demo.",
        string status = "live", string inference = "onnx", int order = 1, string tags = "\"a\", \"b\"",
        string? domain = null)
    {
        // Flagship projects require a domain, so default one in for them.
        domain ??= category == "flagship" ? "finance" : null;
        var domainJson = domain is null ? "" : $"\"domain\": \"{domain}\", ";
        return $$"""
        { "slug": "{{slug}}", "title": "{{title}}", "category": "{{category}}", "summary": "{{summary}}",
          {{domainJson}}"tags": [{{tags}}], "status": "{{status}}", "inferenceMode": "{{inference}}", "order": {{order}} }
        """;
    }

    private static string Json(params string[] entries) => "[" + string.Join(",", entries) + "]";

    [Fact]
    public void Parse_ValidEntry_ExposesAllFields()
    {
        var catalog = ProjectCatalog.Parse(Json(Entry()));

        var p = Assert.Single(catalog.All);
        Assert.Equal("demo", p.Slug);
        Assert.Equal("Demo", p.Title);
        Assert.Equal("zoo", p.Category);
        Assert.Equal(["a", "b"], p.Tags);
        Assert.Equal("Live", p.StatusLabel);
    }

    [Fact]
    public void All_IsSortedByOrderThenTitle()
    {
        var catalog = ProjectCatalog.Parse(Json(
            Entry("c", "Charlie", order: 2),
            Entry("b", "Bravo", order: 1),
            Entry("a", "Alpha", order: 1)));

        Assert.Equal(["a", "b", "c"], catalog.All.Select(p => p.Slug));
    }

    [Fact]
    public void InCategory_FiltersByCategory()
    {
        var catalog = ProjectCatalog.Parse(Json(
            Entry("f", category: "flagship"), Entry("z", category: "zoo")));

        Assert.Equal("f", Assert.Single(catalog.InCategory(ProjectCategories.Flagship)).Slug);
        Assert.Equal("z", Assert.Single(catalog.InCategory(ProjectCategories.Zoo)).Slug);
    }

    [Fact]
    public void Tags_AreDistinctAndSorted()
    {
        var catalog = ProjectCatalog.Parse(Json(
            Entry("a", tags: "\"zeta\", \"Alpha\""), Entry("b", tags: "\"alpha\", \"mid\"")));

        Assert.Equal(["Alpha", "mid", "zeta"], catalog.Tags);
    }

    [Fact]
    public void Domains_ListsOnlyDomainsInUse_SortedByLabel()
    {
        var catalog = ProjectCatalog.Parse(Json(
            Entry("a", category: "flagship", domain: "finance"),
            Entry("b", category: "flagship", domain: "finance"),
            Entry("z")));

        Assert.Equal(["finance"], catalog.Domains);
        Assert.DoesNotContain("cybersecurity", catalog.Domains);
    }

    [Fact]
    public void Domains_WithBothDomainsInUse_AreSortedByLabel()
    {
        var catalog = ProjectCatalog.Parse(Json(
            Entry("a", category: "flagship", domain: "finance"),
            Entry("b", category: "flagship", domain: "cybersecurity")));

        Assert.Equal(["cybersecurity", "finance"], catalog.Domains);
    }

    [Fact]
    public void DomainLabel_IsDisplayNameOrNull()
    {
        var catalog = ProjectCatalog.Parse(Json(
            Entry("a", category: "flagship", domain: "cybersecurity"), Entry("z")));

        Assert.Equal("Cybersecurity", catalog.Find("a")!.DomainLabel);
        Assert.Null(catalog.Find("z")!.DomainLabel);
    }

    [Fact]
    public void Parse_UnknownDomain_Throws()
    {
        var ex = Assert.Throws<ProjectCatalogException>(() =>
            ProjectCatalog.Parse(Json(Entry(category: "flagship", domain: "healthcare"))));

        Assert.Contains("domain 'healthcare'", ex.Message);
    }

    [Fact]
    public void Parse_FlagshipWithoutDomain_Throws()
    {
        var json = """
            [{ "slug": "a", "title": "A", "category": "flagship", "summary": "s", "status": "live" }]
            """;

        var ex = Assert.Throws<ProjectCatalogException>(() => ProjectCatalog.Parse(json));

        Assert.Contains("domain is required for flagship", ex.Message);
    }

    [Fact]
    public void Find_IsCaseInsensitiveAndReturnsNullWhenUnknown()
    {
        var catalog = ProjectCatalog.Parse(Json(Entry("nse-regime")));

        Assert.NotNull(catalog.Find("NSE-Regime"));
        Assert.Null(catalog.Find("nope"));
    }

    [Theory]
    [InlineData("live", "Live")]
    [InlineData("notebook-only", "Notebook only")]
    [InlineData("coming-soon", "Coming soon")]
    public void StatusLabel_MapsEachStatus(string status, string label)
    {
        var catalog = ProjectCatalog.Parse(Json(Entry(status: status)));

        Assert.Equal(label, catalog.All[0].StatusLabel);
    }

    [Fact]
    public void ThumbnailPath_IsNullWithoutThumbnail_AndSlugScopedWithOne()
    {
        var without = ProjectCatalog.Parse(Json(Entry("a"))).All[0];
        var with = ProjectCatalog.Parse(Json(
            """{ "slug": "a", "title": "A", "category": "zoo", "summary": "s", "status": "live", "thumbnail": "charts/t.svg" }""")).All[0];

        Assert.Null(without.ThumbnailPath);
        Assert.Equal("~/projects/a/charts/t.svg", with.ThumbnailPath);
    }

    [Fact]
    public void Parse_AllowsCommentsAndTrailingCommas()
    {
        var json = """
            [
              // a comment
              { "slug": "a", "title": "A", "category": "zoo", "summary": "s", "status": "live", },
            ]
            """;

        Assert.Single(ProjectCatalog.Parse(json).All);
    }

    [Theory]
    [InlineData("slug", "", "slug is required")]
    [InlineData("slug", "Bad Slug", "slug must be lowercase")]
    [InlineData("title", "", "title is required")]
    [InlineData("summary", "", "summary is required")]
    [InlineData("category", "finance", "category 'finance'")]
    [InlineData("status", "beta", "status 'beta'")]
    [InlineData("inference", "magic", "inferenceMode 'magic'")]
    public void Parse_InvalidField_ThrowsWithReadableMessage(string field, string value, string expected)
    {
        var entry = field switch
        {
            "slug" => Entry(slug: value),
            "title" => Entry(title: value),
            "summary" => Entry(summary: value),
            "category" => Entry(category: value),
            "status" => Entry(status: value),
            _ => Entry(inference: value),
        };

        var ex = Assert.Throws<ProjectCatalogException>(() => ProjectCatalog.Parse(Json(entry)));
        Assert.Contains(expected, ex.Message);
    }

    [Fact]
    public void Parse_DuplicateSlug_Throws()
    {
        var ex = Assert.Throws<ProjectCatalogException>(() =>
            ProjectCatalog.Parse(Json(Entry("same"), Entry("same"))));

        Assert.Contains("duplicate slug", ex.Message);
    }

    [Fact]
    public void Parse_ReportsEveryProblemAtOnce()
    {
        var ex = Assert.Throws<ProjectCatalogException>(() =>
            ProjectCatalog.Parse(Json(Entry(title: ""), Entry("b", status: "beta"))));

        Assert.Contains("title is required", ex.Message);
        Assert.Contains("status 'beta'", ex.Message);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("{}")]
    [InlineData("null")]
    public void Parse_NonArrayOrBrokenJson_Throws(string json) =>
        Assert.Throws<ProjectCatalogException>(() => ProjectCatalog.Parse(json));

    [Fact]
    public void LoadFile_MissingFile_Throws() =>
        Assert.Throws<ProjectCatalogException>(() => ProjectCatalog.LoadFile("does-not-exist.json"));
}
