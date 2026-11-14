using Microsoft.AspNetCore.Mvc.Testing;

namespace MLPortfolio.Tests;

public class HomePageTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client = factory.CreateClient(new WebApplicationFactoryClientOptions
    {
        AllowAutoRedirect = false,
    });

    private async Task<string> GetHomeAsync()
    {
        var response = await _client.GetAsync("/");
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    [Fact]
    public async Task Home_ReturnsHtmlWithTitleAndDescription()
    {
        var html = await GetHomeAsync();

        // The separator is HTML-encoded by Razor, so match around it.
        Assert.Matches(@"<title>Kartik .+ ML Portfolio</title>", html);
        Assert.Contains("<meta name=\"description\"", html);
        Assert.Contains("property=\"og:title\"", html);
    }

    [Fact]
    public async Task Home_ShowsBothSections()
    {
        var html = await GetHomeAsync();

        Assert.Contains("Flagship ML", html);
        Assert.Contains("Model Zoo", html);
        Assert.DoesNotContain("Finance &amp; Fintech", html);
    }

    [Theory]
    [InlineData("NSE Market Regime Classifier", "/projects/nse-regime")]
    [InlineData("Logistic Regression", "/projects/logistic-regression")]
    [InlineData("Linear Regression", "/projects/linear-regression")]
    [InlineData("Decision Tree", "/projects/decision-tree")]
    [InlineData("Random Forest", "/projects/random-forest")]
    [InlineData("Support Vector Machine", "/projects/svm")]
    [InlineData("k-Nearest Neighbours", "/projects/knn")]
    [InlineData("Naive Bayes", "/projects/naive-bayes")]
    public async Task Home_ListsEachCatalogProjectWithLink(string title, string href)
    {
        var html = await GetHomeAsync();

        Assert.Contains(title, html);
        Assert.Contains($"href=\"{href}\"", html);
    }

    [Fact]
    public async Task Home_ShowsStatusBadgesAndTagChips()
    {
        var html = await GetHomeAsync();

        Assert.Contains("Coming soon", html);
        Assert.Contains("data-kind=\"tag\" data-value=\"classification\"", html);
        Assert.Contains("id=\"project-search\"", html);
    }

    [Fact]
    public async Task Home_ShowsFinanceDomainPill_ButNoCybersecurityOrNsePill()
    {
        var html = await GetHomeAsync();

        Assert.Contains("data-kind=\"domain\" data-value=\"finance\"", html);
        Assert.DoesNotContain("data-value=\"cybersecurity\"", html);
        Assert.DoesNotContain("data-value=\"nse\"", html);
    }

    [Fact]
    public async Task Home_TagsFlagshipCardWithItsDomain()
    {
        var html = await GetHomeAsync();

        Assert.Contains("data-domain=\"finance\"", html);
        Assert.Contains("class=\"domain-label\">Finance<", html);
    }

    [Fact]
    public async Task Home_HasThemeToggleAndSkipLink()
    {
        var html = await GetHomeAsync();

        Assert.Contains("id=\"theme-toggle\"", html);
        Assert.Contains("Skip to content", html);
    }

    [Fact]
    public async Task Home_OmitsSocialLinksWhenNotConfigured()
    {
        var html = await GetHomeAsync();

        Assert.DoesNotContain(">GitHub<", html);
        Assert.DoesNotContain(">LinkedIn<", html);
    }

    [Theory]
    [InlineData("/css/site.css")]
    [InlineData("/js/site.js")]
    [InlineData("/js/home.js")]
    [InlineData("/projects/logistic-regression/charts/thumbnail.png")]
    [InlineData("/projects/linear-regression/charts/thumbnail.png")]
    [InlineData("/projects/decision-tree/charts/thumbnail.png")]
    [InlineData("/projects/random-forest/charts/thumbnail.png")]
    [InlineData("/projects/svm/charts/thumbnail.png")]
    [InlineData("/projects/knn/charts/thumbnail.png")]
    [InlineData("/projects/naive-bayes/charts/thumbnail.png")]
    [InlineData("/projects/nse-regime/charts/thumbnail.svg")]
    public async Task StaticAssets_AreServed(string path)
    {
        var response = await _client.GetAsync(path);

        Assert.Equal(System.Net.HttpStatusCode.OK, response.StatusCode);
    }
}
