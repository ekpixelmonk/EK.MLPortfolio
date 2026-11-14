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
        Assert.Matches(@"<title>E. Kartik .+ ML Portfolio</title>", html);
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

    [Fact]
    public async Task Home_FilterPills_ExcludeOneOffTags()
    {
        var html = await GetHomeAsync();

        Assert.Contains("data-kind=\"tag\" data-value=\"classification\"", html);   // shared by many projects
        Assert.Contains("data-kind=\"tag\" data-value=\"trees\"", html);            // decision tree + random forest
        Assert.DoesNotContain("data-kind=\"tag\" data-value=\"fraud\"", html);      // only the SVM
        Assert.DoesNotContain("data-kind=\"tag\" data-value=\"volatility\"", html);
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

        Assert.Contains("status-live", html);
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
    public async Task Home_EachSectionHasAShortDescription()
    {
        var html = await GetHomeAsync();

        Assert.Contains("class=\"section-desc\">In-depth projects on real-world data", html);
        Assert.Contains("class=\"section-desc\">Short, focused demos of classic ML algorithms", html);
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

    [Fact]
    public async Task Home_NoSocialIconsWhenNothingIsConfigured()
    {
        using var custom = factory.WithWebHostBuilder(b =>
        {
            foreach (var key in new[] { "GitHubUrl", "LinkedInUrl", "TwitterUrl", "MediumUrl", "Email" })
                b.UseSetting($"Site:{key}", "");
        });

        var html = await (await custom.CreateClient().GetAsync("/")).Content.ReadAsStringAsync();

        Assert.DoesNotContain("class=\"social-links\"", html);
    }

    [Fact]
    public async Task Home_ShowsSocialIconsInHeaderAndFooter_OnlyForConfiguredLinks()
    {
        using var custom = factory.WithWebHostBuilder(b =>
        {
            b.UseSetting("Site:LinkedInUrl", "https://www.linkedin.com/in/example");
            b.UseSetting("Site:TwitterUrl", "https://x.com/example");
            b.UseSetting("Site:Email", "someone@example.com");
            b.UseSetting("Site:MediumUrl", "");
            b.UseSetting("Site:GitHubUrl", "");
        });

        var html = await (await custom.CreateClient().GetAsync("/")).Content.ReadAsStringAsync();

        Assert.Equal(2, System.Text.RegularExpressions.Regex.Matches(html, "class=\"social-links\"").Count);   // navbar and footer
        Assert.Contains("href=\"https://www.linkedin.com/in/example\"", html);
        Assert.Contains("aria-label=\"X (Twitter)\"", html);
        Assert.Contains("href=\"mailto:someone@example.com\"", html);
        Assert.DoesNotContain("aria-label=\"Medium\"", html);      // no URL configured
        Assert.DoesNotContain("aria-label=\"GitHub\"", html);
        var header = html[html.IndexOf("<header", StringComparison.Ordinal)..html.IndexOf("</header>", StringComparison.Ordinal)];
        Assert.Contains("social-links", header);
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
    [InlineData("/projects/nse-regime/charts/thumbnail.png")]
    public async Task StaticAssets_AreServed(string path)
    {
        var response = await _client.GetAsync(path);

        Assert.Equal(System.Net.HttpStatusCode.OK, response.StatusCode);
    }
}
