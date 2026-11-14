using System.Net;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Tests;

public class ProjectPageTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>>
{
    private static readonly WebApplicationFactoryClientOptions NoRedirect = new() { AllowAutoRedirect = false };

    private HttpClient Client(WebApplicationFactory<Program>? custom = null) =>
        (custom ?? factory).CreateClient(NoRedirect);

    private static async Task<(HttpStatusCode Status, string Html)> GetAsync(HttpClient client, string path)
    {
        var response = await client.GetAsync(path);
        return (response.StatusCode, await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task KnownProject_ReturnsPageWithHeaderSummaryAndLinks()
    {
        var (status, html) = await GetAsync(Client(), "/projects/logistic-regression");

        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Contains("<h1 class=\"project-page-title\">Logistic Regression</h1>", html);
        Assert.Contains("Summary</h2>", html);
        Assert.Contains(">Problem</h2>", html);   // rendered from content/logistic-regression.md
        Assert.Contains("status-live", html);
        Assert.Contains("href=\"/projects/linear-regression\"", html);  // related (shares tag "linear models")
    }

    [Fact]
    public async Task Slug_IsCaseInsensitive()
    {
        var (status, _) = await GetAsync(Client(), "/projects/Logistic-Regression");

        Assert.Equal(HttpStatusCode.OK, status);
    }

    [Fact]
    public async Task UnknownProject_Returns404WithStyledPage()
    {
        var (status, html) = await GetAsync(Client(), "/projects/does-not-exist");

        Assert.Equal(HttpStatusCode.NotFound, status);
        Assert.Contains("couldn't find that page", html);
        Assert.Contains("Browse all projects", html);
    }

    [Fact]
    public async Task UnknownUrl_Returns404WithStyledPage()
    {
        var (status, html) = await GetAsync(Client(), "/no/such/page");

        Assert.Equal(HttpStatusCode.NotFound, status);
        Assert.Contains("couldn't find that page", html);
    }

    [Fact]
    public async Task PathTraversalSlug_IsRejectedAs404()
    {
        var (status, _) = await GetAsync(Client(), "/projects/..%2F..%2Fappsettings");

        Assert.Equal(HttpStatusCode.NotFound, status);
    }

    [Fact]
    public async Task LivePilot_ShowsEnabledFormWithPresetsAndCharts()
    {
        var (_, html) = await GetAsync(Client(), "/projects/logistic-regression");

        Assert.Contains("Try the model</h2>", html);
        Assert.DoesNotContain("isn't live yet", html);
        Assert.DoesNotContain("<fieldset class=\"try-fields\" disabled", html);
        Assert.Contains("name=\"annual_income_k\"", html);
        Assert.Contains("Low risk", html);   // preset button
        Assert.Contains("<option value=\"self-employed\"", html);
        Assert.Contains("Visualizations</h2>", html);
        Assert.Contains("/projects/logistic-regression/charts/roc-curve.png", html);
    }

    [Fact]
    public async Task NumericPilot_ShowsItsOwnForm()
    {
        var (_, html) = await GetAsync(Client(), "/projects/linear-regression");

        Assert.Contains("name=\"market_return_pct\"", html);
        Assert.Contains("name=\"volume_ratio\"", html);
        Assert.Contains("Strong market day", html);
    }

    [Fact]
    public async Task NotLiveProjectWithSchema_ShowsFormAsDisabledPreview()
    {
        var json = """
            [{ "slug": "logistic-regression", "title": "Logistic Regression", "category": "zoo", "summary": "s",
               "status": "coming-soon", "inferenceMode": "onnx" }]
            """;
        using var custom = WithCatalog(json);

        var (_, html) = await GetAsync(Client(custom), "/projects/logistic-regression");

        Assert.Contains("isn't live yet", html);
        Assert.Contains("<fieldset class=\"try-fields\" disabled", html);
    }

    [Fact]
    public async Task OnnxProjectWithoutSchemaFile_RendersPageWithoutForm()
    {
        var json = """
            [{ "slug": "no-schema", "title": "No Schema", "category": "zoo", "summary": "s",
               "status": "live", "inferenceMode": "onnx" }]
            """;
        using var custom = WithCatalog(json);

        var (status, html) = await GetAsync(Client(custom), "/projects/no-schema");

        Assert.Equal(HttpStatusCode.OK, status);
        Assert.DoesNotContain("Try the model", html);
        Assert.DoesNotContain("id=\"try-form\"", html);
    }

    [Fact]
    public async Task FinanceProject_ShowsDisclaimerAndNoForm()
    {
        var (_, html) = await GetAsync(Client(), "/projects/nse-regime");

        Assert.Contains("Educational purposes only. Not investment advice.", html);
        Assert.DoesNotContain("Try the model", html);   // inferenceMode: none
        Assert.Contains("Finance", html);
    }

    [Theory]
    [InlineData("logistic-regression")]
    [InlineData("linear-regression")]
    [InlineData("decision-tree")]
    [InlineData("random-forest")]
    [InlineData("svm")]
    [InlineData("knn")]
    [InlineData("naive-bayes")]
    public async Task FinancePilots_ShowDisclaimer(string slug)
    {
        var (_, html) = await GetAsync(Client(), "/projects/" + slug);

        Assert.Contains("Educational purposes only. Not investment advice.", html);
    }

    private WebApplicationFactory<Program> WithCatalog(string json) =>
        factory.WithWebHostBuilder(b => b.ConfigureServices(services =>
            services.Replace(ServiceDescriptor.Singleton<IProjectCatalog>(ProjectCatalog.Parse(json)))));

    [Fact]
    public async Task LiveProject_EnablesFormAndShowsChartGallery()
    {
        var json = """
            [{ "slug": "logistic-regression", "title": "Logistic Regression", "category": "zoo", "summary": "s",
               "tags": ["classification"], "status": "live", "inferenceMode": "onnx",
               "repoUrl": "https://example.com/repo", "notebookUrl": "https://example.com/nb",
               "charts": [{ "file": "roc-curve.png", "caption": "Decision curve" }] }]
            """;
        using var custom = WithCatalog(json);

        var (status, html) = await GetAsync(Client(custom), "/projects/logistic-regression");

        Assert.Equal(HttpStatusCode.OK, status);
        Assert.DoesNotContain("isn't live yet", html);
        Assert.DoesNotContain("<fieldset class=\"try-fields\" disabled", html);
        Assert.Contains("Visualizations</h2>", html);
        Assert.Contains("<figcaption>Decision curve</figcaption>", html);
        Assert.Contains("alt=\"Decision curve\"", html);
        Assert.Contains("href=\"https://example.com/repo\"", html);
        Assert.Contains("View on GitHub", html);
        Assert.Contains("Notebook</a>", html);
    }

    [Fact]
    public async Task ProjectWithoutContentOrCharts_StillRenders()
    {
        var json = """
            [{ "slug": "bare", "title": "Bare", "category": "zoo", "summary": "s", "status": "live", "inferenceMode": "none" }]
            """;
        using var custom = WithCatalog(json);

        var (status, html) = await GetAsync(Client(custom), "/projects/bare");

        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Contains("hasn't been published yet", html);
        Assert.DoesNotContain("Visualizations", html);
        Assert.DoesNotContain("Not investment advice", html);   // no disclaimer unless the project sets one
    }

    [Fact]
    public async Task ProjectScript_IsServed()
    {
        var response = await Client().GetAsync("/js/project.js");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
