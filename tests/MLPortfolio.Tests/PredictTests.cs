using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Tests;

/// <summary>Parity (Python vs .NET), the /predict endpoint, and rate limiting.</summary>
public class PredictTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public PredictTests(WebApplicationFactory<Program> factory)
    {
        // A high limit so the many requests in this class never trip rate limiting by accident.
        _factory = factory.WithWebHostBuilder(b => b.UseSetting("RateLimiting:PredictPermitsPerMinute", "100000"));
    }

    // ---------------------------------------------------------------------------------------------
    // Parity: reference inputs/outputs produced by ml/<slug>/train.py with scikit-learn
    // ---------------------------------------------------------------------------------------------

    private static JsonElement LoadFixtures(string slug) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", slug + ".json"))).RootElement;

    private PredictionResult RunDotNet(string slug, JsonElement input)
    {
        var schema = _factory.Services.GetRequiredService<ISchemaStore>().Get(slug)!;
        var validation = InputValidator.Validate(schema, input);
        Assert.True(validation.IsValid, string.Join("; ", validation.Errors.Select(e => $"{e.Key}: {e.Value}")));
        return _factory.Services.GetRequiredService<IModelRunner>().Predict(slug, schema, validation.Values);
    }

    public static TheoryData<string> ClassificationSlugs => new()
    {
        "logistic-regression", "decision-tree", "svm", "knn", "naive-bayes",
    };

    public static TheoryData<string> RegressionSlugs => new()
    {
        "linear-regression", "random-forest",
    };

    [Theory]
    [MemberData(nameof(ClassificationSlugs))]
    public void Parity_Classifier_MatchesPythonOnReferenceInputs(string slug)
    {
        var cases = LoadFixtures(slug).EnumerateArray().ToList();
        Assert.True(cases.Count >= 5, "need at least 5 reference inputs");

        foreach (var c in cases)
        {
            var result = RunDotNet(slug, c.GetProperty("input"));
            var expected = c.GetProperty("probabilities").EnumerateObject().ToList();

            // SVM probabilities come from Platt scaling and differ slightly in float32.
            var precision = slug == "svm" ? 2 : 4;
            foreach (var e in expected)
                Assert.Equal(e.Value.GetDouble(), result.Probabilities![e.Name], precision);

            // The predicted class must match unless the top two classes are a near tie.
            var sorted = expected.Select(e => e.Value.GetDouble()).OrderByDescending(v => v).ToList();
            if (sorted[0] - sorted[1] > 0.01)
                Assert.Equal(c.GetProperty("label").GetString(), result.Label);
        }
    }

    [Theory]
    [MemberData(nameof(RegressionSlugs))]
    public void Parity_Regressor_MatchesPythonOnReferenceInputs(string slug)
    {
        var cases = LoadFixtures(slug).EnumerateArray().ToList();
        Assert.True(cases.Count >= 5, "need at least 5 reference inputs");

        foreach (var c in cases)
        {
            var result = RunDotNet(slug, c.GetProperty("input"));

            Assert.Equal(c.GetProperty("prediction").GetDouble(), result.Prediction, precision: 3);
        }
    }

    [Theory]
    [MemberData(nameof(ClassificationSlugs))]
    [MemberData(nameof(RegressionSlugs))]
    public async Task EveryPreset_PredictsSuccessfullyThroughTheEndpoint(string slug)
    {
        var schema = _factory.Services.GetRequiredService<ISchemaStore>().Get(slug)!;
        Assert.NotEmpty(schema.Presets);

        foreach (var preset in schema.Presets)
        {
            var json = JsonSerializer.Serialize(preset.Values);
            var response = await _factory.CreateClient().PostAsync($"/predict/{slug}", Json(json));

            Assert.True(response.StatusCode == HttpStatusCode.OK, $"{slug} / {preset.Name}: {response.StatusCode}");
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Endpoint
    // ---------------------------------------------------------------------------------------------

    private static StringContent Json(string json) => new(json, Encoding.UTF8, "application/json");

    private const string ValidLogistic = """
        { "annual_income_k": 120, "debt_to_income_pct": 15, "credit_history_years": 15, "late_payments_12m": 0, "employment": "salaried" }
        """;

    [Fact]
    public async Task Classification_ReturnsLabelProbabilitiesExplanationAndVersion()
    {
        var response = await _factory.CreateClient().PostAsync("/predict/logistic-regression", Json(ValidLogistic));
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("Repaid", body.GetProperty("label").GetString());   // the "Low risk" preset
        var probs = body.GetProperty("probabilities");
        Assert.Equal(1.0, probs.GetProperty("Repaid").GetDouble() + probs.GetProperty("Default").GetDouble(), precision: 3);
        Assert.Matches(@"most likely outcome is Repaid \(\d+\.\d% probability\)\.", body.GetProperty("explanation").GetString());
        Assert.Equal("1.0.0", body.GetProperty("modelVersion").GetString());
        Assert.Equal(0, body.GetProperty("prediction").GetInt32());
    }

    [Fact]
    public async Task Regression_ReturnsFormattedLabelWithUnit()
    {
        var json = """{ "market_return_pct": 2.1, "volume_ratio": 1.6 }""";

        var response = await _factory.CreateClient().PostAsync("/predict/linear-regression", Json(json));
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(2.5442, body.GetProperty("prediction").GetDouble(), precision: 2);
        Assert.Equal("+2.54%", body.GetProperty("label").GetString());
        Assert.Equal(JsonValueKind.Null, body.GetProperty("probabilities").ValueKind);
    }

    [Fact]
    public async Task InvalidInput_Returns400WithPerFieldErrors()
    {
        var json = """{ "annual_income_k": 5000, "debt_to_income_pct": "high", "credit_history_years": 3, "late_payments_12m": 0, "employment": "pirate" }""";

        var response = await _factory.CreateClient().PostAsync("/predict/logistic-regression", Json(json));
        var errors = (await response.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("errors");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("at most 300", errors.GetProperty("annual_income_k").GetString());
        Assert.Contains("Must be a number", errors.GetProperty("debt_to_income_pct").GetString());
        Assert.Contains("Must be one of", errors.GetProperty("employment").GetString());
        Assert.False(errors.TryGetProperty("credit_history_years", out _));
    }

    [Theory]
    [InlineData("{ not json")]
    [InlineData("[1,2,3]")]
    [InlineData("null")]
    public async Task MalformedOrNonObjectBody_Returns400(string body)
    {
        var response = await _factory.CreateClient().PostAsync("/predict/logistic-regression", Json(body));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task NonJsonContentType_IsRejected()
    {
        var content = new StringContent(ValidLogistic, Encoding.UTF8, "text/plain");

        var response = await _factory.CreateClient().PostAsync("/predict/logistic-regression", content);

        Assert.Equal(HttpStatusCode.UnsupportedMediaType, response.StatusCode);
    }

    [Theory]
    [InlineData("/predict/does-not-exist")]
    [InlineData("/predict/nse-regime")]          // coming soon, inferenceMode none
    public async Task UnknownOrNotLiveProject_Returns404WithJsonBody(string url)
    {
        var response = await _factory.CreateClient().PostAsync(url, Json("{}"));
        var body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Contains("No live model", body);   // JSON error, not the styled HTML 404 page
        Assert.DoesNotContain("<html", body);
    }

    [Fact]
    public async Task GetOnPredict_IsNotAllowed()
    {
        var response = await _factory.CreateClient().GetAsync("/predict/logistic-regression");

        Assert.True(response.StatusCode is HttpStatusCode.MethodNotAllowed or HttpStatusCode.NotFound);
    }

    private sealed class ThrowingRunner : IModelRunner
    {
        public PredictionResult Predict(string slug, FormSchema schema, IReadOnlyDictionary<string, object> values) =>
            throw new ModelUnavailableException("boom");
    }

    [Fact]
    public async Task ModelFailure_Returns503WithoutLeakingDetails()
    {
        using var broken = _factory.WithWebHostBuilder(b => b.ConfigureServices(s =>
            s.Replace(ServiceDescriptor.Singleton<IModelRunner>(new ThrowingRunner()))));

        var response = await broken.CreateClient().PostAsync("/predict/logistic-regression", Json(ValidLogistic));
        var text = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Contains("unavailable", text);
        Assert.DoesNotContain("boom", text);
    }

    [Fact]
    public async Task MissingModelFile_Returns503()
    {
        // A live project whose schema exists but whose model file does not.
        var json = """
            [{ "slug": "logistic-regression", "title": "L", "category": "zoo", "summary": "s", "status": "live", "inferenceMode": "onnx" }]
            """;
        using var custom = _factory.WithWebHostBuilder(b => b.ConfigureServices(s =>
        {
            s.Replace(ServiceDescriptor.Singleton<IProjectCatalog>(ProjectCatalog.Parse(json)));
            s.Replace(ServiceDescriptor.Singleton<ISchemaStore>(new FakeSchemaStore("does-not-exist.onnx")));
        }));

        var response = await custom.CreateClient().PostAsync("/predict/logistic-regression", Json(ValidLogistic));

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
    }

    private sealed class FakeSchemaStore(string modelFile) : ISchemaStore
    {
        public FormSchema? Get(string slug)
        {
            var real = SchemaStore.Parse(File.ReadAllText(Path.Combine(
                AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "MLPortfolio.Web", "App_Data", "models", slug, "schema.json")));
            return real with { ModelFile = modelFile };
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Rate limiting
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public async Task RateLimit_Returns429AfterThePermitIsUsed()
    {
        using var limited = _factory.WithWebHostBuilder(b => b.UseSetting("RateLimiting:PredictPermitsPerMinute", "2"));
        var client = limited.CreateClient();

        var statuses = new List<HttpStatusCode>();
        for (var i = 0; i < 4; i++)
            statuses.Add((await client.PostAsync("/predict/logistic-regression", Json(ValidLogistic))).StatusCode);

        Assert.Equal([HttpStatusCode.OK, HttpStatusCode.OK, HttpStatusCode.TooManyRequests, HttpStatusCode.TooManyRequests], statuses);
    }

    [Fact]
    public async Task RateLimit_DoesNotApplyToPages()
    {
        using var limited = _factory.WithWebHostBuilder(b => b.UseSetting("RateLimiting:PredictPermitsPerMinute", "1"));
        var client = limited.CreateClient();

        for (var i = 0; i < 4; i++)
            Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/")).StatusCode);
    }
}
