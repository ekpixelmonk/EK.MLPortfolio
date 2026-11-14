using System.Text.Json;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Tests;

public class InputValidatorTests
{
    private static readonly Web.Models.FormSchema Schema = SchemaStore.Parse("""
        {
          "modelFile": "model.onnx", "modelVersion": "1.0.0",
          "inputs": [
            { "name": "vol", "label": "Vol", "type": "number", "min": 0, "max": 100, "default": 10 },
            { "name": "gap", "label": "Gap", "type": "select", "options": ["none", "up", "down"], "default": "none" },
            { "name": "extra", "label": "Extra", "type": "number", "default": 5, "required": false }
          ],
          "outputs": { "type": "regression" }
        }
        """);

    private static ValidationResult Validate(string json) =>
        InputValidator.Validate(Schema, JsonDocument.Parse(json).RootElement);

    [Fact]
    public void ValidInput_ProducesTypedValues()
    {
        var r = Validate("""{ "vol": 14.5, "gap": "up", "extra": 1 }""");

        Assert.True(r.IsValid);
        Assert.Equal(14.5, r.Values["vol"]);
        Assert.Equal("up", r.Values["gap"]);
        Assert.Equal(1.0, r.Values["extra"]);
    }

    [Fact]
    public void MissingOptionalField_FallsBackToDefault()
    {
        var r = Validate("""{ "vol": 1, "gap": "none" }""");

        Assert.True(r.IsValid);
        Assert.Equal(5.0, r.Values["extra"]);
    }

    [Fact]
    public void MissingRequiredFields_AreReportedTogether()
    {
        var r = Validate("{}");

        Assert.False(r.IsValid);
        Assert.Equal("This field is required.", r.Errors["vol"]);
        Assert.Equal("This field is required.", r.Errors["gap"]);
        Assert.DoesNotContain("extra", r.Errors.Keys);
    }

    [Theory]
    [InlineData("""{ "vol": -1, "gap": "none" }""", "vol", "at least 0")]
    [InlineData("""{ "vol": 101, "gap": "none" }""", "vol", "at most 100")]
    [InlineData("""{ "vol": "12", "gap": "none" }""", "vol", "Must be a number")]
    [InlineData("""{ "vol": null, "gap": "none" }""", "vol", "required")]
    [InlineData("""{ "vol": 1, "gap": "sideways" }""", "gap", "Must be one of")]
    [InlineData("""{ "vol": 1, "gap": 3 }""", "gap", "Must be one of")]
    [InlineData("""{ "vol": 1, "gap": "none", "surprise": 1 }""", "surprise", "Unknown field")]
    public void InvalidField_IsRejectedWithMessage(string json, string field, string expected)
    {
        var r = Validate(json);

        Assert.False(r.IsValid);
        Assert.Contains(expected, r.Errors[field]);
    }

    [Theory]
    [InlineData("[1, 2]")]
    [InlineData("\"text\"")]
    [InlineData("42")]
    public void NonObjectBody_IsRejected(string json)
    {
        var r = Validate(json);

        Assert.False(r.IsValid);
        Assert.Contains(InputValidator.RequestKey, r.Errors.Keys);
    }

    [Fact]
    public void UndefinedBody_IsRejected()
    {
        var r = InputValidator.Validate(Schema, default);

        Assert.Contains(InputValidator.RequestKey, r.Errors.Keys);
    }
}
