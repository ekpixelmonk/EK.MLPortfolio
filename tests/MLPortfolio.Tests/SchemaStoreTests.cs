using MLPortfolio.Web.Services;

namespace MLPortfolio.Tests;

public class SchemaStoreTests
{
    private const string Valid = """
        {
          "modelFile": "model.onnx", "modelVersion": "1.0.0",
          "inputs": [
            { "name": "vol", "label": "Volatility", "type": "number", "min": 0, "max": 100, "default": 14.5, "help": "Annualised" },
            { "name": "gap", "label": "Gap", "type": "select", "options": ["none", "up", "down"], "default": "none" }
          ],
          "outputs": { "type": "classification", "labels": ["Calm", "Choppy"] },
          "presets": [ { "name": "Crisis", "values": { "vol": 45, "gap": "down" } } ]
        }
        """;

    private static string With(string find, string replace) => Valid.Replace(find, replace);

    [Fact]
    public void Parse_ValidSchema_ExposesInputsOutputsAndPresets()
    {
        var s = SchemaStore.Parse(Valid);

        Assert.Equal("1.0.0", s.ModelVersion);
        Assert.Equal(2, s.Inputs.Count);
        Assert.True(s.Inputs[0].IsNumber);
        Assert.Equal("14.5", s.Inputs[0].DefaultText);
        Assert.True(s.Inputs[1].IsSelect);
        Assert.Equal("none", s.Inputs[1].DefaultText);
        Assert.Equal(["Calm", "Choppy"], s.Outputs.Labels);
        Assert.Equal("Crisis", Assert.Single(s.Presets).Name);
    }

    [Fact]
    public void Parse_AllowsCommentsAndTrailingCommas()
    {
        var json = With("{ \"modelFile\"", "// note\n{ \"modelFile\"").Replace("\"Choppy\"]", "\"Choppy\",]");

        Assert.Equal(2, SchemaStore.Parse(json).Inputs.Count);
    }

    [Fact]
    public void Parse_RegressionOutput_NeedsNoLabels()
    {
        var json = With("{ \"type\": \"classification\", \"labels\": [\"Calm\", \"Choppy\"] }",
                        "{ \"type\": \"regression\", \"unit\": \"%\" }");

        Assert.Equal("%", SchemaStore.Parse(json).Outputs.Unit);
    }

    [Theory]
    [InlineData("\"min\": 0, \"max\": 100", "\"min\": 100, \"max\": 0", "min is greater than max")]
    [InlineData("\"default\": 14.5", "\"default\": 500", "default is not valid")]
    [InlineData("\"default\": \"none\"", "\"default\": \"sideways\"", "default is not valid")]
    [InlineData("\"options\": [\"none\", \"up\", \"down\"]", "\"options\": []", "select inputs need options")]
    [InlineData("\"type\": \"select\"", "\"type\": \"slider\"", "type 'slider'")]
    [InlineData("\"name\": \"gap\"", "\"name\": \"vol\"", "duplicate name")]
    [InlineData("\"name\": \"gap\"", "\"name\": \"bad name\"", "must be an identifier")]
    [InlineData("\"type\": \"classification\"", "\"type\": \"clustering\"", "outputs.type 'clustering'")]
    [InlineData("[\"Calm\", \"Choppy\"]", "[\"Calm\"]", "at least two labels")]
    [InlineData("\"vol\": 45", "\"vol\": 450", "invalid value for 'vol'")]
    [InlineData("\"vol\": 45", "\"nope\": 45", "unknown input 'nope'")]
    public void Parse_InvalidSchema_ThrowsWithReadableMessage(string find, string replace, string expected)
    {
        var ex = Assert.Throws<SchemaException>(() => SchemaStore.Parse(With(find, replace)));

        Assert.Contains(expected, ex.Message);
    }

    [Fact]
    public void Parse_NoInputs_Throws()
    {
        var json = """{ "inputs": [], "outputs": { "type": "regression" } }""";

        var ex = Assert.Throws<SchemaException>(() => SchemaStore.Parse(json));

        Assert.Contains("at least one input", ex.Message);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("null")]
    public void Parse_BrokenJson_Throws(string json) =>
        Assert.Throws<SchemaException>(() => SchemaStore.Parse(json));
}
