using System.Globalization;
using System.Text.Json;

namespace MLPortfolio.Web.Models;

/// <summary>Contract between a Python training project and the site: <c>App_Data/models/&lt;slug&gt;/schema.json</c>.</summary>
public sealed record FormSchema
{
    public string ModelFile { get; init; } = "";
    public string ModelVersion { get; init; } = "";
    public IReadOnlyList<SchemaInput> Inputs { get; init; } = [];
    public SchemaOutputs Outputs { get; init; } = new();
    public IReadOnlyList<SchemaPreset> Presets { get; init; } = [];
}

public sealed record SchemaInput
{
    public string Name { get; init; } = "";
    public string Label { get; init; } = "";
    /// <summary><c>number</c> or <c>select</c>.</summary>
    public string Type { get; init; } = "";
    public double? Min { get; init; }
    public double? Max { get; init; }
    public double? Step { get; init; }
    public IReadOnlyList<string> Options { get; init; } = [];
    public JsonElement? Default { get; init; }
    public string? Help { get; init; }
    public bool Required { get; init; } = true;

    public bool IsNumber => Type == SchemaInputTypes.Number;
    public bool IsSelect => Type == SchemaInputTypes.Select;

    public string DefaultText => SchemaValues.AsText(Default);
}

public sealed record SchemaOutputs
{
    /// <summary><c>classification</c> or <c>regression</c>.</summary>
    public string Type { get; init; } = "";
    public IReadOnlyList<string> Labels { get; init; } = [];
    public string? Unit { get; init; }
}

public sealed record SchemaPreset
{
    public string Name { get; init; } = "";
    public Dictionary<string, JsonElement> Values { get; init; } = [];
}

public static class SchemaInputTypes
{
    public const string Number = "number";
    public const string Select = "select";
    public static readonly IReadOnlySet<string> All = new HashSet<string> { Number, Select };
}

public static class SchemaOutputTypes
{
    public const string Classification = "classification";
    public const string Regression = "regression";
    public static readonly IReadOnlySet<string> All = new HashSet<string> { Classification, Regression };
}

public static class SchemaValues
{
    public static string AsText(JsonElement? value) => value is { } v
        ? v.ValueKind switch
        {
            JsonValueKind.Number => v.GetDouble().ToString("0.###############", CultureInfo.InvariantCulture),
            JsonValueKind.String => v.GetString() ?? "",
            _ => "",
        }
        : "";
}
