using System.Text.Json;
using MLPortfolio.Web.Models;

namespace MLPortfolio.Web.Services;

/// <summary>Validated model inputs, keyed by schema input name. Numbers are <see cref="double"/>, selects are <see cref="string"/>.</summary>
public sealed record ValidationResult(IReadOnlyDictionary<string, object> Values, IReadOnlyDictionary<string, string> Errors)
{
    public bool IsValid => Errors.Count == 0;
}

/// <summary>Server-side validation of a prediction request against the project's <see cref="FormSchema"/>.</summary>
public static class InputValidator
{
    /// <summary>Error key used for problems that are not tied to a single field.</summary>
    public const string RequestKey = "_request";

    public static ValidationResult Validate(FormSchema schema, JsonElement body)
    {
        var values = new Dictionary<string, object>();
        var errors = new Dictionary<string, string>();

        if (body.ValueKind != JsonValueKind.Object)
        {
            errors[RequestKey] = "The request body must be a JSON object of field names to values.";
            return new ValidationResult(values, errors);
        }

        var known = schema.Inputs.Select(i => i.Name).ToHashSet();
        foreach (var property in body.EnumerateObject())
        {
            if (!known.Contains(property.Name)) errors[property.Name] = "Unknown field.";
        }

        foreach (var input in schema.Inputs)
        {
            if (errors.ContainsKey(input.Name)) continue;

            if (!body.TryGetProperty(input.Name, out var raw) || raw.ValueKind == JsonValueKind.Null)
            {
                if (input.Required) errors[input.Name] = "This field is required.";
                else if (input.Default is not null) AddDefault(input, values);
                continue;
            }

            if (input.IsNumber) ValidateNumber(input, raw, values, errors);
            else ValidateSelect(input, raw, values, errors);
        }

        return new ValidationResult(values, errors);
    }

    private static void ValidateNumber(SchemaInput input, JsonElement raw, Dictionary<string, object> values, Dictionary<string, string> errors)
    {
        if (raw.ValueKind != JsonValueKind.Number || !raw.TryGetDouble(out var number) || !double.IsFinite(number))
        {
            errors[input.Name] = "Must be a number.";
            return;
        }
        if (input.Min is { } min && number < min) { errors[input.Name] = $"Must be at least {min:0.###}."; return; }
        if (input.Max is { } max && number > max) { errors[input.Name] = $"Must be at most {max:0.###}."; return; }
        values[input.Name] = number;
    }

    private static void ValidateSelect(SchemaInput input, JsonElement raw, Dictionary<string, object> values, Dictionary<string, string> errors)
    {
        var text = raw.ValueKind == JsonValueKind.String ? raw.GetString() : null;
        if (text is null || !input.Options.Contains(text))
        {
            errors[input.Name] = $"Must be one of: {string.Join(", ", input.Options)}.";
            return;
        }
        values[input.Name] = text;
    }

    private static void AddDefault(SchemaInput input, Dictionary<string, object> values)
    {
        var d = input.Default!.Value;
        values[input.Name] = input.IsNumber ? d.GetDouble() : d.GetString()!;
    }
}
