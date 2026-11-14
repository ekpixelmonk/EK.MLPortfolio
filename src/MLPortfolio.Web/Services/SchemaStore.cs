using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.RegularExpressions;
using MLPortfolio.Web.Models;

namespace MLPortfolio.Web.Services;

public interface ISchemaStore
{
    /// <summary>The project's form schema, or null when it has none or it is invalid (the page then renders without a form).</summary>
    FormSchema? Get(string slug);
}

public sealed class SchemaException(string message) : Exception(message);

public sealed partial class SchemaStore(IWebHostEnvironment env, ILogger<SchemaStore> logger) : ISchemaStore
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    private readonly ConcurrentDictionary<string, FormSchema?> _cache = new(StringComparer.OrdinalIgnoreCase);

    /// <remarks>The slug must already have been validated against the catalog; it is never taken from user input directly.</remarks>
    public FormSchema? Get(string slug) => _cache.GetOrAdd(slug, Load);

    private FormSchema? Load(string slug)
    {
        var path = Path.Combine(env.ContentRootPath, "App_Data", "models", slug, "schema.json");
        if (!File.Exists(path)) return null;

        try
        {
            return Parse(File.ReadAllText(path), path);
        }
        catch (SchemaException ex)
        {
            logger.LogWarning("Ignoring invalid schema for {Slug}: {Message}", slug, ex.Message);
            return null;
        }
    }

    public static FormSchema Parse(string json, string source = "schema.json")
    {
        FormSchema? schema;
        try
        {
            schema = JsonSerializer.Deserialize<FormSchema>(json, JsonOptions);
        }
        catch (JsonException ex)
        {
            throw new SchemaException($"{source}: invalid JSON. {ex.Message}");
        }

        if (schema is null) throw new SchemaException($"{source}: expected a JSON object.");

        var errors = Validate(schema);
        if (errors.Count > 0)
            throw new SchemaException($"{source} is invalid:{Environment.NewLine}- " +
                                      string.Join(Environment.NewLine + "- ", errors));
        return schema;
    }

    private static List<string> Validate(FormSchema s)
    {
        var errors = new List<string>();

        if (s.Inputs.Count == 0) errors.Add("at least one input is required.");
        if (!SchemaOutputTypes.All.Contains(s.Outputs.Type))
            errors.Add($"outputs.type '{s.Outputs.Type}' must be one of {string.Join(", ", SchemaOutputTypes.All)}.");
        if (s.Outputs.Type == SchemaOutputTypes.Classification && s.Outputs.Labels.Count < 2)
            errors.Add("classification outputs need at least two labels.");

        var byName = new Dictionary<string, SchemaInput>();
        foreach (var input in s.Inputs)
        {
            var who = $"input '{input.Name}'";
            if (!NamePattern().IsMatch(input.Name)) { errors.Add($"{who}: name must be an identifier (letters, digits, underscore)."); continue; }
            if (!byName.TryAdd(input.Name, input)) { errors.Add($"{who}: duplicate name."); continue; }
            if (string.IsNullOrWhiteSpace(input.Label)) errors.Add($"{who}: label is required.");

            if (!SchemaInputTypes.All.Contains(input.Type))
            {
                errors.Add($"{who}: type '{input.Type}' must be one of {string.Join(", ", SchemaInputTypes.All)}.");
                continue;
            }

            if (input.IsNumber)
            {
                if (input.Min is { } min && input.Max is { } max && min > max) errors.Add($"{who}: min is greater than max.");
                if (input.Step is <= 0) errors.Add($"{who}: step must be positive.");
            }
            else if (input.Options.Count == 0)
            {
                errors.Add($"{who}: select inputs need options.");
            }

            if (input.Default is not null && !IsValidValue(input, input.Default.Value))
                errors.Add($"{who}: default is not valid for this input.");
        }

        foreach (var preset in s.Presets)
        {
            if (string.IsNullOrWhiteSpace(preset.Name)) errors.Add("a preset is missing its name.");
            foreach (var (key, value) in preset.Values)
            {
                if (!byName.TryGetValue(key, out var input)) errors.Add($"preset '{preset.Name}': unknown input '{key}'.");
                else if (!IsValidValue(input, value)) errors.Add($"preset '{preset.Name}': invalid value for '{key}'.");
            }
        }

        return errors;
    }

    private static bool IsValidValue(SchemaInput input, JsonElement value) => input.Type switch
    {
        SchemaInputTypes.Number => value.ValueKind == JsonValueKind.Number
            && (input.Min is not { } min || value.GetDouble() >= min)
            && (input.Max is not { } max || value.GetDouble() <= max),
        SchemaInputTypes.Select => value.ValueKind == JsonValueKind.String && input.Options.Contains(value.GetString()!),
        _ => false,
    };

    [GeneratedRegex("^[A-Za-z_][A-Za-z0-9_]*$")]
    private static partial Regex NamePattern();
}
