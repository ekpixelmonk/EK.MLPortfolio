using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;
using MLPortfolio.Web.Models;

namespace MLPortfolio.Web.Services;

public interface IRegimeReadingService
{
    /// <summary>
    /// The model's reading of the project's snapshot, or null when the project has no valid snapshot or schema
    /// or the model cannot run (the page then renders without the reading).
    /// </summary>
    RegimeReading? Get(string slug);
}

/// <summary>Reads <c>App_Data/models/&lt;slug&gt;/snapshot.json</c> and runs the project's ONNX model on it, once, then caches the result.</summary>
public sealed class RegimeReadingService(
    IWebHostEnvironment env,
    ISchemaStore schemas,
    IModelRunner runner,
    ILogger<RegimeReadingService> logger) : IRegimeReadingService
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly ConcurrentDictionary<string, RegimeReading> _cache = new(StringComparer.OrdinalIgnoreCase);

    /// <remarks>The slug must already have been validated against the catalog; it is never taken from user input directly.</remarks>
    public RegimeReading? Get(string slug)
    {
        if (_cache.TryGetValue(slug, out var cached)) return cached;
        var reading = Load(slug);
        if (reading is not null) _cache[slug] = reading;   // failures are not cached, so a fixed file is picked up
        return reading;
    }

    private RegimeReading? Load(string slug)
    {
        var path = Path.Combine(env.ContentRootPath, "App_Data", "models", slug, "snapshot.json");
        var schema = schemas.Get(slug);
        if (!File.Exists(path) || schema is null || schema.Outputs.Type != SchemaOutputTypes.Classification) return null;

        try
        {
            var snapshot = JsonSerializer.Deserialize<SnapshotFile>(File.ReadAllText(path), JsonOptions);
            if (snapshot is null || !DateOnly.TryParseExact(snapshot.AsOf, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var asOf))
                throw new InvalidDataException("asOf must be a date (yyyy-MM-dd).");

            var values = new Dictionary<string, object>();
            var inputs = new List<ReadingInput>();
            foreach (var input in schema.Inputs)
            {
                if (!snapshot.Features.TryGetValue(input.Name, out var value))
                    throw new InvalidDataException($"feature '{input.Name}' is missing.");
                if (!double.IsFinite(value) || (input.Min is { } min && value < min) || (input.Max is { } max && value > max))
                    throw new InvalidDataException($"feature '{input.Name}' is outside the model's range.");
                values[input.Name] = value;
                inputs.Add(new ReadingInput(input.Label, value));
            }

            var result = runner.Predict(slug, schema, values);
            var ordered = result.Probabilities!.OrderByDescending(p => p.Value).ToList();
            return new RegimeReading(asOf, result.Label!, ordered, inputs);
        }
        catch (Exception ex) when (ex is JsonException or InvalidDataException or ModelUnavailableException)
        {
            logger.LogWarning("No reading for {Slug}: {Message}", slug, ex.Message);
            return null;
        }
    }

    private sealed class SnapshotFile
    {
        public string AsOf { get; set; } = "";
        public Dictionary<string, double> Features { get; set; } = [];
    }
}
