using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Web.Controllers;

public class PredictController(
    IProjectCatalog catalog,
    ISchemaStore schemas,
    IModelRunner runner,
    ILogger<PredictController> logger) : Controller
{
    public const int MaxBodyBytes = 16 * 1024;

    /// <summary>
    /// Validates the JSON body against the project's schema and returns the model output.
    /// Input values are never logged; only the slug, outcome and latency are.
    /// </summary>
    [HttpPost("/predict/{slug}")]
    [RequestSizeLimit(MaxBodyBytes)]
    [EnableRateLimiting(RateLimitPolicies.Predict)]
    public IActionResult Predict(string slug, [FromBody] JsonElement body)
    {
        var project = catalog.Find(slug);
        if (project is null || project.Status != ProjectStatuses.Live || project.InferenceMode != ProjectInferenceModes.Onnx)
            return NotFound(new { error = "No live model for this project." });

        var schema = schemas.Get(project.Slug);
        if (schema is null)
            return NotFound(new { error = "No live model for this project." });

        var validation = InputValidator.Validate(schema, body);
        if (!validation.IsValid)
        {
            logger.LogInformation("Predict {Slug}: {Count} validation error(s)", project.Slug, validation.Errors.Count);
            return BadRequest(new { errors = validation.Errors });
        }

        var timer = Stopwatch.StartNew();
        PredictionResult result;
        try
        {
            result = runner.Predict(project.Slug, schema, validation.Values);
        }
        catch (ModelUnavailableException)
        {
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new { error = "The model is unavailable right now. Please try again later." });
        }
        logger.LogInformation("Predict {Slug}: ok in {ElapsedMs} ms", project.Slug, timer.ElapsedMilliseconds);

        return Ok(ToResponse(schema, result));
    }

    private static PredictResponse ToResponse(FormSchema schema, PredictionResult r)
    {
        if (schema.Outputs.Type == SchemaOutputTypes.Regression)
        {
            var value = Math.Round(r.Prediction, 4);
            var unit = schema.Outputs.Unit ?? "";
            return new PredictResponse(
                value,
                Label: value.ToString("+0.00;-0.00;0.00", CultureInfo.InvariantCulture) + unit,
                Probabilities: null,
                Explanation: $"For these inputs the model predicts {value.ToString("0.00", CultureInfo.InvariantCulture)}{unit}.",
                schema.ModelVersion);
        }

        var probabilities = r.Probabilities!.ToDictionary(p => p.Key, p => Math.Round(p.Value, 4));
        var top = probabilities[r.Label!];
        return new PredictResponse(
            r.Prediction,
            r.Label,
            probabilities,
            Explanation: $"The model's most likely outcome is {r.Label} ({(top * 100).ToString("0.0", CultureInfo.InvariantCulture)}% probability).",
            schema.ModelVersion);
    }
}

/// <summary>Response body of <c>POST /predict/{slug}</c> (SPEC §4.3).</summary>
public sealed record PredictResponse(
    double Prediction,
    string? Label,
    IReadOnlyDictionary<string, double>? Probabilities,
    string? Explanation,
    string ModelVersion);

public static class RateLimitPolicies
{
    public const string Predict = "predict";
}
