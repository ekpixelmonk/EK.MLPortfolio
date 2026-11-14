using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLPortfolio.Web.Models;

namespace MLPortfolio.Web.Services;

/// <summary>Raw model output; formatting for the client happens in the controller.</summary>
/// <param name="Prediction">Class index for classification, the predicted value for regression.</param>
/// <param name="Label">Class label for classification; null for regression.</param>
/// <param name="Probabilities">Probability per label for classification; null for regression.</param>
public sealed record PredictionResult(double Prediction, string? Label, IReadOnlyDictionary<string, double>? Probabilities);

public interface IModelRunner
{
    /// <summary>Runs the project's model on already-validated inputs (see <see cref="InputValidator"/>).</summary>
    /// <exception cref="ModelUnavailableException">The model file is missing or cannot be run.</exception>
    PredictionResult Predict(string slug, FormSchema schema, IReadOnlyDictionary<string, object> values);
}

public sealed class ModelUnavailableException(string message, Exception? inner = null) : Exception(message, inner);

/// <summary>
/// Runs ONNX models in-process. One <see cref="InferenceSession"/> per project is created lazily and reused.
/// Convention shared with the Python training code: one ONNX input per schema input, named after it,
/// shape [1,1] (float for numbers, string for selects).
/// </summary>
public sealed class OnnxModelRunner(IWebHostEnvironment env, ILogger<OnnxModelRunner> logger) : IModelRunner, IDisposable
{
    private readonly ConcurrentDictionary<string, Lazy<InferenceSession>> _sessions = new(StringComparer.OrdinalIgnoreCase);

    public PredictionResult Predict(string slug, FormSchema schema, IReadOnlyDictionary<string, object> values)
    {
        InferenceSession session;
        try
        {
            session = _sessions.GetOrAdd(slug, s => new Lazy<InferenceSession>(() => CreateSession(s, schema))).Value;
        }
        catch (Exception ex)
        {
            // Drop the failed lazy so a later request can retry after the model is fixed.
            _sessions.TryRemove(slug, out _);
            logger.LogError(ex, "Could not load model for {Slug}", slug);
            throw new ModelUnavailableException($"Model for '{slug}' could not be loaded.", ex);
        }

        try
        {
            using var results = session.Run(BuildInputs(schema, values));
            return ReadOutputs(schema, results);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or KeyNotFoundException)
        {
            logger.LogError(ex, "Inference failed for {Slug}", slug);
            throw new ModelUnavailableException($"Inference failed for '{slug}'.", ex);
        }
    }

    private InferenceSession CreateSession(string slug, FormSchema schema)
    {
        var path = Path.Combine(env.ContentRootPath, "App_Data", "models", slug, schema.ModelFile);
        if (!File.Exists(path)) throw new FileNotFoundException($"Model file not found for '{slug}'.", path);
        return new InferenceSession(path);
    }

    private static List<NamedOnnxValue> BuildInputs(FormSchema schema, IReadOnlyDictionary<string, object> values)
    {
        var inputs = new List<NamedOnnxValue>(schema.Inputs.Count);
        foreach (var input in schema.Inputs)
        {
            var value = values[input.Name];
            inputs.Add(input.IsNumber
                ? NamedOnnxValue.CreateFromTensor(input.Name, new DenseTensor<float>(new[] { (float)(double)value }, [1, 1]))
                : NamedOnnxValue.CreateFromTensor(input.Name, new DenseTensor<string>(new[] { (string)value }, [1, 1])));
        }
        return inputs;
    }

    private static PredictionResult ReadOutputs(FormSchema schema, IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        if (schema.Outputs.Type == SchemaOutputTypes.Regression)
        {
            var value = results[0].AsTensor<float>().GetValue(0);
            return new PredictionResult(value, null, null);
        }

        // skl2onnx classifiers emit (label, probabilities). Find the probability tensor by name, else take the 2nd output.
        var probOutput = results.FirstOrDefault(r => r.Name.Contains("prob", StringComparison.OrdinalIgnoreCase)) ?? results[1];
        var tensor = probOutput.AsTensor<float>();
        var labels = schema.Outputs.Labels;
        if (tensor.Length != labels.Count)
            throw new InvalidOperationException($"Model returned {tensor.Length} probabilities but the schema lists {labels.Count} labels.");

        var probabilities = new Dictionary<string, double>(labels.Count);
        var best = 0;
        for (var i = 0; i < labels.Count; i++)
        {
            var p = tensor.GetValue(i);
            probabilities[labels[i]] = p;
            if (p > tensor.GetValue(best)) best = i;
        }
        return new PredictionResult(best, labels[best], probabilities);
    }

    public void Dispose()
    {
        foreach (var lazy in _sessions.Values)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
    }
}
