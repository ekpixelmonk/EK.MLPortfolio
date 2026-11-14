namespace MLPortfolio.Web.Models;

/// <summary>
/// The model's reading of one day's indicators, shown on a project page with <c>inferenceMode: "snapshot"</c>
/// instead of a form. The inputs come from <c>snapshot.json</c> (written by the project's <c>train.py</c>); the
/// label and probabilities are produced here by the exported ONNX model.
/// </summary>
/// <param name="AsOf">The market close the indicators describe.</param>
/// <param name="Label">The most likely class.</param>
/// <param name="Probabilities">Every class with its probability, most likely first.</param>
/// <param name="Inputs">The indicator values behind the reading, in schema order, with their labels.</param>
public sealed record RegimeReading(
    DateOnly AsOf,
    string Label,
    IReadOnlyList<KeyValuePair<string, double>> Probabilities,
    IReadOnlyList<ReadingInput> Inputs)
{
    public double TopProbability => Probabilities[0].Value;
}

public sealed record ReadingInput(string Label, double Value);
