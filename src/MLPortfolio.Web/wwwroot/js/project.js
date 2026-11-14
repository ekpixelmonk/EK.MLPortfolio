// "Try the model" form: example presets, validation and the prediction call.
// Expected response from POST /predict/{slug}:
//   { prediction, label?, probabilities?: { "<label>": 0..1 }, explanation?, modelVersion }
// Expected 400 body: { errors: { "<field>": "message" | ["message", ...] } }
(function () {
  var form = document.getElementById("try-form");
  if (!form) return;

  var slug = form.dataset.slug;
  var result = document.getElementById("try-result");
  var labelEl = document.getElementById("result-label");
  var probsEl = document.getElementById("result-probs");
  var explanationEl = document.getElementById("result-explanation");
  var metaEl = document.getElementById("result-meta");
  var submit = form.querySelector("button[type=submit]");

  function field(name) { return form.elements[name]; }

  function clearErrors() {
    form.querySelectorAll("[data-error-for]").forEach(function (el) { el.textContent = ""; });
    Array.prototype.forEach.call(form.elements, function (el) { el.removeAttribute("aria-invalid"); });
  }

  function showError(name, message) {
    var el = form.querySelector('[data-error-for="' + name + '"]');
    if (el) el.textContent = message;
    var input = field(name);
    if (input) input.setAttribute("aria-invalid", "true");
  }

  function validateLocally() {
    var ok = true;
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || el.type === "submit" || el.type === "button" || el.type === "fieldset") return;
      if (!el.checkValidity()) { showError(el.name, el.validationMessage); ok = false; }
    });
    return ok;
  }

  function payload() {
    var data = {};
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!el.name || el.type === "submit" || el.type === "button" || el.type === "fieldset") return;
      data[el.name] = el.dataset.type === "number" ? Number(el.value) : el.value;
    });
    return data;
  }

  function showResult(data) {
    labelEl.textContent = data.label != null ? String(data.label) : String(data.prediction);
    probsEl.replaceChildren();
    if (data.probabilities && typeof data.probabilities === "object") {
      Object.keys(data.probabilities).forEach(function (key) {
        var pct = Math.round(Number(data.probabilities[key]) * 1000) / 10;
        var li = document.createElement("li");
        var name = document.createElement("span");
        name.className = "prob-name";
        name.textContent = key;
        var bar = document.createElement("span");
        bar.className = "prob-bar";
        var fill = document.createElement("span");
        fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
        bar.appendChild(fill);
        var value = document.createElement("span");
        value.className = "prob-value";
        value.textContent = pct + "%";
        li.append(name, bar, value);
        probsEl.appendChild(li);
      });
    }
    explanationEl.textContent = data.explanation || "";
    metaEl.textContent = data.modelVersion ? "Model version " + data.modelVersion : "";
    result.hidden = false;
  }

  function showFailure(message) {
    labelEl.textContent = message;
    probsEl.replaceChildren();
    explanationEl.textContent = "";
    metaEl.textContent = "";
    result.hidden = false;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    clearErrors();
    if (!validateLocally()) return;

    submit.disabled = true;
    fetch("/predict/" + encodeURIComponent(slug), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload())
    })
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (body) {
          return { status: response.status, ok: response.ok, body: body };
        });
      })
      .then(function (r) {
        if (r.ok && r.body) { showResult(r.body); return; }
        if (r.status === 400 && r.body && r.body.errors) {
          result.hidden = true;
          Object.keys(r.body.errors).forEach(function (name) {
            var msg = r.body.errors[name];
            showError(name, Array.isArray(msg) ? msg.join(" ") : String(msg));
          });
          return;
        }
        showFailure("A prediction isn't available right now. Please try again later.");
      })
      .catch(function () { showFailure("Couldn't reach the server. Please try again."); })
      .then(function () { submit.disabled = false; });
  });

  document.querySelectorAll("[data-preset]").forEach(function (button) {
    button.addEventListener("click", function () {
      var values;
      try { values = JSON.parse(button.dataset.preset); } catch (e) { return; }
      clearErrors();
      Object.keys(values).forEach(function (name) {
        var input = field(name);
        if (input) input.value = values[name];
      });
      result.hidden = true;
    });
  });
})();
