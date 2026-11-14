// Theme toggle. The initial theme is applied by an inline script in _Layout (before first paint).
(function () {
  var root = document.documentElement;
  var button = document.getElementById("theme-toggle");
  if (!button) return;

  function sync() {
    var dark = root.getAttribute("data-bs-theme") === "dark";
    button.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }

  button.addEventListener("click", function () {
    var next = root.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-bs-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) { }
    sync();
  });

  sync();
})();
