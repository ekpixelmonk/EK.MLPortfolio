// Client-side domain/tag filter + text search for the homepage project cards.
(function () {
  var cards = Array.prototype.slice.call(document.querySelectorAll(".project-card"));
  var sections = Array.prototype.slice.call(document.querySelectorAll(".project-section"));
  var chips = Array.prototype.slice.call(document.querySelectorAll(".chip"));
  var search = document.getElementById("project-search");
  var status = document.getElementById("filter-status");
  var empty = document.getElementById("no-results");
  var active = { kind: "all", value: "" };

  function matchesChip(card) {
    if (active.kind === "domain") return card.dataset.domain === active.value;
    if (active.kind === "tag") {
      var tags = card.dataset.tags ? card.dataset.tags.split("|") : [];
      return tags.indexOf(active.value) !== -1;
    }
    return true;
  }

  function apply() {
    var query = search.value.trim().toLowerCase();
    var visible = 0;

    cards.forEach(function (card) {
      var matchesText = !query || card.dataset.search.indexOf(query) !== -1;
      var show = matchesChip(card) && matchesText;
      card.hidden = !show;
      if (show) visible++;
    });

    sections.forEach(function (section) {
      section.hidden = !section.querySelector(".project-card:not([hidden])");
    });

    empty.hidden = visible !== 0;
    status.textContent = visible + (visible === 1 ? " project shown" : " projects shown");
  }

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      active = { kind: chip.dataset.kind, value: chip.dataset.value };
      chips.forEach(function (c) {
        var on = c === chip;
        c.classList.toggle("is-active", on);
        c.setAttribute("aria-pressed", on ? "true" : "false");
      });
      apply();
    });
  });

  search.addEventListener("input", apply);
})();
