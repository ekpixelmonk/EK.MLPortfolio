using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Web.Controllers;

public class ProjectsController(
    IProjectCatalog catalog,
    ISchemaStore schemas,
    IProjectContentStore content,
    IRegimeReadingService readings,
    IOptions<SiteOptions> site) : Controller
{
    [HttpGet("/projects/{slug}")]
    public IActionResult Details(string slug)
    {
        // Only slugs that exist in the catalog reach the file system.
        var project = catalog.Find(slug);
        if (project is null) return NotFound();

        // A snapshot project has no form: its schema is only used to run the model on the stored snapshot.
        var hasForm = project.InferenceMode is not (ProjectInferenceModes.None or ProjectInferenceModes.Snapshot);
        var schema = hasForm ? schemas.Get(project.Slug) : null;
        var reading = project.InferenceMode == ProjectInferenceModes.Snapshot && project.Status == ProjectStatuses.Live
            ? readings.Get(project.Slug)
            : null;
        var model = new ProjectPageViewModel(
            project,
            content.GetHtml(project.Slug),
            schema,
            FormEnabled: project.Status == ProjectStatuses.Live,
            catalog.Related(project),
            site.Value.CodeUrlFor(project),
            site.Value.NotebookUrlFor(project),
            reading);
        return View(model);
    }
}
