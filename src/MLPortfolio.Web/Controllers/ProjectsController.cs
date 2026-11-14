using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Web.Controllers;

public class ProjectsController(
    IProjectCatalog catalog,
    ISchemaStore schemas,
    IProjectContentStore content,
    IOptions<SiteOptions> site) : Controller
{
    [HttpGet("/projects/{slug}")]
    public IActionResult Details(string slug)
    {
        // Only slugs that exist in the catalog reach the file system.
        var project = catalog.Find(slug);
        if (project is null) return NotFound();

        var schema = project.InferenceMode == ProjectInferenceModes.None ? null : schemas.Get(project.Slug);
        var model = new ProjectPageViewModel(
            project,
            content.GetHtml(project.Slug),
            schema,
            FormEnabled: project.Status == ProjectStatuses.Live,
            catalog.Related(project),
            site.Value.CodeUrlFor(project),
            site.Value.NotebookUrlFor(project));
        return View(model);
    }
}
