using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

namespace MLPortfolio.Web.Controllers;

public class HomeController(IProjectCatalog catalog, IOptions<SiteOptions> site) : Controller
{
    public IActionResult Index()
    {
        var model = new HomeViewModel(
            site.Value,
            catalog.InCategory(ProjectCategories.Flagship),
            catalog.InCategory(ProjectCategories.Zoo),
            catalog.Domains,
            catalog.SharedTags);
        return View(model);
    }

    /// <summary>Styled page for bare status codes (e.g. unknown URL or unknown project slug). The original status code is preserved.</summary>
    [Route("/error/{code:int}")]
    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult StatusCodePage(int code) =>
        code == StatusCodes.Status404NotFound
            ? View("NotFound")
            : View("Error", new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
