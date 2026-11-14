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
            catalog.Tags);
        return View(model);
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
