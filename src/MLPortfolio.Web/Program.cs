using MLPortfolio.Web.Models;
using MLPortfolio.Web.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();
builder.Services.Configure<SiteOptions>(builder.Configuration.GetSection(SiteOptions.SectionName));

// Loaded once at startup; an invalid projects.json fails fast with a readable message.
builder.Services.AddSingleton<IProjectCatalog>(_ =>
    ProjectCatalog.LoadFile(Path.Combine(builder.Environment.ContentRootPath, "App_Data", "projects.json")));

builder.Services.AddSingleton<ISchemaStore, SchemaStore>();
builder.Services.AddSingleton<IProjectContentStore, ProjectContentStore>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

// Styled 404 (and other bare status codes) while keeping the real status code.
app.UseStatusCodePagesWithReExecute("/error/{0}");

app.UseHttpsRedirection();
app.UseRouting();

app.MapStaticAssets();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}")
    .WithStaticAssets();

app.Run();

// Exposed so WebApplicationFactory<Program> can host the app in integration tests.
public partial class Program;
