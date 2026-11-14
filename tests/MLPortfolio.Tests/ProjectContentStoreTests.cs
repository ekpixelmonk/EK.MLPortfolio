using MLPortfolio.Web.Services;

namespace MLPortfolio.Tests;

public class ProjectContentStoreTests
{
    [Fact]
    public void Render_ConvertsMarkdownToHtml()
    {
        var html = ProjectContentStore.Render("## Problem\n\nSome **bold** text.");

        Assert.Contains("<h2", html);
        Assert.Contains("Problem</h2>", html);
        Assert.Contains("<strong>bold</strong>", html);
    }

    [Fact]
    public void Render_SupportsTables()
    {
        var html = ProjectContentStore.Render("| a | b |\n|---|---|\n| 1 | 2 |");

        Assert.Contains("<table>", html);
    }

    [Fact]
    public void Render_DoesNotPassRawHtmlThrough()
    {
        var html = ProjectContentStore.Render("Hello <script>alert(1)</script> <img src=x onerror=alert(1)>");

        Assert.DoesNotContain("<script>", html);
        Assert.DoesNotContain("<img", html);
    }
}
