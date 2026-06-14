using Microsoft.AspNetCore.Hosting;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Options;
using MondialXboost.Web;
using MondialXboost.Web.DAL;
using MondialXboost.Web.Helpers;
using MondialXboost.Web.Models;
using MondialXboost.Web.Models.ApiFootballModels;
using MondialXboost.Web.Models.CsvModels;
using MondialXboost.Web.Predictors;
using MondialXboost.Web.Probability;
using MondialXboost.Web.Services;
using MondialXboost.Web.Services.Simulation;
using System.Globalization;
using System.Net;
using System.Text.Json;

namespace MondialXboost.Web.Tests;

public class TeamNameAndFlagTests : TestFixtures
{
    [Theory]
    [InlineData("Korea Republic", "south-korea")]
    [InlineData("Türkiye", "turkey")]
    [InlineData("USA", "united-states")]
    public void TeamNameNormalizer_HandlesAliases(string input, string expected)
    {
        Assert.Equal(expected, TeamNameNormalizer.ToId(input));
    }

    [Theory]
    [InlineData("argentina", "Argentina", "ar")]
    [InlineData("brazil", "Brazil", "br")]
    [InlineData("france", "France", "fr")]
    [InlineData("japan", "Japan", "jp")]
    [InlineData("united-states", "United States", "us")]
    [InlineData("south-korea", "South Korea", "kr")]
    [InlineData("turkey", "Turkey", "tr")]
    [InlineData("ivory-coast", "Ivory Coast", "ci")]
    [InlineData("congo-dr", "Congo DR", "cd")]
    [InlineData("curacao", "Curacao", "cw")]
    [InlineData("cape-verde", "Cape Verde", "cv")]
    [InlineData("england", "England", "gb-eng")]
    [InlineData("scotland", "Scotland", "gb-sct")]
    [InlineData("wales", "Wales", "gb-wls")]
    [InlineData("northern-ireland", "Northern Ireland", "gb-nir")]
    [InlineData("china-pr", "China PR", "cn")]
    [InlineData("chinese-taipei", "Chinese Taipei", "tw")]
    [InlineData("korea-dpr", "Korea DPR", "kp")]
    [InlineData("republic-of-ireland", "Republic of Ireland", "ie")]
    [InlineData("palestine", "Palestine", "ps")]
    [InlineData("faroe-islands", "Faroe Islands", "fo")]
    public void TeamFlagCatalog_ResolvesStandardAndFootballTeamNames(string id, string name, string expected)
    {
        Assert.Equal(expected, TeamFlagCatalog.CodeFor(id, name));
    }

    [Fact]
    public void TeamFlagCatalog_ReturnsNoFlagForUnknownTeams()
    {
        Assert.Null(TeamFlagCatalog.CodeFor("made-up-xi", "Made Up XI"));
    }

}
