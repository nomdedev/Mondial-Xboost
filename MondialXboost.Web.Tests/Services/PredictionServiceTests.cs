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

public class PredictionServiceTests : TestFixtures
{
    [Fact]
    public async Task PredictionService_BulkPredictsImportedGroupFixtures()
    {
        await using var db = await ImportedDb();
        var fixtures = await db.Fixtures.AsNoTracking().ToListAsync();
        var service = new PredictionService(db, SimulationOptions(1, 1));

        var results = await service.PredictFixturesAsync(fixtures);

        Assert.Equal(72, results.Count);
        Assert.All(results, result =>
        {
            Assert.False(string.IsNullOrWhiteSpace(result.Fixture.Id));
            Assert.True(result.BestPrediction.Outcome.IsValid);
        });
    }

    [Fact]
    public async Task PredictionService_BulkPredictionsMatchSingleFixturePredictions()
    {
        await using var db = await ImportedDb();
        var fixtures = await db.Fixtures
            .AsNoTracking()
            .OrderBy(f => f.Group)
            .ThenBy(f => f.HomeTeamId)
            .ThenBy(f => f.AwayTeamId)
            .Take(3)
            .ToListAsync();
        var service = new PredictionService(db, SimulationOptions(1, 1));

        var bulk = await service.PredictFixturesAsync(fixtures);

        foreach (var fixture in fixtures)
        {
            var expected = await service.PredictFixtureAsync(fixture.Id);
            var actual = bulk.Single(result => result.Fixture.Id == fixture.Id);

            Assert.NotNull(expected);
            AssertPredictionResultEqual(expected, actual);
        }
    }

    [Fact]
    public async Task PredictionService_BulkPredictionUsesFixtureIdsWhenTeamsAreMissing()
    {
        await using var db = await NewDb();
        var fixture = new Fixture { Id = "f1", Group = "A", HomeTeamId = "ghost-home", AwayTeamId = "ghost-away" };
        db.Fixtures.Add(fixture);
        await db.SaveChangesAsync();
        var service = new PredictionService(db, SimulationOptions(1, 1));

        var result = Assert.Single(await service.PredictFixturesAsync([fixture]));

        Assert.Equal("ghost-home", result.HomeTeamName);
        Assert.Equal("ghost-away", result.AwayTeamName);
        Assert.True(result.BestPrediction.Outcome.IsValid);
    }

}
