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

public class ProbabilityTests : TestFixtures
{
    [Fact]
    public void OutcomeProbabilities_NormalizesAndUsesOutcomeLabels()
    {
        var p = new OutcomeProbabilities(2, 1, 1).Normalize();

        Assert.True(p.IsValid);
        Assert.Equal(0.5, p.HomeWin, 3);
        Assert.Equal("Home", p.TopPick);
    }

    [Fact]
    public void OutcomeFromExpectation_TreatsEqualMagnitudeGapsSymmetrically()
    {
        var strongerHome = ProbabilityHelper.OutcomeFromExpectation(.78, 400);
        var strongerAway = ProbabilityHelper.OutcomeFromExpectation(.22, -400);

        Assert.Equal(strongerHome.Draw, strongerAway.Draw, 6);
    }

    [Fact]
    public void PoissonScoreline_ProducesARealProbabilityGrid()
    {
        var dist = ProbabilityHelper.PoissonScoreline(2.2, .7);
        var sum = 0.0;
        for (var h = 0; h <= dist.MaxGoals; h++)
            for (var a = 0; a <= dist.MaxGoals; a++)
                sum += dist.Probability(h, a);

        Assert.Equal(1.0, sum, 6);
        Assert.True(dist.ToOutcome().HomeWin > dist.ToOutcome().AwayWin);
        Assert.NotEqual((0, 0), dist.MostLikelyScoreline());
    }

}
