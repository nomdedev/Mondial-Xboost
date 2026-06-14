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

public class CsvImportServiceTests : TestFixtures
{
    [Fact]
    public async Task CsvImport_CreatesTeamsGroupsFixturesRatingsAndResults()
    {
        await using var db = await NewDb();
        var importer = new CsvImportService(db, new TestEnvironment(WebProjectRoot()));

        var report = await importer.ImportAllAsync();

        Assert.True(report.Teams >= 48);
        Assert.Equal(12, report.Groups);
        Assert.Equal(72, report.Fixtures);
        Assert.True(report.Ratings > 0);
        Assert.True(report.Results > 0);
        Assert.Equal(ExpectedUniqueHistoricalResultIds(), report.Results);
        Assert.DoesNotContain(await db.Fixtures.ToListAsync(), f => string.IsNullOrWhiteSpace(f.Group));
    }

    private static int ExpectedUniqueHistoricalResultIds()
    {
        var rows = CsvParsingHelper.ReadCsv<HistoricalResultCsvRow>(Path.Combine(WebProjectRoot(), "Data", MondialXboostDataFiles.HistoricalResultsCsv));
        var ids = new HashSet<string>(StringComparer.Ordinal);

        foreach (var row in rows)
        {
            if (!DateTimeOffset.TryParse(row.Date, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var date) ||
                !double.TryParse(row.HomeScore, NumberStyles.Any, CultureInfo.InvariantCulture, out var homeScoreD) ||
                !double.TryParse(row.AwayScore, NumberStyles.Any, CultureInfo.InvariantCulture, out var awayScoreD))
            {
                continue;
            }

            var homeScore = (int)homeScoreD;
            var awayScore = (int)awayScoreD;
            var homeId = TeamNameNormalizer.ToId(row.HomeTeam);
            var awayId = TeamNameNormalizer.ToId(row.AwayTeam);
            ids.Add(CryptoUtil.GetSha256($"{homeId}-{awayId}-{date:O}-{row.Tournament}-{homeScore}-{awayScore}"));
        }

        return ids.Count;
    }

}
