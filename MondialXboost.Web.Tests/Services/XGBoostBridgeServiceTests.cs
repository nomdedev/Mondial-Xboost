using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using MondialXboost.Web.Services;
using System.Net;
using System.Text.Json;

namespace MondialXboost.Web.Tests;

public class XGBoostBridgeServiceTests : TestFixtures
{
    [Fact]
    public async Task IsHealthyAsync_ReturnsTrue_WhenHealthEndpointResponds200()
    {
        var handler = new FakeHttpMessageHandler(new Dictionary<string, string>
        {
            { "http://127.0.0.1:8000/health", JsonSerializer.Serialize(new { status = "ok", model_loaded = true }) }
        });
        var client = new HttpClient(handler) { BaseAddress = new Uri("http://127.0.0.1:8000") };
        var service = new XGBoostBridgeService(client, BridgeOptions(), new NullLogger<XGBoostBridgeService>());

        var healthy = await service.IsHealthyAsync();

        Assert.True(healthy);
    }

    [Fact]
    public async Task IsHealthyAsync_ReturnsFalse_WhenHealthEndpointFails()
    {
        var handler = new FakeHttpMessageHandler(new Dictionary<string, string>());
        var client = new HttpClient(handler) { BaseAddress = new Uri("http://127.0.0.1:8000") };
        var service = new XGBoostBridgeService(client, BridgeOptions(), new NullLogger<XGBoostBridgeService>());

        var healthy = await service.IsHealthyAsync();

        Assert.False(healthy);
    }

    [Fact]
    public async Task PredictAsync_ReturnsCoherentPredictions()
    {
        var response = new PredictResponse
        {
            Predictions =
            [
                new XGBoostPrediction
                {
                    HomeTeam = "Argentina",
                    AwayTeam = "Brazil",
                    ProbHomeWin = 0.45,
                    ProbDraw = 0.25,
                    ProbAwayWin = 0.30,
                    ExpectedHomeGoals = 1.5,
                    ExpectedAwayGoals = 1.1,
                    TopPick = "Home"
                }
            ]
        };

        var handler = new FakeHttpMessageHandler(new Dictionary<string, string>
        {
            { "http://127.0.0.1:8000/predict", JsonSerializer.Serialize(response) }
        });
        var client = new HttpClient(handler) { BaseAddress = new Uri("http://127.0.0.1:8000") };
        var service = new XGBoostBridgeService(client, BridgeOptions(), new NullLogger<XGBoostBridgeService>());

        var fixtures = new List<XGBoostFixture>
        {
            new() { Date = "2026-06-15", HomeTeam = "Argentina", AwayTeam = "Brazil", Neutral = true }
        };

        var predictions = await service.PredictAsync(fixtures);

        var prediction = Assert.Single(predictions);
        Assert.Equal("Argentina", prediction.HomeTeam);
        Assert.Equal("Brazil", prediction.AwayTeam);
        Assert.True(Math.Abs(prediction.ProbHomeWin + prediction.ProbDraw + prediction.ProbAwayWin - 1.0) < 0.001);
        Assert.Equal("Home", prediction.TopPick);
    }

    private static IOptions<MondialXboostConfig> BridgeOptions() =>
        Options.Create(new MondialXboostConfig
        {
            XGBoostBridgeUrl = "http://127.0.0.1:8000"
        });
}
