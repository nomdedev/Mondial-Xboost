using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Options;

namespace Oloraculo.Web.Services
{
    public class XGBoostBridgeService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<XGBoostBridgeService> _logger;

        public XGBoostBridgeService(HttpClient httpClient, IOptions<OloraculoConfig> config, ILogger<XGBoostBridgeService> logger)
        {
            _httpClient = httpClient;
            _httpClient.BaseAddress = new Uri(config.Value.XGBoostBridgeUrl);
            _logger = logger;
        }

        public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
        {
            try
            {
                var response = await _httpClient.GetAsync("/health", ct);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "XGBoost bridge health check failed");
                return false;
            }
        }

        public async Task<IReadOnlyList<XGBoostPrediction>> PredictAsync(IEnumerable<XGBoostFixture> fixtures, CancellationToken ct = default)
        {
            var payload = new PredictRequest { Fixtures = fixtures.ToList() };
            var response = await _httpClient.PostAsJsonAsync("/predict", payload, ct);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<PredictResponse>(ct);
            return result?.Predictions ?? [];
        }

        public async Task<TrainResponse> TrainAsync(string minDate = "2010-01-01", CancellationToken ct = default)
        {
            var response = await _httpClient.PostAsync($"/train?min_date={minDate}", null, ct);
            response.EnsureSuccessStatusCode();
            var result = await response.Content.ReadFromJsonAsync<TrainResponse>(ct);
            return result ?? throw new InvalidOperationException("Train response was null");
        }
    }

    public class XGBoostFixture
    {
        [JsonPropertyName("date")]
        public string Date { get; set; } = "";

        [JsonPropertyName("home_team")]
        public string HomeTeam { get; set; } = "";

        [JsonPropertyName("away_team")]
        public string AwayTeam { get; set; } = "";

        [JsonPropertyName("neutral")]
        public bool Neutral { get; set; } = true;
    }

    public class XGBoostPrediction
    {
        [JsonPropertyName("home_team")]
        public string HomeTeam { get; set; } = "";

        [JsonPropertyName("away_team")]
        public string AwayTeam { get; set; } = "";

        [JsonPropertyName("prob_away_win")]
        public double ProbAwayWin { get; set; }

        [JsonPropertyName("prob_draw")]
        public double ProbDraw { get; set; }

        [JsonPropertyName("prob_home_win")]
        public double ProbHomeWin { get; set; }

        [JsonPropertyName("expected_home_goals")]
        public double ExpectedHomeGoals { get; set; }

        [JsonPropertyName("expected_away_goals")]
        public double ExpectedAwayGoals { get; set; }

        [JsonPropertyName("top_pick")]
        public string TopPick { get; set; } = "";
    }

    public class PredictRequest
    {
        [JsonPropertyName("fixtures")]
        public List<XGBoostFixture> Fixtures { get; set; } = [];
    }

    public class PredictResponse
    {
        [JsonPropertyName("predictions")]
        public List<XGBoostPrediction> Predictions { get; set; } = [];
    }

    public class TrainResponse
    {
        [JsonPropertyName("status")]
        public string Status { get; set; } = "";

        [JsonPropertyName("metrics")]
        public Dictionary<string, object> Metrics { get; set; } = [];

        [JsonPropertyName("paths")]
        public Dictionary<string, string> Paths { get; set; } = [];
    }
}
