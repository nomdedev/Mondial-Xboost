namespace MondialXboost.Web
{
    public class MondialXboostConfig
    {
        public int SimulationCount { get; set; }
        public int? SimulationSeed { get; set; }
        public string ApiFootballBaseUrl { get; set; } = "https://v3.football.api-sports.io/";
        public string? ApiFootballApiKey { get; set; }
        public int ApiFootballLeagueId { get; set; }
        public int ApiFootballSeason { get; set; }
        public string OpenRouterBaseUrl { get; set; } = "https://openrouter.ai/api/v1/";
        public string? OpenRouterApiKey { get; set; }
        public string OpenRouterModel { get; set; } = "openai/gpt-4o-mini";
        public string[] AvailabilitySourceUrls { get; set; } =
        [
            "https://www.espn.com/soccer/story/_/id/48572979/2026-fifa-world-cup-injuries-tracker-which-stars-miss-latest-info",
            "https://talksport.com/football/world-cup/4311921/world-cup-2026-injury-tracker-full-squads-messi/"
        ];
        public string AvailabilityRefreshUserAgent { get; set; } = "MondialXboost";
        public int AvailabilityMaxArticleChars { get; set; } = 24000;
        public bool AvailabilityRequireCrossCheck { get; set; } = true;
        public string XGBoostBridgeUrl { get; set; } = "http://127.0.0.1:8000";
        public bool XGBoostPredictorEnabled { get; set; } = true;
    }

    public static class MondialXboostDataFiles
    {
        public const string GroupsCsv = "wc2026_groups.csv";
        public const string EloCsv = "elo_snapshot.csv";
        public const string FifaRankingsCsv = "fifa_rankings.csv";
        public const string HistoricalResultsCsv = "historical_results.csv";
    }
}
