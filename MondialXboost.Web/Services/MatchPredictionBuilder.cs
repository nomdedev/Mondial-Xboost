using MondialXboost.Web.Models;
using MondialXboost.Web.Probability;

namespace MondialXboost.Web.Services
{
    public static class MatchPredictionBuilder
    {
        public const string XgboostPredictorName = "XGBoost (ML)";

        public static async Task<MatchPrediction> BuildAsync(
            MatchContext context,
            XGBoostBridgeService? bridge,
            CancellationToken ct = default)
        {
            if (bridge is null)
                return Degraded(context, "no ML bridge configured");

            var fixture = context.Fixture;
            var kickoff = fixture.KickoffUtc?.UtcDateTime.ToString("yyyy-MM-dd")
                ?? DateTimeOffset.UtcNow.ToString("yyyy-MM-dd");

            var fixtures = new[]
            {
                new XGBoostFixture
                {
                    Date = kickoff,
                    HomeTeam = context.HomeTeam.Name,
                    AwayTeam = context.AwayTeam.Name,
                    Neutral = fixture.NeutralVenue
                }
            };

            try
            {
                var predictions = await bridge.PredictAsync(fixtures, ct);
                var pred = predictions.FirstOrDefault();

                if (pred is null)
                    return Degraded(context, "bridge returned no predictions");

                var outcome = new OutcomeProbabilities(pred.ProbHomeWin, pred.ProbDraw, pred.ProbAwayWin).Normalize();
                var scoreline = ProbabilityHelper.PoissonScoreline(pred.ExpectedHomeGoals, pred.ExpectedAwayGoals);
                var mostLikely = scoreline.MostLikelyScoreline();

                return new MatchPrediction
                {
                    PredictorName = XgboostPredictorName,
                    PredictorPriority = 1,
                    FixtureId = fixture.Id,
                    HomeTeamId = context.HomeTeamId,
                    AwayTeamId = context.AwayTeamId,
                    Outcome = outcome,
                    ExpectedHomeGoals = pred.ExpectedHomeGoals,
                    ExpectedAwayGoals = pred.ExpectedAwayGoals,
                    Scoreline = scoreline,
                    MostLikelyScore = mostLikely,
                    Explanation = $"XGBoost: {context.HomeTeam.Name} {pred.ExpectedHomeGoals:0.00} - {pred.ExpectedAwayGoals:0.00} {context.AwayTeam.Name}. Top pick: {pred.TopPick}.",
                    Drivers = [$"Distribución 1X2: H {pred.ProbHomeWin:0.0%} / D {pred.ProbDraw:0.0%} / A {pred.ProbAwayWin:0.0%}"],
                    FeaturesUsed =
                    [
                        "Elo diff",
                        "Forma reciente (5 y 10 partidos)",
                        "Goles anotados/recibidos",
                        "Head-to-head",
                        "XGBoost ensemble"
                    ],
                    FeaturesMissing = [],
                    Sources = [SourceMetadata.HistoricalResultsCsv],
                    Degraded = false
                };
            }
            catch (Exception ex)
            {
                return Degraded(context, ex.Message);
            }
        }

        private static MatchPrediction Degraded(MatchContext context, string reason) => new()
        {
            PredictorName = XgboostPredictorName,
            PredictorPriority = 1,
            FixtureId = context.Fixture.Id,
            HomeTeamId = context.HomeTeamId,
            AwayTeamId = context.AwayTeamId,
            Outcome = OutcomeProbabilities.Uniform,
            Explanation = $"XGBoost degradado: {reason}",
            Drivers = ["Bridge a Python no disponible"],
            FeaturesMissing = ["xgboost bridge"],
            Sources = [],
            Degraded = true
        };
    }
}
