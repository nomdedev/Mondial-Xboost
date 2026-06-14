using MondialXboost.Web.Models;
using MondialXboost.Web.Probability;
using MondialXboost.Web.Services;

namespace MondialXboost.Web.Predictors
{
    public class XGBoostPredictor : IPredictor
    {
        private readonly XGBoostBridgeService _bridge;

        public string Name => "XGBoost (ML)";
        public int Priority => 6;

        public XGBoostPredictor(XGBoostBridgeService bridge)
        {
            _bridge = bridge;
        }

        public MatchPrediction Predict(MatchContext context)
        {
            var fixture = context.Fixture;
            var kickoff = fixture.KickoffUtc?.UtcDateTime.ToString("yyyy-MM-dd") ?? DateTimeOffset.UtcNow.ToString("yyyy-MM-dd");

            try
            {
                // Synchronous call for IPredictor.Predict; in production consider async variant
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

                var predictions = _bridge.PredictAsync(fixtures).GetAwaiter().GetResult();
                var pred = predictions.FirstOrDefault();

                if (pred is null)
                {
                    return DegradedPrediction(context, "bridge returned no predictions");
                }

                var outcome = new OutcomeProbabilities(pred.ProbHomeWin, pred.ProbDraw, pred.ProbAwayWin).Normalize();
                var scoreline = ProbabilityHelper.PoissonScoreline(pred.ExpectedHomeGoals, pred.ExpectedAwayGoals);
                var mostLikely = scoreline.MostLikelyScoreline();

                return new MatchPrediction
                {
                    PredictorName = Name,
                    PredictorPriority = Priority,
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
                return DegradedPrediction(context, ex.Message);
            }
        }

        private static MatchPrediction DegradedPrediction(MatchContext context, string reason)
        {
            return new MatchPrediction
            {
                PredictorName = "XGBoost (ML)",
                PredictorPriority = 6,
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
}
