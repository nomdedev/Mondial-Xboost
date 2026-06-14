using Microsoft.EntityFrameworkCore;
using MondialXboost.Web.DAL;
using MondialXboost.Web.Models;

namespace MondialXboost.Web.Services.Simulation
{
    public sealed class SimulationPredictionContext
    {
        private readonly IReadOnlyList<Team> _teams;
        private readonly XGBoostBridgeService? _bridge;

        private SimulationPredictionContext(IReadOnlyList<Team> teams, XGBoostBridgeService? bridge)
        {
            _teams = teams;
            _bridge = bridge;
        }

        public static async Task<SimulationPredictionContext> CreateAsync(
            MondialXboostDbContext db,
            MondialXboostConfig config,
            XGBoostBridgeService? bridge = null,
            CancellationToken ct = default)
        {
            var teams = await db.Teams.AsNoTracking().ToListAsync(ct);
            return new SimulationPredictionContext(teams, config.XGBoostPredictorEnabled ? bridge : null);
        }

        public Task<MatchPredictionResult> PredictPairAsync(string homeId, string awayId, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            var fixture = new Fixture { Id = $"pair:{homeId}:{awayId}", HomeTeamId = homeId, AwayTeamId = awayId, NeutralVenue = true };
            return Task.FromResult(Predict(fixture, ct));
        }

        private MatchPredictionResult Predict(Fixture fixture, CancellationToken ct)
        {
            var context = BuildContext(fixture);
            var prediction = MatchPredictionBuilder.BuildAsync(context, _bridge, ct).GetAwaiter().GetResult();

            return new MatchPredictionResult
            {
                Fixture = fixture,
                HomeTeamName = context.HomeTeam.Name,
                AwayTeamName = context.AwayTeam.Name,
                Predictions = [prediction],
                BestPrediction = prediction
            };
        }

        private MatchContext BuildContext(Fixture fixture)
        {
            var home = _teams.FirstOrDefault(t => t.Id == fixture.HomeTeamId) ?? new Team { Id = fixture.HomeTeamId, Name = fixture.HomeTeamId };
            var away = _teams.FirstOrDefault(t => t.Id == fixture.AwayTeamId) ?? new Team { Id = fixture.AwayTeamId, Name = fixture.AwayTeamId };

            return new MatchContext
            {
                Fixture = fixture,
                HomeTeam = home,
                AwayTeam = away
            };
        }
    }
}
