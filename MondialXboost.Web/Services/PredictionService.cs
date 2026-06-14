using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using MondialXboost.Web.DAL;
using MondialXboost.Web.Models;

namespace MondialXboost.Web.Services
{
    public class PredictionService
    {
        private readonly MondialXboostDbContext _db;
        private readonly MondialXboostConfig _config;
        private readonly XGBoostBridgeService? _bridge;

        public PredictionService(MondialXboostDbContext db, IOptions<MondialXboostConfig> config, XGBoostBridgeService? bridge = null)
        {
            _db = db;
            _config = config.Value;
            _bridge = _config.XGBoostPredictorEnabled ? bridge : null;
        }

        public async Task<MatchPredictionResult?> PredictFixtureAsync(string fixtureId, CancellationToken ct = default)
        {
            var fixture = await _db.Fixtures.FindAsync([fixtureId], ct);
            return fixture is null ? null : await PredictAsync(fixture, ct);
        }

        public async Task<MatchPredictionResult> PredictPairAsync(string homeId, string awayId, CancellationToken ct = default)
        {
            var fixture = new Fixture { Id = $"pair:{homeId}:{awayId}", HomeTeamId = homeId, AwayTeamId = awayId, NeutralVenue = true };
            return await PredictAsync(fixture, ct);
        }

        public async Task<IReadOnlyList<MatchPredictionResult>> PredictFixturesAsync(IEnumerable<Fixture> fixtures, CancellationToken ct = default)
        {
            var fixtureList = fixtures.ToList();
            var results = new List<MatchPredictionResult>(fixtureList.Count);

            foreach (var fixture in fixtureList)
                results.Add(await PredictAsync(fixture, ct));

            return results;
        }

        public async Task<MatchPredictionResult> PredictAsync(Fixture fixture, CancellationToken ct = default)
        {
            var context = await BuildContextAsync(fixture, ct);
            var prediction = await MatchPredictionBuilder.BuildAsync(context, _bridge, ct);

            return new MatchPredictionResult
            {
                Fixture = fixture,
                HomeTeamName = context.HomeTeam.Name,
                AwayTeamName = context.AwayTeam.Name,
                Predictions = [prediction],
                BestPrediction = prediction
            };
        }

        public async Task<MatchContext> BuildContextAsync(Fixture fixture, CancellationToken ct = default)
        {
            var home = await _db.Teams.FindAsync([fixture.HomeTeamId], ct) ?? new Team { Id = fixture.HomeTeamId, Name = fixture.HomeTeamId };
            var away = await _db.Teams.FindAsync([fixture.AwayTeamId], ct) ?? new Team { Id = fixture.AwayTeamId, Name = fixture.AwayTeamId };

            return new MatchContext
            {
                Fixture = fixture,
                HomeTeam = home,
                AwayTeam = away,
                FixtureContext = await _db.FixtureContexts.FindAsync([fixture.Id], ct)
            };
        }
    }
}
