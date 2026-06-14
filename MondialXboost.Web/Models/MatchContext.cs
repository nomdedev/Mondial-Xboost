namespace MondialXboost.Web.Models
{
    public class MatchContext
    {
        public required Fixture Fixture { get; init; }
        public required Team HomeTeam { get; init; }
        public required Team AwayTeam { get; init; }
        public FixtureContext? FixtureContext { get; set; }
        public string HomeTeamId => HomeTeam.Id;
        public string AwayTeamId => AwayTeam.Id;
    }
}
