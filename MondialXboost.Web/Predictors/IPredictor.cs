using MondialXboost.Web.Models;

namespace MondialXboost.Web.Predictors
{
    public interface IPredictor
    {
        string Name { get; }
        int Priority { get; }
        MatchPrediction Predict(MatchContext context);
    }
}
