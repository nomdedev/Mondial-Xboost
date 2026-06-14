using CsvHelper.Configuration.Attributes;

namespace MondialXboost.Web.Models.CsvModels
{
    public class GroupCsvRow
    {
        [Name("group")]
        public string Group { get; set; } = "";
        [Name("team")]
        public string Team { get; set; } = "";
    }
}
