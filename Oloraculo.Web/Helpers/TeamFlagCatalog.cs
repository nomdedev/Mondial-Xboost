using System.Globalization;

namespace Oloraculo.Web.Helpers
{
    public static class TeamFlagCatalog
    {
        private static readonly Dictionary<string, string> Overrides = new(StringComparer.OrdinalIgnoreCase)
        {
            ["aland-islands"] = "ax",
            ["american-samoa"] = "as",
            ["aotearoa-new-zealand"] = "nz",
            ["aruba"] = "aw",
            ["bosnia-and-herzegovina"] = "ba",
            ["brunei-darussalam"] = "bn",
            ["cabo-verde"] = "cv",
            ["cape-verde"] = "cv",
            ["canary-islands"] = "ic",
            ["china-pr"] = "cn",
            ["chinese-taipei"] = "tw",
            ["congo-dr"] = "cd",
            ["curacao"] = "cw",
            ["czechoslovakia"] = "cz",
            ["dem-rep-of-congo"] = "cd",
            ["dr-congo"] = "cd",
            ["czechia"] = "cz",
            ["england"] = "gb-eng",
            ["faroe-islands"] = "fo",
            ["german-dr"] = "de",
            ["hong-kong-china"] = "hk",
            ["ir-iran"] = "ir",
            ["ivory-coast"] = "ci",
            ["korea-dpr"] = "kp",
            ["korea-republic"] = "kr",
            ["kyrgyz-republic"] = "kg",
            ["macau"] = "mo",
            ["micronesia"] = "fm",
            ["north-korea"] = "kp",
            ["north-macedonia"] = "mk",
            ["northern-ireland"] = "gb-nir",
            ["palestine"] = "ps",
            ["republic-of-ireland"] = "ie",
            ["reunion"] = "re",
            ["south-korea"] = "kr",
            ["st-lucia"] = "lc",
            ["st-vincent-and-the-grenadines"] = "vc",
            ["syria"] = "sy",
            ["scotland"] = "gb-sct",
            ["turkey"] = "tr",
            ["united-states"] = "us",
            ["venezuela"] = "ve",
            ["wales"] = "gb-wls"
        };

        private static readonly Lazy<IReadOnlyDictionary<string, string>> RegionCodesByTeamId = new(CreateRegionCodeLookup);

        public static string? CodeFor(string? teamId, string? teamName = null)
        {
            foreach (var candidate in Candidates(teamId, teamName))
            {
                if (Overrides.TryGetValue(candidate, out var code))
                    return code;

                if (RegionCodesByTeamId.Value.TryGetValue(candidate, out code))
                    return code;
            }

            return null;
        }

        private static IEnumerable<string> Candidates(string? teamId, string? teamName)
        {
            if (!string.IsNullOrWhiteSpace(teamId))
                yield return teamId.Trim().ToLowerInvariant();

            if (!string.IsNullOrWhiteSpace(teamName))
                yield return TeamNameNormalizer.ToId(teamName);
        }

        private static IReadOnlyDictionary<string, string> CreateRegionCodeLookup()
        {
            var lookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var regions = CultureInfo.GetCultures(CultureTypes.SpecificCultures)
                .Select(culture =>
                {
                    try
                    {
                        return new RegionInfo(culture.Name);
                    }
                    catch (ArgumentException)
                    {
                        return null;
                    }
                })
                .OfType<RegionInfo>()
                .DistinctBy(region => region.TwoLetterISORegionName);

            foreach (var region in regions)
            {
                Add(region.EnglishName, region.TwoLetterISORegionName);
                Add(region.NativeName, region.TwoLetterISORegionName);
                Add(region.DisplayName, region.TwoLetterISORegionName);
            }

            return lookup;

            void Add(string name, string code)
            {
                if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(code) || code.Length != 2)
                    return;

                lookup.TryAdd(TeamNameNormalizer.ToId(name), code.ToLowerInvariant());
            }
        }
    }
}
