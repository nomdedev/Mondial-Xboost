using Microsoft.EntityFrameworkCore;
using MudBlazor.Services;
using MondialXboost.Web;
using MondialXboost.Web.Components;
using MondialXboost.Web.DAL;
using MondialXboost.Web.Services;
using MondialXboost.Web.Services.Simulation;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
builder.Services.AddMudServices();

builder.Services.Configure<MondialXboostConfig>(builder.Configuration.GetSection("MondialXboost"));
var ConnectionString = builder.Configuration.GetConnectionString("MondialXboost") ?? 
    throw new ArgumentNullException("No connection string found in the config!");


builder.Services.AddDbContext<MondialXboostDbContext>(options => options.UseSqlite(ConnectionString));

builder.Services.AddScoped<CsvImportService>();
builder.Services.AddScoped<PredictionService>();
builder.Services.AddScoped<EvaluationService>();
builder.Services.AddScoped<SnapshotService>();
builder.Services.AddScoped<SimulationService>();
builder.Services.AddScoped<ReadmeSnapshotExportService>();
builder.Services.AddHttpClient<RankingRefreshService>((sp, client) =>
{
    var options = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<MondialXboostConfig>>().Value;
    client.Timeout = TimeSpan.FromSeconds(30);
    client.DefaultRequestHeaders.UserAgent.ParseAdd(options.RankingRefreshUserAgent);
});
builder.Services.AddHttpClient<ApiFootballService>((sp, client) =>
{
    var options = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<MondialXboostConfig>>().Value;
    client.BaseAddress = new Uri(options.ApiFootballBaseUrl);
    client.Timeout = TimeSpan.FromSeconds(45);
    client.DefaultRequestHeaders.Add("User-Agent", "MondialXboost");
    if (!string.IsNullOrWhiteSpace(options.ApiFootballApiKey))
    {
        client.DefaultRequestHeaders.Add("x-apisports-key", options.ApiFootballApiKey);
    }
});
builder.Services.AddHttpClient<AvailabilityNewsService>((sp, client) =>
{
    var options = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<MondialXboostConfig>>().Value;
    client.BaseAddress = new Uri(options.OpenRouterBaseUrl);
    client.Timeout = TimeSpan.FromSeconds(60);
    client.DefaultRequestHeaders.UserAgent.ParseAdd(options.AvailabilityRefreshUserAgent);
});
builder.Services.AddHttpClient<XGBoostBridgeService>((sp, client) =>
{
    var options = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<MondialXboostConfig>>().Value;
    client.BaseAddress = new Uri(options.XGBoostBridgeUrl);
    client.Timeout = TimeSpan.FromSeconds(60);
});

var app = builder.Build();
var exportReadmeSnapshots = args.Any(arg => string.Equals(arg, "--export-readme-snapshots", StringComparison.OrdinalIgnoreCase));

using (var Scope = app.Services.CreateScope())
{
    var Config = Scope.ServiceProvider.GetRequiredService<Microsoft.Extensions.Options.IOptions<MondialXboostConfig>>().Value;
    var CsvImporterService = Scope.ServiceProvider.GetRequiredService<CsvImportService>();
    if (Config.RankingRefreshOnStartup && !exportReadmeSnapshots)
    {
        try
        {
            var RankingRefresh = Scope.ServiceProvider.GetRequiredService<RankingRefreshService>();
            var RankingReport = await RankingRefresh.RefreshAsync();
            foreach (var note in RankingReport.Notes)
                app.Logger.LogInformation("{Note}", note);
            foreach (var error in RankingReport.Errors)
                app.Logger.LogWarning("{Error}", error);

            if (RankingReport.AnyFileUpdated)
            {
                var Db = Scope.ServiceProvider.GetRequiredService<MondialXboostDbContext>();
                var HasImportedData =
                    await Db.Groups.AnyAsync() &&
                    await Db.Teams.AnyAsync() &&
                    await Db.Fixtures.AnyAsync() &&
                    await Db.Results.AnyAsync();

                if (HasImportedData)
                    await CsvImporterService.ImportRatingsOnlyAsync();
            }
        }
        catch (Exception ex)
        {
            app.Logger.LogWarning(ex, "Ranking refresh failed during startup. Existing CSV data will be used.");
        }
    }

    await CsvImporterService.ImportIfNeededAsync();
}

if (exportReadmeSnapshots)
{
    using var scope = app.Services.CreateScope();
    var exporter = scope.ServiceProvider.GetRequiredService<ReadmeSnapshotExportService>();
    await exporter.ExportAsync();
    return;
}

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();


app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
