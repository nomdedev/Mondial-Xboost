// World Cup 2026 predictions dashboard.

const WC_API_URL = window.API_URL || ''; // same origin
const WC_POLL_INTERVAL_MS = 5000;

let fixtures = [];
let predictions = [];

function formatPct(value) {
    if (value === undefined || value === null) return '-';
    return `${(value * 100).toFixed(1)}%`;
}

function pickColor(pick) {
    if (pick === 'Home') return 'text-green-400';
    if (pick === 'Draw') return 'text-yellow-400';
    return 'text-red-400';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function groupBy(items, key) {
    const groups = {};
    for (const item of items) {
        const g = item[key] || 'Unknown';
        groups[g] = groups[g] || [];
        groups[g].push(item);
    }
    return groups;
}

async function fetchJson(path, options = {}) {
    try {
        const response = await fetch(`${WC_API_URL}${path}`, options);
        if (!response.ok) return { error: `HTTP ${response.status}` };
        return await response.json();
    } catch (e) {
        return { error: e.message };
    }
}

async function loadData() {
    try {
        console.log('[WC2026] loadData start');
        const [fx, pr] = await Promise.all([
            fetchJson('/wc_fixtures'),
            fetchJson('/wc_predictions'),
        ]);
        console.log('[WC2026] data', fx, pr);

        if (!fx.error) fixtures = fx;
        if (!pr.error) predictions = pr;

        renderGroups();
        updateStatus(`Loaded ${fixtures.length} fixtures and ${predictions.length} predictions.`);
    } catch (e) {
        console.error('[WC2026] loadData error', e);
        updateStatus(`Load error: ${e.message}`);
    }
}

function findPrediction(fixture) {
    return predictions.find(p =>
        p.home_team === fixture.home_team &&
        p.away_team === fixture.away_team &&
        p.date === fixture.date
    );
}

function renderGroups() {
    const container = document.getElementById('wc-groups');
    if (!container) return;

    if (fixtures.length === 0 || predictions.length === 0) {
        container.innerHTML = '<p class="text-slate-400">No fixtures or predictions loaded.</p>';
        return;
    }

    const fixtureGroups = groupBy(fixtures, 'group');
    const predGroups = groupBy(predictions, 'group');
    const groupNames = Object.keys(fixtureGroups).sort();

    let html = '';
    for (const group of groupNames) {
        html += `
            <div class="card mb-6">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-bold">Group ${escapeHtml(group)}</h3>
                    <span class="text-sm text-slate-400">${fixtureGroups[group].length} matches</span>
                </div>
                <div class="overflow-auto">
                    <table class="w-full text-sm text-left">
                        <thead class="text-slate-400 border-b border-slate-600">
                            <tr>
                                <th class="py-2">Date</th>
                                <th class="py-2">Home</th>
                                <th class="py-2">Away</th>
                                <th class="py-2 text-right">Home</th>
                                <th class="py-2 text-right">Draw</th>
                                <th class="py-2 text-right">Away</th>
                                <th class="py-2 text-right">xG</th>
                                <th class="py-2">Pick</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${fixtureGroups[group].map(fx => renderMatchRow(fx, findPrediction(fx))).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderMatchRow(fixture, pred) {
    const home = escapeHtml(fixture.home_team);
    const away = escapeHtml(fixture.away_team);
    const date = escapeHtml(fixture.date);
    if (!pred) {
        return `<tr class="border-b border-slate-700"><td class="py-2">${date}</td><td class="py-2">${home}</td><td class="py-2">${away}</td><td colspan="5" class="py-2 text-slate-500">No prediction</td></tr>`;
    }
    const pick = pred.top_pick || '-';
    const pickClass = pickColor(pick);
    return `
        <tr class="border-b border-slate-700">
            <td class="py-2">${date}</td>
            <td class="py-2 font-medium">${home}</td>
            <td class="py-2 font-medium">${away}</td>
            <td class="py-2 text-right">${formatPct(pred.prob_home_win)}</td>
            <td class="py-2 text-right">${formatPct(pred.prob_draw)}</td>
            <td class="py-2 text-right">${formatPct(pred.prob_away_win)}</td>
            <td class="py-2 text-right">${pred.expected_home_goals ?? '-'}-${pred.expected_away_goals ?? '-'}</td>
            <td class="py-2 ${pickClass} font-semibold">${pick}</td>
        </tr>
    `;
}

async function predictSingle() {
    const home = document.getElementById('wc-pred-home').value.trim();
    const away = document.getElementById('wc-pred-away').value.trim();
    const date = document.getElementById('wc-pred-date').value || '2026-07-15';
    const resultDiv = document.getElementById('wc-single-result');

    if (!home || !away) {
        resultDiv.innerHTML = '<p class="text-red-400">Ingresá ambos equipos.</p>';
        return;
    }

    resultDiv.innerHTML = '<p class="text-slate-400">Predicting...</p>';
    const data = await fetchJson('/wc_predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: home, away_team: away, date }),
    });

    if (data.error) {
        resultDiv.innerHTML = `<p class="text-red-400">Error: ${escapeHtml(data.error)}</p>`;
        return;
    }

    resultDiv.innerHTML = `
        <div class="card">
            <h4 class="font-semibold mb-3">${escapeHtml(home)} vs ${escapeHtml(away)}</h4>
            <div class="grid grid-cols-3 gap-4 text-center">
                <div><p class="text-slate-400">Home</p><p class="text-2xl font-bold text-green-400">${formatPct(data.prob_home_win)}</p></div>
                <div><p class="text-slate-400">Draw</p><p class="text-2xl font-bold text-yellow-400">${formatPct(data.prob_draw)}</p></div>
                <div><p class="text-slate-400">Away</p><p class="text-2xl font-bold text-red-400">${formatPct(data.prob_away_win)}</p></div>
            </div>
            <p class="mt-3 text-sm text-slate-300">Pick: <span class="font-bold ${pickColor(data.top_pick)}">${data.top_pick}</span></p>
            <p class="text-sm text-slate-300">xG: ${data.expected_home_goals} - ${data.expected_away_goals}</p>
        </div>
    `;
}

async function regenerate() {
    const btn = document.getElementById('wc-regenerate-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Regenerando...';
    }
    updateStatus('Regenerating predictions with current canonical model...');

    const data = await fetchJson('/wc_regenerate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
    });

    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Regenerate predictions';
    }

    if (data.error) {
        updateStatus(`Error: ${data.error}`);
        return;
    }

    predictions = data.predictions || [];
    renderGroups();
    updateStatus(`Regenerated ${data.count} predictions.`);
}

function updateStatus(message) {
    const el = document.getElementById('wc-status');
    if (el) el.textContent = message;
}

function initWC2026() {
    try {
        console.log('[WC2026] init');
        updateStatus('Initializing...');
        loadData();
        setInterval(loadData, WC_POLL_INTERVAL_MS);

        const btn = document.getElementById('wc-regenerate-btn');
        if (btn) btn.addEventListener('click', regenerate);

        const predBtn = document.getElementById('wc-predict-btn');
        if (predBtn) predBtn.addEventListener('click', predictSingle);
    } catch (e) {
        console.error('[WC2026] init error', e);
        updateStatus(`Init error: ${e.message}`);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWC2026);
} else {
    initWC2026();
}
