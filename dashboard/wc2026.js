// World Cup 2026 predictions panel.

(function () {
  'use strict';

  const POLL_INTERVAL_MS = MXConfig.pollIntervalMs;

  let fixtures = [];
  let predictions = [];
  let lastGroupsHtml = '';

  function formatPct(value) {
    if (value === undefined || value === null || Number.isNaN(value)) return '-';
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function pickClass(pick) {
    if (pick === 'Home') return 'mx-pick--home';
    if (pick === 'Draw') return 'mx-pick--draw';
    return 'mx-pick--away';
  }

  function pickLabel(pick) {
    if (pick === 'Home') return 'Local';
    if (pick === 'Draw') return 'Empate';
    if (pick === 'Away') return 'Visitante';
    return '-';
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

  function findPrediction(fixture) {
    return predictions.find(p =>
      p.home_team === fixture.home_team &&
      p.away_team === fixture.away_team &&
      p.date === fixture.date
    );
  }

  function renderMatchRow(fixture, pred) {
    const home = MXEscape(fixture.home_team);
    const away = MXEscape(fixture.away_team);
    const date = MXEscape(fixture.date);
    if (!pred) {
      return `<tr class="border-b border-[var(--border)]">
        <td class="py-2">${date}</td>
        <td class="py-2">${home}</td>
        <td class="py-2">${away}</td>
        <td colspan="5" class="py-2 text-[var(--muted)]">Sin predicción</td>
      </tr>`;
    }
    const pick = pred.top_pick || '-';
    return `
      <tr class="border-b border-[var(--border)]">
        <td class="py-2">${date}</td>
        <td class="py-2 font-medium">${home}</td>
        <td class="py-2 font-medium">${away}</td>
        <td class="py-2 text-right">${formatPct(pred.prob_home_win)}</td>
        <td class="py-2 text-right">${formatPct(pred.prob_draw)}</td>
        <td class="py-2 text-right">${formatPct(pred.prob_away_win)}</td>
        <td class="py-2 text-right">${pred.expected_home_goals ?? '-'}-${pred.expected_away_goals ?? '-'}</td>
        <td class="py-2 ${pickClass(pick)} font-semibold">${pickLabel(pick)}</td>
      </tr>
    `;
  }

  function renderGroups() {
    const container = document.getElementById('wc-groups');
    if (!container) return;

    if (fixtures.length === 0 || predictions.length === 0) {
      container.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>No hay fixtures o predicciones cargadas.</p></div>`;
      lastGroupsHtml = '';
      return;
    }

    const fixtureGroups = groupBy(fixtures, 'group');
    const groupNames = Object.keys(fixtureGroups).sort();

    let html = '';
    for (const group of groupNames) {
      html += `
        <div class="mx-card">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-bold">Grupo ${MXEscape(group)}</h3>
            <span class="text-sm text-[var(--muted)]">${fixtureGroups[group].length} partidos</span>
          </div>
          <div class="overflow-x-auto">
            <table class="mx-table">
              <thead>
                <tr>
                  <th>Fecha</th>
                  <th>Local</th>
                  <th>Visitante</th>
                  <th class="num">Local</th>
                  <th class="num">Empate</th>
                  <th class="num">Visitante</th>
                  <th class="num">xG</th>
                  <th>Pick</th>
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

    if (html !== lastGroupsHtml) {
      container.innerHTML = html;
      lastGroupsHtml = html;
    }
  }

  async function loadData() {
    try {
      const [fx, pr] = await Promise.all([
        MXApi.get('/wc_fixtures'),
        MXApi.get('/wc_predictions'),
      ]);

      if (!fx.error) fixtures = Array.isArray(fx) ? fx : [];
      if (!pr.error) predictions = Array.isArray(pr) ? pr : [];

      renderGroups();
      updateStatus(`Cargados ${fixtures.length} fixtures y ${predictions.length} predicciones.`);
    } catch (err) {
      console.error('[WC2026] loadData error', err);
      updateStatus(`Error al cargar: ${err.message}`);
      MXNotify.error('No se pudieron cargar las predicciones del Mundial.');
    }
  }

  async function predictSingle() {
    const homeInput = document.getElementById('wc-pred-home');
    const awayInput = document.getElementById('wc-pred-away');
    const dateInput = document.getElementById('wc-pred-date');
    const resultDiv = document.getElementById('wc-single-result');

    const home = (homeInput?.value || '').trim();
    const away = (awayInput?.value || '').trim();
    const date = dateInput?.value || '2026-07-15';

    if (!home || !away) {
      resultDiv.innerHTML = '<p class="text-red-400">Ingresá ambos equipos.</p>';
      return;
    }

    resultDiv.innerHTML = '<p class="text-[var(--muted)]">Prediciendo…</p>';

    try {
      const data = await MXApi.post('/wc_predict', { home_team: home, away_team: away, date });
      if (data.error) {
        resultDiv.innerHTML = `<p class="text-red-400">Error: ${MXEscape(data.error)}</p>`;
        return;
      }

      resultDiv.innerHTML = `
        <div class="mx-card" style="background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 100%);">
          <h4 class="font-semibold mb-4 text-lg">${MXEscape(home)} vs ${MXEscape(away)}</h4>
          <div class="grid grid-cols-3 gap-4 text-center mb-4">
            <div>
              <p class="text-sm text-[var(--muted)]">Local</p>
              <p class="text-3xl font-bold mx-pick--home">${formatPct(data.prob_home_win)}</p>
            </div>
            <div>
              <p class="text-sm text-[var(--muted)]">Empate</p>
              <p class="text-3xl font-bold mx-pick--draw">${formatPct(data.prob_draw)}</p>
            </div>
            <div>
              <p class="text-sm text-[var(--muted)]">Visitante</p>
              <p class="text-3xl font-bold mx-pick--away">${formatPct(data.prob_away_win)}</p>
            </div>
          </div>
          <div class="flex flex-wrap gap-3 text-sm">
            <span class="mx-badge mx-badge--info">Pick: <span class="${pickClass(data.top_pick)}">${pickLabel(data.top_pick)}</span></span>
            <span class="mx-badge mx-badge--info">xG: ${data.expected_home_goals ?? '-'} - ${data.expected_away_goals ?? '-'}</span>
          </div>
        </div>
      `;
    } catch (err) {
      resultDiv.innerHTML = `<p class="text-red-400">Error: ${MXEscape(err.message)}</p>`;
    }
  }

  async function regenerate() {
    if (!confirm('¿Regenerar todas las predicciones de la fase de grupos? Puede tardar unos segundos.')) return;

    const btn = document.getElementById('wc-regenerate-btn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-outlined animate-spin" aria-hidden="true">refresh</span> Regenerando…';
    }
    updateStatus('Regenerando predicciones con el modelo canónico…');

    try {
      const data = await MXApi.post('/wc_regenerate', {});
      if (data.error) {
        updateStatus(`Error: ${data.error}`);
        MXNotify.error(data.error);
        return;
      }
      predictions = data.predictions || [];
      renderGroups();
      updateStatus(`Regeneradas ${data.count} predicciones.`);
      MXNotify.success(`Regeneradas ${data.count} predicciones.`);
    } catch (err) {
      updateStatus(`Error: ${err.message}`);
      MXNotify.error('Error al regenerar predicciones: ' + err.message);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined" aria-hidden="true">refresh</span> Regenerar predicciones';
      }
    }
  }

  function updateStatus(message) {
    const el = document.getElementById('wc-status');
    if (el) el.textContent = message;
  }

  async function loadDashboardModels() {
    try {
      const data = await MXApi.get('/dashboard/models');
      const container = document.getElementById('models-list');
      if (!container) return;

      const models = data.models || [];
      if (models.length === 0) {
        container.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>No hay modelos entrenados.</p></div>`;
        return;
      }

      container.innerHTML = models.map(m => `
        <div class="mx-card">
          <div class="flex items-center justify-between mb-2">
            <h4 class="font-semibold truncate" title="${MXEscape(m.name)}">${MXEscape(m.name)}</h4>
            <span class="mx-badge mx-badge--info">${m.size_mb} MB</span>
          </div>
          <p class="text-sm text-[var(--muted)] mb-2">Creado: ${MXFormat.date(m.created)}</p>
          <p class="text-xs text-[var(--muted)]">${m.feature_cols?.length || 0} features</p>
        </div>
      `).join('');
    } catch (err) {
      const container = document.getElementById('models-list');
      if (container) MXUI.setError('models-list', err.message);
    }
  }

  async function loadDashboardFeatures() {
    try {
      const data = await MXApi.get('/dashboard/features');
      const container = document.getElementById('feature-bars');
      if (!container) return;

      const features = data.features || [];
      if (features.length === 0) {
        container.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Entrená un modelo para ver la importancia de features.</p></div>`;
        return;
      }

      const maxImp = Math.max(...features.map(f => f.importance));
      container.innerHTML = features.map((f, index) => {
        const pct = maxImp > 0 ? (f.importance / maxImp * 100).toFixed(1) : 0;
        return `
          <div class="flex items-center gap-4">
            <span class="w-40 sm:w-52 text-sm truncate" title="${MXEscape(f.feature)}">${MXEscape(f.feature)}</span>
            <div class="flex-1 mx-feature-bar">
              <div class="mx-feature-bar__fill" style="width: ${pct}%"></div>
            </div>
            <span class="w-20 text-right text-sm font-variant-numeric">${f.importance.toFixed(4)}</span>
            <span class="w-10 text-right text-xs text-[var(--muted)]">${pct}%</span>
          </div>
        `;
      }).join('');
    } catch (err) {
      const container = document.getElementById('feature-bars');
      if (container) MXUI.setError('feature-bars', err.message);
    }
  }

  async function loadDashboardStats() {
    try {
      const stats = await MXApi.get('/dashboard/stats');
      const metrics = await MXApi.get('/dashboard/metrics');

      document.getElementById('total-matches').textContent = MXFormat.number(stats.total_matches);
      document.getElementById('total-teams').textContent = MXFormat.number(stats.teams);
      document.getElementById('best-accuracy').textContent = MXFormat.pct(metrics.accuracy, 2);
      document.getElementById('models-count').textContent = MXFormat.number(metrics.feature_count);

      const dateRange = stats.date_range || {};
      document.getElementById('dashboard-date-range').innerHTML = `
        <p class="text-2xl font-bold mb-1">${MXFormat.date(dateRange.min)} — ${MXFormat.date(dateRange.max)}</p>
        <p class="text-sm text-[var(--muted)]">${MXFormat.number(stats.total_matches)} partidos · ${MXFormat.number(stats.teams)} equipos</p>
      `;

      renderOutcomeChart(stats.outcome_distribution || {});
    } catch (err) {
      MXNotify.error('Error al cargar el panel: ' + err.message);
    }
  }

  function renderOutcomeChart(distribution) {
    const ctx = document.getElementById('dashboard-outcome-chart');
    if (!ctx) return;

    const data = [distribution.home || 0, distribution.draw || 0, distribution.away || 0];

    if (window._outcomeChart) window._outcomeChart.destroy();
    window._outcomeChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Local', 'Empate', 'Visitante'],
        datasets: [{
          data,
          backgroundColor: ['#34d399', '#fbbf24', '#60a5fa'],
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#f3f4f6' } }
        }
      }
    });
  }

  function init() {
    loadData();
    setInterval(loadData, POLL_INTERVAL_MS);

    const regenerateBtn = document.getElementById('wc-regenerate-btn');
    if (regenerateBtn) regenerateBtn.addEventListener('click', regenerate);

    const predBtn = document.getElementById('wc-predict-btn');
    if (predBtn) predBtn.addEventListener('click', predictSingle);

    // Load dashboard tabs lazily when shown.
    window.addEventListener('mx:tab:switch', (e) => {
      if (e.detail.tabId === 'dashboard') loadDashboardStats();
      if (e.detail.tabId === 'models') loadDashboardModels();
      if (e.detail.tabId === 'features') loadDashboardFeatures();
    });

    // If dashboard is the initial tab, load it.
    if (document.querySelector('#mx-panel-dashboard.active')) {
      loadDashboardStats();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
