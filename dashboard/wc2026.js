// World Cup 2026 predictions panel.

(function () {
  'use strict';

  const POLL_INTERVAL_MS = MXConfig.pollIntervalMs;

  let fixtures = [];
  let predictions = [];
  let lastGroupsHtml = '';

  const ROUND_LABELS = {
    group_stage: 'Fase de grupos',
    round_of_32: '32avos de final',
    round_of_16: 'Octavos de final',
    quarter_finals: 'Cuartos de final',
    semi_finals: 'Semifinales',
    final: 'Final',
    third_place: 'Tercer puesto',
  };

  // A small seeded palette for team shields.
  const SHIELD_COLORS = [
    '#126a5a', '#315a9f', '#c98926', '#7c3aed', '#db2777',
    '#dc2626', '#059669', '#2563eb', '#d97706', '#0891b2',
  ];

  function formatPct(value) {
    if (value === undefined || value === null || Number.isNaN(value)) return '-';
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function formatGoals(value) {
    if (value === undefined || value === null || Number.isNaN(value)) return '-';
    return Number(value).toFixed(1);
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

  function stringHash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = (h << 5) - h + str.charCodeAt(i);
      h |= 0;
    }
    return Math.abs(h);
  }

  function teamShield(name) {
    const words = name.trim().split(/\s+/);
    let initials = words[0][0];
    if (words.length > 1) initials += words[words.length - 1][0];
    initials = initials.slice(0, 2).toUpperCase();
    const color = SHIELD_COLORS[stringHash(name) % SHIELD_COLORS.length];
    return `<div class="mx-match__shield" style="background:${color};" aria-hidden="true">${MXEscape(initials)}</div>`;
  }

  function determineWinner(pred) {
    if (pred.winner) return pred.winner;
    if (pred.top_pick === 'Home') return pred.home_team;
    if (pred.top_pick === 'Away') return pred.away_team;
    return null;
  }

  function roundLabel(round, fallback) {
    if (round && ROUND_LABELS[round]) return ROUND_LABELS[round];
    return fallback || '';
  }

  function renderProbabilityBar(label, value, type) {
    const pct = formatPct(value);
    const width = value === undefined || value === null || Number.isNaN(value) ? 0 : Math.max(0, Math.min(100, Number(value) * 100));
    return `
      <div class="mx-match__prob-row">
        <span>${MXEscape(label)}</span>
        <div class="mx-match__prob-bar" aria-hidden="true">
          <div class="mx-match__prob-fill mx-match__prob-fill--${type}" style="width: ${width.toFixed(1)}%"></div>
        </div>
        <span class="num">${pct}</span>
      </div>
    `;
  }

  /**
   * Render a match card. `match` can be a flat prediction object or a knockout
   * object with a nested `prediction` property.
   */
  function renderMatchCard(match, opts = {}) {
    const pred = match.prediction || match;
    const home = MXEscape(pred.home_team);
    const away = MXEscape(pred.away_team);
    const winner = determineWinner(match);
    const homeWinner = winner === pred.home_team;
    const awayWinner = winner === pred.away_team;
    const pick = pred.top_pick || '-';
    const date = pred.date ? MXFormat.date(pred.date) : '';
    const subtitle = opts.subtitle || roundLabel(match.round, pred.group ? `Grupo ${pred.group}` : '');

    return `
      <div class="mx-match">
        <div class="mx-match__header">
          <span class="flex items-center gap-1">
            <span class="material-symbols-outlined" style="font-size:1rem">event</span>
            ${date ? `<span>${date}</span>` : ''}
            ${date && subtitle ? '<span aria-hidden="true">·</span>' : ''}
            ${subtitle ? `<span>${MXEscape(subtitle)}</span>` : ''}
          </span>
          ${homeWinner || awayWinner ? '<span class="material-symbols-outlined mx-match__winner-icon" title="Ganador predicho">emoji_events</span>' : ''}
        </div>
        <div class="mx-match__teams">
          <div class="mx-match__team ${homeWinner ? 'mx-match__team--winner' : ''}">
            ${teamShield(pred.home_team)}
            <span class="mx-match__name">${home}</span>
            <span class="mx-match__score">${formatGoals(pred.expected_home_goals)}</span>
          </div>
          <div class="mx-match__team ${awayWinner ? 'mx-match__team--winner' : ''}">
            ${teamShield(pred.away_team)}
            <span class="mx-match__name">${away}</span>
            <span class="mx-match__score">${formatGoals(pred.expected_away_goals)}</span>
          </div>
        </div>
        <div class="mx-match__probabilities">
          ${renderProbabilityBar('Local', pred.prob_home_win, 'home')}
          ${renderProbabilityBar('Empate', pred.prob_draw, 'draw')}
          ${renderProbabilityBar('Visit.', pred.prob_away_win, 'away')}
        </div>
        <div class="mx-match__footer">
          <span class="mx-match__pick ${pickClass(pick) ? `mx-match__pick--${pick === 'Home' ? 'home' : pick === 'Draw' ? 'draw' : 'away'}` : ''}">
            Pick: ${pickLabel(pick)}
          </span>
          <span class="text-xs text-[var(--muted)]">xG ${formatGoals(pred.expected_home_goals)} - ${formatGoals(pred.expected_away_goals)}</span>
        </div>
      </div>
    `;
  }

  function renderGroupTable(teams) {
    if (!teams || teams.length === 0) return '';
    return `
      <table class="mx-mini-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Equipo</th>
            <th class="num">Pts</th>
            <th class="num">GF</th>
            <th class="num">DG</th>
          </tr>
        </thead>
        <tbody>
          ${teams.map((t, i) => `
            <tr>
              <td class="text-[var(--muted)]">${i + 1}</td>
              <td>${MXEscape(t.team)}</td>
              <td class="num font-semibold">${t.points}</td>
              <td class="num">${Number(t.goals_for).toFixed(1)}</td>
              <td class="num">${t.goal_diff > 0 ? '+' : ''}${Number(t.goal_diff).toFixed(1)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }

  function renderGroups() {
    const container = document.getElementById('wc-groups');
    if (!container) return;

    if (fixtures.length === 0 || predictions.length === 0) {
      container.innerHTML = `<div class="mx-state mx-state--empty col-span-full"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>No hay fixtures o predicciones cargadas.</p></div>`;
      lastGroupsHtml = '';
      return;
    }

    const fixtureGroups = groupBy(fixtures, 'group');
    const groupNames = Object.keys(fixtureGroups).sort();

    let html = '';
    for (const group of groupNames) {
      const groupFixtures = fixtureGroups[group].slice().sort((a, b) => (a.date || '').localeCompare(b.date || ''));
      html += `
        <div class="mx-group-card">
          <div class="mx-group-card__title">
            <span>Grupo ${MXEscape(group)}</span>
            <span class="text-xs text-[var(--muted)] font-normal">${groupFixtures.length} partidos</span>
          </div>
          <div class="mx-group-card__matches">
            ${groupFixtures.map(fx => renderMatchCard(findPrediction(fx) || fx, { subtitle: `Grupo ${group}` })).join('')}
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

      resultDiv.innerHTML = renderMatchCard({ ...data, date });
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
    const container = document.getElementById('models-list');
    if (!container) return;

    try {
      const data = await MXApi.get('/dashboard/models');
      const models = data.models || [];
      if (models.length === 0) {
        container.innerHTML = `<div class="mx-state mx-state--empty col-span-full"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>No hay modelos entrenados.</p></div>`;
        return;
      }

      // Heuristic: the canonical model is the one loaded by the API.
      const canonicalName = data.canonical || 'xgboost_football';

      container.innerHTML = models.map(m => {
        const isCanonical = m.name === canonicalName;
        const metrics = m.metrics || {};
        return `
          <div class="mx-model-card ${isCanonical ? 'mx-model-card--canonical' : ''}">
            <div class="mx-model-card__header">
              <div>
                <div class="mx-model-card__name" title="${MXEscape(m.name)}">${MXEscape(m.name)}</div>
                <div class="mx-model-card__meta">${MXFormat.date(m.created)} · ${m.size_mb} MB · ${m.feature_cols?.length || 0} features</div>
              </div>
              ${isCanonical ? `<span class="mx-model-card__canonical-badge"><span class="material-symbols-outlined" style="font-size:0.875rem">verified</span>Canónico</span>` : `<span class="mx-badge mx-badge--info">${m.size_mb} MB</span>`}
            </div>
            <div class="mx-model-card__metrics">
              <div class="mx-model-card__metric">
                <div class="mx-model-card__metric-value ${metrics.accuracy ? 'text-emerald-400' : 'text-[var(--muted)]'}">${metrics.accuracy ? MXFormat.pct(metrics.accuracy, 2) : '-'}</div>
                <div class="mx-model-card__metric-label">Accuracy</div>
              </div>
              <div class="mx-model-card__metric">
                <div class="mx-model-card__metric-value ${metrics.log_loss ? 'text-amber-400' : 'text-[var(--muted)]'}">${metrics.log_loss ? metrics.log_loss.toFixed(4) : '-'}</div>
                <div class="mx-model-card__metric-label">Log loss</div>
              </div>
              <div class="mx-model-card__metric" title="${MXEscape(metrics.top_feature || '')}">
                <div class="mx-model-card__metric-value text-[var(--ink)] truncate">${metrics.top_feature ? MXEscape(metrics.top_feature.split('_').slice(0, 2).join('_')) : '-'}</div>
                <div class="mx-model-card__metric-label">Top feature</div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    } catch (err) {
      MXUI.setError('models-list', err.message);
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

  function renderGroupStandings(standings, fixtures) {
    const container = document.getElementById('tournament-groups');
    if (!container) return;

    const groups = Object.keys(standings).sort();
    if (groups.length === 0) {
      container.innerHTML = `<div class="mx-state mx-state--empty col-span-full"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Sin posiciones de grupos.</p></div>`;
      return;
    }

    const fixturesByGroup = groupBy(fixtures || [], 'group');

    container.innerHTML = groups.map(group => {
      const teams = standings[group];
      const groupFixtures = (fixturesByGroup[group] || []).slice().sort((a, b) => (a.date || '').localeCompare(b.date || ''));
      return `
        <div class="mx-group-card">
          <div class="mx-group-card__title">Grupo ${MXEscape(group)}</div>
          <div class="mx-group-card__matches">
            ${groupFixtures.map(fx => renderMatchCard(fx, { subtitle: `Grupo ${group}` })).join('')}
          </div>
          ${renderGroupTable(teams)}
        </div>
      `;
    }).join('');
  }

  function renderKnockoutRound(matches, containerId, gridClass) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!matches || matches.length === 0) {
      container.innerHTML = `<div class="mx-state mx-state--empty col-span-full"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Sin datos</p></div>`;
      return;
    }
    container.innerHTML = `
      <div class="${gridClass}">
        ${matches.map(m => renderMatchCard(m)).join('')}
      </div>
    `;
  }

  function renderTournament(data) {
    if (!data || data.error) {
      MXNotify.error('No se pudo cargar la simulación del torneo.');
      return;
    }

    const championEl = document.getElementById('tournament-champion-name');
    const championDetailEl = document.getElementById('tournament-champion-detail');
    if (championEl) championEl.textContent = MXEscape(data.champion || '-');
    if (championDetailEl) {
      const final = data.knockout?.final?.[0];
      if (final) {
        const pred = final.prediction || final;
        championDetailEl.innerHTML = `Final: <strong>${MXEscape(pred.home_team)} ${formatGoals(pred.expected_home_goals)} - ${formatGoals(pred.expected_away_goals)} ${MXEscape(pred.away_team)}</strong>`;
      } else {
        championDetailEl.textContent = 'Simulación completa del Mundial 2026';
      }
    }

    const knockout = data.knockout || {};
    renderKnockoutRound(knockout.round_of_32, 'tournament-round-of-32', 'mx-round-grid mx-round-grid--4');
    renderKnockoutRound(knockout.round_of_16, 'tournament-round-of-16', 'mx-round-grid mx-round-grid--4');
    renderKnockoutRound(knockout.quarter_finals, 'tournament-quarter', 'mx-round-grid mx-round-grid--2');
    renderKnockoutRound(knockout.semi_finals, 'tournament-semi', 'mx-round-grid mx-round-grid--2');

    const finalContainer = document.getElementById('tournament-final');
    if (finalContainer) {
      if (knockout.final && knockout.final.length > 0) {
        finalContainer.innerHTML = renderMatchCard(knockout.final[0]);
      } else {
        finalContainer.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Sin datos</p></div>`;
      }
    }

    const thirdContainer = document.getElementById('tournament-third');
    if (thirdContainer) {
      if (knockout.third_place && knockout.third_place.length > 0) {
        thirdContainer.innerHTML = renderMatchCard(knockout.third_place[0]);
      } else {
        thirdContainer.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Sin datos</p></div>`;
      }
    }

    renderGroupStandings(data.group_stage?.standings || {}, data.group_stage?.fixtures || []);
  }

  async function loadTournament() {
    try {
      const data = await MXApi.get('/wc_tournament');
      renderTournament(data);
    } catch (err) {
      console.error('[Tournament] load error', err);
      const container = document.getElementById('dashboard-tournament');
      if (container) container.innerHTML = `<div class="mx-state mx-state--error" role="alert"><span class="material-symbols-outlined" aria-hidden="true">error</span><p>Error al cargar la simulación del torneo: ${MXEscape(err.message)}</p></div>`;
    }
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
      if (e.detail.tabId === 'dashboard') {
        loadDashboardStats();
        loadTournament();
      }
      if (e.detail.tabId === 'models') loadDashboardModels();
      if (e.detail.tabId === 'features') loadDashboardFeatures();
    });

    // If dashboard is the initial tab, load it.
    if (document.querySelector('#mx-panel-dashboard.active')) {
      loadDashboardStats();
      loadTournament();
    }
  }

  // Expose reusable helpers for lab.js and other panels.
  window.MXMatchCard = { render: renderMatchCard };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
