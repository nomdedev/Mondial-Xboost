// Lab — unified operator view for live training, models, features and predictions.

(function () {
  'use strict';

  const POLL_INTERVAL_MS = MXConfig.pollIntervalMs;
  let isTraining = false;
  let progressInterval = null;

  function log(message, type = 'info') {
    const container = document.getElementById('lab-log');
    if (!container) return;
    const time = new Date().toLocaleTimeString('es-AR');
    const colorClass = type === 'error' ? 'text-red-400' : type === 'success' ? 'text-emerald-400' : type === 'warning' ? 'text-amber-400' : 'text-[var(--muted)]';
    const p = document.createElement('p');
    p.className = 'text-sm';
    p.innerHTML = `<span class="${colorClass}">[${MXEscape(time)}]</span> ${MXEscape(message)}`;
    container.prepend(p);
    // Keep last 100 lines.
    while (container.children.length > 100) {
      container.lastChild.remove();
    }
  }

  function setTrainStatus(status, type = 'warning') {
    const el = document.getElementById('lab-train-status');
    if (!el) return;
    el.textContent = status;
    el.className = `mx-badge mx-badge--${type}`;
  }

  function setProgress(percent, text) {
    const bar = document.getElementById('lab-train-progress');
    const txt = document.getElementById('lab-train-progress-text');
    if (bar) bar.style.width = `${percent}%`;
    if (txt) txt.textContent = text;
  }

  function startProgressAnimation() {
    let pct = 0;
    setProgress(0, 'Inicializando…');
    if (progressInterval) clearInterval(progressInterval);
    progressInterval = setInterval(() => {
      if (pct < 90) {
        pct += Math.random() * 3;
        if (pct > 90) pct = 90;
        setProgress(pct, `Entrenando… ${Math.round(pct)}%`);
      }
    }, 800);
  }

  function stopProgressAnimation() {
    if (progressInterval) clearInterval(progressInterval);
    progressInterval = null;
    setProgress(100, 'Completado');
  }

  async function train(engine) {
    if (isTraining) return;
    isTraining = true;

    const xgbBtn = document.getElementById('lab-train-xgboost');
    const rfBtn = document.getElementById('lab-train-rf');
    if (xgbBtn) xgbBtn.disabled = true;
    if (rfBtn) rfBtn.disabled = true;

    setTrainStatus('Entrenando…', 'info');
    log(`Iniciando entrenamiento ${engine}…`, 'info');
    startProgressAnimation();

    const startedAt = Date.now();
    try {
      const result = await MXApi.post(`/train?engine=${engine}`, {});
      const metrics = result.metrics || {};
      document.getElementById('lab-train-acc').textContent = metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(2)}%` : '-';
      document.getElementById('lab-train-loss').textContent = metrics.log_loss ? metrics.log_loss.toFixed(4) : '-';
      document.getElementById('lab-train-top').textContent = metrics.top_feature || '-';
      document.getElementById('lab-train-model').textContent = result.status || '-';

      const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
      document.getElementById('lab-train-eta').textContent = `Terminó en ${elapsed}s`;

      stopProgressAnimation();
      setTrainStatus('Listo', 'success');
      log(`Entrenamiento ${engine} finalizado. Accuracy: ${metrics.accuracy ? (metrics.accuracy * 100).toFixed(2) + '%' : '-'}.`, 'success');
      MXNotify.success(`Modelo ${engine} entrenado en ${elapsed}s.`);

      // Refresh dependent data.
      loadStats();
      loadModels();
      loadFeatures();
    } catch (err) {
      stopProgressAnimation();
      setProgress(0, 'Error');
      setTrainStatus('Error', 'danger');
      log(`Error entrenando ${engine}: ${err.message}`, 'error');
      MXNotify.error('Error en entrenamiento: ' + err.message);
    } finally {
      isTraining = false;
      if (xgbBtn) xgbBtn.disabled = false;
      if (rfBtn) rfBtn.disabled = false;
    }
  }

  async function loadStats() {
    try {
      const [stats, metrics] = await Promise.all([
        MXApi.get('/dashboard/stats'),
        MXApi.get('/dashboard/metrics'),
      ]);
      const totalMatchesEl = document.getElementById('lab-total-matches');
      const totalTeamsEl = document.getElementById('lab-total-teams');
      const bestAccEl = document.getElementById('lab-best-accuracy');
      const featuresCountEl = document.getElementById('lab-features-count');
      if (totalMatchesEl) totalMatchesEl.textContent = MXFormat.number(stats.total_matches);
      if (totalTeamsEl) totalTeamsEl.textContent = MXFormat.number(stats.teams);
      if (bestAccEl) bestAccEl.textContent = MXFormat.pct(metrics.accuracy, 2);
      if (featuresCountEl) featuresCountEl.textContent = MXFormat.number(metrics.feature_count);
    } catch (err) {
      log('Error cargando estadísticas: ' + err.message, 'error');
    }
  }

  async function loadModels() {
    try {
      const data = await MXApi.get('/dashboard/models');
      const container = document.getElementById('lab-models-list');
      if (!container) return;
      const models = data.models || [];
      if (models.length === 0) {
        container.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>No hay modelos entrenados.</p></div>`;
        return;
      }
      container.innerHTML = models.slice(0, 6).map(m => `
        <div class="flex items-center justify-between p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)]">
          <div class="min-w-0">
            <p class="font-medium truncate" title="${MXEscape(m.name)}">${MXEscape(m.name)}</p>
            <p class="text-xs text-[var(--muted)]">${MXFormat.date(m.created)} · ${m.feature_cols?.length || 0} features</p>
          </div>
          <span class="mx-badge mx-badge--info">${m.size_mb} MB</span>
        </div>
      `).join('');
    } catch (err) {
      const container = document.getElementById('lab-models-list');
      if (container) MXUI.setError('lab-models-list', err.message);
    }
  }

  async function loadFeatures() {
    try {
      const data = await MXApi.get('/dashboard/features');
      const container = document.getElementById('lab-feature-bars');
      if (!container) return;
      const features = data.features || [];
      if (features.length === 0) {
        container.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Entrená un modelo para ver la importancia de features.</p></div>`;
        return;
      }
      const maxImp = Math.max(...features.map(f => f.importance));
      container.innerHTML = features.slice(0, 10).map(f => {
        const pct = maxImp > 0 ? (f.importance / maxImp * 100).toFixed(1) : 0;
        return `
          <div class="flex items-center gap-3">
            <span class="w-32 sm:w-48 text-sm truncate" title="${MXEscape(f.feature)}">${MXEscape(f.feature)}</span>
            <div class="flex-1 mx-feature-bar"><div class="mx-feature-bar__fill" style="width: ${pct}%"></div></div>
            <span class="w-16 text-right text-xs font-variant-numeric">${f.importance.toFixed(4)}</span>
          </div>
        `;
      }).join('');
    } catch (err) {
      const container = document.getElementById('lab-feature-bars');
      if (container) MXUI.setError('lab-feature-bars', err.message);
    }
  }

  async function loadWcSummary() {
    try {
      const [fixtures, predictions] = await Promise.all([
        MXApi.get('/wc_fixtures'),
        MXApi.get('/wc_predictions'),
      ]);
      const fx = Array.isArray(fixtures) ? fixtures : [];
      const pr = Array.isArray(predictions) ? predictions : [];
      const statusEl = document.getElementById('lab-wc-status');
      const container = document.getElementById('lab-wc-summary');
      if (statusEl) statusEl.textContent = `${fx.length} fixtures · ${pr.length} predicciones generadas`;
      if (!container) return;
      if (pr.length === 0) {
        container.innerHTML = `<div class="mx-state mx-state--empty"><span class="material-symbols-outlined" aria-hidden="true">info</span><p>Todavía no hay predicciones. Usá "Regenerar" para crearlas.</p></div>`;
        return;
      }
      const groups = {};
      for (const p of pr) {
        const g = p.group || '?';
        groups[g] = (groups[g] || 0) + 1;
      }
      const homePicks = pr.filter(p => p.top_pick === 'Home').length;
      const drawPicks = pr.filter(p => p.top_pick === 'Draw').length;
      const awayPicks = pr.filter(p => p.top_pick === 'Away').length;
      const recentMatches = pr.slice(0, 4).map(p =>
        window.MXMatchCard ? window.MXMatchCard.render(p, { subtitle: `Grupo ${p.group || '?'}` }) : ''
      ).join('');
      container.innerHTML = `
        <div class="grid grid-cols-3 gap-3 mb-4">
          <div class="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-center"><p class="text-xs text-[var(--muted)]">Local</p><p class="text-xl font-bold mx-pick--home">${homePicks}</p></div>
          <div class="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-center"><p class="text-xs text-[var(--muted)]">Empate</p><p class="text-xl font-bold mx-pick--draw">${drawPicks}</p></div>
          <div class="p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] text-center"><p class="text-xs text-[var(--muted)]">Visitante</p><p class="text-xl font-bold mx-pick--away">${awayPicks}</p></div>
        </div>
        <p class="text-sm text-[var(--muted)] mb-4">Grupos con predicciones: ${Object.keys(groups).sort().join(', ')}</p>
        ${recentMatches ? `<div class="grid grid-cols-1 sm:grid-cols-2 gap-3"><h4 class="text-xs font-semibold uppercase tracking-wider text-[var(--muted)] col-span-full">Últimas predicciones</h4>${recentMatches}</div>` : ''}
      `;
    } catch (err) {
      const statusEl = document.getElementById('lab-wc-status');
      if (statusEl) statusEl.textContent = 'Error al cargar predicciones: ' + err.message;
    }
  }

  async function regenerate() {
    const btn = document.getElementById('lab-regenerate-btn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-outlined animate-spin" aria-hidden="true">refresh</span> Regenerando…';
    }
    log('Regenerando predicciones del Mundial…', 'info');
    try {
      const data = await MXApi.post('/wc_regenerate', {});
      const count = data.count || 0;
      log(`${count} predicciones regeneradas.`, 'success');
      MXNotify.success(`${count} predicciones regeneradas.`);
      loadWcSummary();
    } catch (err) {
      log('Error regenerando predicciones: ' + err.message, 'error');
      MXNotify.error('Error al regenerar: ' + err.message);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-outlined" aria-hidden="true">refresh</span> Regenerar';
      }
    }
  }

  function init() {
    const panel = document.getElementById('mx-panel-lab');
    if (!panel) return;

    const xgbBtn = document.getElementById('lab-train-xgboost');
    const rfBtn = document.getElementById('lab-train-rf');
    const regenBtn = document.getElementById('lab-regenerate-btn');

    if (xgbBtn) xgbBtn.addEventListener('click', () => train('xgboost'));
    if (rfBtn) rfBtn.addEventListener('click', () => train('random_forest'));
    if (regenBtn) regenBtn.addEventListener('click', regenerate);

    // Initial load and polling when Lab is active.
    function refresh() {
      loadStats();
      loadModels();
      loadFeatures();
      loadWcSummary();
    }

    refresh();
    setInterval(() => {
      if (document.querySelector('#mx-panel-lab.active')) {
        refresh();
      }
    }, POLL_INTERVAL_MS);

    window.addEventListener('mx:tab:switch', (e) => {
      if (e.detail.tabId === 'lab') refresh();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
