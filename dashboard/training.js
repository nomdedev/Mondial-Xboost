// Training monitor — polls the local training_server.py and renders live charts.

(function () {
  'use strict';

  const POLL_INTERVAL_MS = MXConfig.pollIntervalMs;
  const isLocalhost = MXConfig.isLocalhost;

  let charts = {};
  let lastEventCount = 0;

  function formatDuration(seconds) {
    if (seconds === undefined || seconds === null || Number.isNaN(seconds)) return '-';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    return `${m}m ${s}s`;
  }

  function statusColor(status) {
    switch (status) {
      case 'running': return 'text-blue-400';
      case 'completed': return 'text-emerald-400';
      case 'error': return 'text-red-400';
      default: return 'text-[var(--muted)]';
    }
  }

  function createChart(canvasId, type, label, color, fill = false) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    return new Chart(ctx, {
      type,
      data: {
        labels: [],
        datasets: [{
          label,
          data: [],
          borderColor: color,
          backgroundColor: type === 'line' ? `${color}33` : color,
          fill,
          tension: 0.2,
          pointRadius: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 500 },
        scales: {
          x: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
          y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
        },
        plugins: {
          legend: { labels: { color: '#f3f4f6' } }
        }
      }
    });
  }

  function initCharts() {
    charts.acc = createChart('tm-acc-chart', 'line', 'Test Accuracy %', '#34d399', true);
    charts.loss = createChart('tm-loss-chart', 'line', 'Log Loss', '#fbbf24', true);
    charts.gap = createChart('tm-gap-chart', 'bar', 'Overfit Gap %', '#60a5fa');
    charts.dist = createChart('tm-dist-chart', 'bar', 'Trials', '#a78bfa');
  }

  function updateText(id, text, colorClass = null) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (colorClass) {
      el.className = `mx-metric__value ${colorClass}`;
    } else {
      el.className = 'mx-metric__value';
    }
  }

  function updateStatusHeader(status, phase, completed, total, elapsed, eta) {
    const statusEl = document.getElementById('tm-status');
    if (statusEl) {
      statusEl.textContent = status || 'Inactivo';
      statusEl.className = `mx-metric__value ${statusColor(status)}`;
    }

    updateText('tm-phase', phase || '-');
    updateText('tm-trials', total ? `${completed} / ${total}` : '-');
    updateText('tm-time', `${formatDuration(elapsed)} / ${formatDuration(eta)}`);

    const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
    const progressText = document.getElementById('tm-progress-text');
    const progressBar = document.getElementById('tm-progress-bar');
    if (progressText) progressText.textContent = `${progress}%`;
    if (progressBar) progressBar.style.width = `${progress}%`;
  }

  function updateMetrics(bestAcc, bestLoss, bestGap, canonicalAcc) {
    updateText('tm-best-acc', bestAcc ? `${bestAcc.toFixed(2)}%` : '-', 'text-emerald-400');
    updateText('tm-best-loss', bestLoss ? bestLoss.toFixed(4) : '-', 'text-amber-400');
    updateText('tm-best-gap', bestGap !== undefined ? `${bestGap.toFixed(2)}%` : '-', 'text-blue-400');
    updateText('tm-canonical-acc', canonicalAcc ? `${(canonicalAcc * 100).toFixed(2)}%` : '-', 'text-purple-400');
  }

  function updateEventLog(events) {
    if (!events || events.length === lastEventCount) return;
    lastEventCount = events.length;

    const container = document.getElementById('tm-event-log');
    if (!container) return;

    container.innerHTML = events.slice().reverse().map(e => {
      const time = e.time ? new Date(e.time).toLocaleTimeString('es-AR') : '';
      const levelClass = e.level === 'error' ? 'text-red-400' : e.level === 'warning' ? 'text-amber-400' : 'text-[var(--muted)]';
      return `<p class="text-sm"><span class="${levelClass}">[${MXEscape(time)}]</span> ${MXEscape(e.message)}</p>`;
    }).join('');
    container.scrollTop = 0;
  }

  function updateTopTrials(runs) {
    const tbody = document.getElementById('tm-top-trials');
    if (!tbody) return;

    if (!runs || runs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="py-4 text-[var(--muted)]">Sin datos de entrenamiento</td></tr>';
      return;
    }

    const top = [...runs].sort((a, b) => (b.test_accuracy || 0) - (a.test_accuracy || 0)).slice(0, 15);
    tbody.innerHTML = top.map(run => {
      const params = run.params || {};
      const paramsStr = Object.entries(params)
        .slice(0, 4)
        .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toPrecision(3) : v}`)
        .join(', ');
      return `
        <tr class="border-b border-[var(--border)]">
          <td class="py-2">${run.batch || '-'}</td>
          <td class="py-2">${run.trial || '-'}</td>
          <td class="py-2 text-right text-emerald-400">${(run.test_accuracy || 0).toFixed(2)}%</td>
          <td class="py-2 text-right">${(run.val_accuracy || 0).toFixed(2)}%</td>
          <td class="py-2 text-right">${(run.log_loss || 0).toFixed(4)}</td>
          <td class="py-2 text-right">${(run.overfit_gap || 0).toFixed(2)}%</td>
          <td class="py-2 text-xs text-[var(--muted)]">${MXEscape(paramsStr)}</td>
        </tr>
      `;
    }).join('');
  }

  function updateCharts(runs) {
    if (!runs || runs.length === 0) return;

    const labels = runs.map((_, i) => `#${i + 1}`);
    const accData = runs.map(r => r.test_accuracy || 0);
    const lossData = runs.map(r => r.log_loss || 0);
    const gapData = runs.map(r => r.overfit_gap || 0);

    let bestSoFar = -Infinity;
    const accBest = accData.map(v => {
      bestSoFar = Math.max(bestSoFar, v);
      return bestSoFar;
    });

    if (charts.acc) {
      charts.acc.data.labels = labels;
      charts.acc.data.datasets[0].data = accData;
      if (charts.acc.data.datasets.length === 1) {
        charts.acc.data.datasets.push({
          label: 'Mejor hasta ahora',
          data: accBest,
          borderColor: '#10b981',
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0,
          tension: 0.1,
        });
      } else {
        charts.acc.data.datasets[1].data = accBest;
      }
      charts.acc.update();
    }

    if (charts.loss) {
      charts.loss.data.labels = labels;
      charts.loss.data.datasets[0].data = lossData;
      charts.loss.update();
    }

    if (charts.gap) {
      charts.gap.data.labels = labels;
      charts.gap.data.datasets[0].data = gapData;
      charts.gap.update();
    }

    if (charts.dist) {
      const bins = [55, 56, 57, 58, 59, 60, 61];
      const counts = bins.map((b, i) => {
        const next = bins[i + 1] || 100;
        return accData.filter(v => v >= b && v < next).length;
      });
      charts.dist.data.labels = bins.map(b => `${b}%`);
      charts.dist.data.datasets[0].data = counts;
      charts.dist.update();
    }
  }

  async function updateGpuStatus() {
    try {
      const data = await MXApi.training.get('/gpu');
      const el = document.getElementById('tm-gpu-status');
      if (!el || !data) return;
      const isPass = data.status === 'PASS';
      el.className = `mx-badge ${isPass ? 'mx-badge--success' : 'mx-badge--danger'}`;
      el.textContent = `GPU: ${data.device || 'unknown'} (${data.status})`;
    } catch {
      const el = document.getElementById('tm-gpu-status');
      if (el) {
        el.className = 'mx-badge mx-badge--warning';
        el.textContent = 'GPU: no detectado';
      }
    }
  }

  async function updateAdaptiveStatus() {
    try {
      const data = await MXApi.training.get('/train/status');
      const statusEl = document.getElementById('tm-adapt-status');
      const startBtn = document.getElementById('tm-adapt-start');
      const stopBtn = document.getElementById('tm-adapt-stop');
      if (!data || !statusEl) return;

      const report = data.report || {};
      const status = report.status || (data.running ? 'RUNNING' : 'IDLE');
      const message = report.message || (data.running ? 'Entrenamiento adaptivo en curso…' : 'Listo para iniciar');
      statusEl.innerHTML = `<span class="font-semibold">${MXEscape(status)}</span> — ${MXEscape(message)}`;

      if (startBtn) startBtn.disabled = data.running;
      if (stopBtn) stopBtn.disabled = !data.running;
    } catch {
      const statusEl = document.getElementById('tm-adapt-status');
      if (statusEl) statusEl.textContent = 'No se pudo consultar el estado del entrenamiento.';
    }
  }

  async function updateDashboard() {
    try {
      const [status, results, canonical] = await Promise.all([
        MXApi.training.get('/status'),
        MXApi.training.get('/results'),
        MXApi.training.get('/canonical'),
      ]);

      if (status) {
        updateStatusHeader(
          status.status,
          status.phase,
          status.completed_trials || 0,
          status.total_trials || 0,
          status.elapsed_seconds || 0,
          status.eta_seconds || 0
        );
        updateMetrics(
          status.best_test_accuracy,
          status.best_log_loss,
          status.best_overfit_gap,
          canonical?.accuracy
        );
        updateEventLog(status.recent_events || []);
      }

      const runs = status?.runs || results?.all_runs || [];
      updateTopTrials(runs);
      updateCharts(runs);
    } catch (err) {
      console.error('[Training] update error', err);
    }
  }

  async function startAdaptiveTraining() {
    const body = {
      max_auto_batches: parseInt(document.getElementById('tm-adapt-batches').value, 10) || 3,
      trials_per_batch: parseInt(document.getElementById('tm-adapt-trials').value, 10) || 30,
      cv_folds: parseInt(document.getElementById('tm-adapt-folds').value, 10) || 3,
      cv_embargo: parseInt(document.getElementById('tm-adapt-embargo').value, 10) || 60,
    };

    try {
      const data = await MXApi.training.post('/train/adaptive', body);
      updateAdaptiveStatus();
      MXNotify.info(data.message || data.status || 'Entrenamiento iniciado');
    } catch (err) {
      MXNotify.error('Error al iniciar entrenamiento: ' + err.message);
    }
  }

  async function stopAdaptiveTraining() {
    MXNotify.warning('En este entorno el botón solo desconecta la UI; el proceso del servidor debe detenerse manualmente.');
    const startBtn = document.getElementById('tm-adapt-start');
    const stopBtn = document.getElementById('tm-adapt-stop');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
  }

  function setOfflineState() {
    const offline = document.getElementById('training-offline');
    const content = document.getElementById('training-content');
    if (offline) offline.classList.remove('hidden');
    if (content) content.classList.add('hidden');
  }

  function init() {
    const panel = document.getElementById('mx-panel-training');
    if (!panel) return;

    if (!isLocalhost || !MXConfig.trainingUrl) {
      setOfflineState();
      return;
    }

    initCharts();
    updateDashboard();
    setInterval(updateDashboard, POLL_INTERVAL_MS);

    updateGpuStatus();
    updateAdaptiveStatus();
    setInterval(() => {
      updateGpuStatus();
      updateAdaptiveStatus();
    }, POLL_INTERVAL_MS);

    const startBtn = document.getElementById('tm-adapt-start');
    const stopBtn = document.getElementById('tm-adapt-stop');
    if (startBtn) startBtn.addEventListener('click', startAdaptiveTraining);
    if (stopBtn) stopBtn.addEventListener('click', stopAdaptiveTraining);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
