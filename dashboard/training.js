// Training monitor dashboard — polls training server and renders live charts.

const TRAINING_URL = 'http://localhost:8765';
const POLL_INTERVAL_MS = 2000;

let charts = {};
let lastEventCount = 0;

function formatDuration(seconds) {
    if (!seconds && seconds !== 0) return '-';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    return `${m}m ${s}s`;
}

function statusColor(status) {
    switch (status) {
        case 'running': return 'text-blue-400';
        case 'completed': return 'text-green-400';
        case 'error': return 'text-red-400';
        default: return 'text-slate-400';
    }
}

function createChart(canvasId, type, label, color, fill = false) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    return new Chart(ctx, {
        type: type,
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: color,
                backgroundColor: type === 'line' ? `${color}33` : color,
                fill: fill,
                tension: 0.2,
                pointRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 500 },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
            },
            plugins: {
                legend: { labels: { color: '#e2e8f0' } }
            }
        }
    });
}

function initCharts() {
    charts.acc = createChart('tm-acc-chart', 'line', 'Test Accuracy %', '#10b981', true);
    charts.loss = createChart('tm-loss-chart', 'line', 'Log Loss', '#f59e0b', true);
    charts.gap = createChart('tm-gap-chart', 'bar', 'Overfit Gap %', '#3b82f6');
    charts.dist = createChart('tm-dist-chart', 'bar', 'Trials', '#8b5cf6');
}

function updateText(id, text, colorClass = null) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (colorClass) {
        el.className = `text-2xl font-bold ${colorClass}`;
    }
}

function updateStatusHeader(status, phase, completed, total, elapsed, eta) {
    const statusEl = document.getElementById('tm-status');
    statusEl.textContent = status || 'idle';
    statusEl.className = `text-2xl font-bold ${statusColor(status)}`;

    document.getElementById('tm-phase').textContent = phase || '-';
    document.getElementById('tm-trials').textContent = total ? `${completed} / ${total}` : '-';
    document.getElementById('tm-time').textContent = `${formatDuration(elapsed)} / ${formatDuration(eta)}`;

    const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
    document.getElementById('tm-progress-text').textContent = `${progress}%`;
    document.getElementById('tm-progress-bar').style.width = `${progress}%`;
}

function updateMetrics(bestAcc, bestLoss, bestGap, canonicalAcc) {
    updateText('tm-best-acc', bestAcc ? `${bestAcc.toFixed(2)}%` : '-', 'text-green-400');
    updateText('tm-best-loss', bestLoss ? bestLoss.toFixed(4) : '-', 'text-yellow-400');
    updateText('tm-best-gap', bestGap !== undefined ? `${bestGap.toFixed(2)}%` : '-', 'text-blue-400');
    updateText('tm-canonical-acc', canonicalAcc ? `${(canonicalAcc * 100).toFixed(2)}%` : '-', 'text-purple-400');
}

function updateEventLog(events) {
    if (!events || events.length === lastEventCount) return;
    lastEventCount = events.length;

    const container = document.getElementById('tm-event-log');
    if (!container) return;

    container.innerHTML = events.slice().reverse().map(e => {
        const time = e.time ? new Date(e.time).toLocaleTimeString() : '';
        return `<p class="text-slate-300"><span class="text-slate-500">[${time}]</span> ${escapeHtml(e.message)}</p>`;
    }).join('');
    container.scrollTop = 0;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateTopTrials(runs) {
    const tbody = document.getElementById('tm-top-trials');
    if (!tbody) return;

    if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="py-4 text-slate-400">No training data yet</td></tr>';
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
            <tr class="border-b border-slate-700">
                <td class="py-2">${run.batch || '-'}</td>
                <td class="py-2">${run.trial || '-'}</td>
                <td class="py-2 text-green-400">${(run.test_accuracy || 0).toFixed(2)}%</td>
                <td class="py-2">${(run.val_accuracy || 0).toFixed(2)}%</td>
                <td class="py-2">${(run.log_loss || 0).toFixed(4)}</td>
                <td class="py-2">${(run.overfit_gap || 0).toFixed(2)}%</td>
                <td class="py-2 text-xs text-slate-400">${escapeHtml(paramsStr)}</td>
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

    // Best-so-far accuracy line.
    let bestSoFar = -Infinity;
    const accBest = accData.map(v => {
        bestSoFar = Math.max(bestSoFar, v);
        return bestSoFar;
    });

    // Update line charts.
    if (charts.acc) {
        charts.acc.data.labels = labels;
        charts.acc.data.datasets[0].data = accData;
        if (charts.acc.data.datasets.length === 1) {
            charts.acc.data.datasets.push({
                label: 'Best so far',
                data: accBest,
                borderColor: '#34d399',
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

    // Distribution histogram.
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

async function fetchJson(path) {
    try {
        const response = await fetch(`${TRAINING_URL}${path}`);
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        return null;
    }
}

async function updateDashboard() {
    const [status, results, canonical] = await Promise.all([
        fetchJson('/status'),
        fetchJson('/results'),
        fetchJson('/canonical'),
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

    const runs = results?.all_runs || [];
    updateTopTrials(runs);
    updateCharts(runs);
}

// Initialize when training tab is shown or on load if active.
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    updateDashboard();
    setInterval(updateDashboard, POLL_INTERVAL_MS);
});
