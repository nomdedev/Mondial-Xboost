/**
 * Mondial-Xboost Dashboard — core shell and API client.
 *
 * Responsibilities:
 * - Theme/CSS variables aligned with MondialXboost.Web (dark lab variant).
 * - Accessible tab navigation.
 * - Shared API client for predictors/api.py (Vercel / same-origin).
 * - Shared API client for training_server.py (localhost only).
 * - Toast notification system.
 * - Loading, error and empty states.
 */

(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // Config
  // -------------------------------------------------------------------------

  const isLocalhost = ['localhost', '127.0.0.1'].includes(window.location.hostname);

  window.MXConfig = {
    // Main API served by predictors/api.py (same-origin in Vercel, :8000 locally).
    apiUrl: window.location.origin,
    // Training monitor served by scripts/training_server.py. Only available locally.
    trainingUrl: isLocalhost ? 'http://localhost:8765' : null,
    pollIntervalMs: isLocalhost ? 2000 : 5000,
    isLocalhost,
  };

  // -------------------------------------------------------------------------
  // Icons (Material Symbols Outlined)
  // -------------------------------------------------------------------------

  const ICONS = {
    dashboard: 'dashboard',
    training: 'model_training',
    predictions: 'sports_soccer',
    models: 'smart_toy',
    features: 'insights',
    agents: 'psychology',
    strategy: 'strategy',
    menu: 'menu',
    close: 'close',
    check: 'check_circle',
    error: 'error',
    warning: 'warning',
    info: 'info',
    refresh: 'refresh',
    arrowBack: 'arrow_back',
    settings: 'tune',
  };

  window.MXIcons = ICONS;

  // -------------------------------------------------------------------------
  // API client
  // -------------------------------------------------------------------------

  async function apiFetch(path, options = {}, baseUrl = MXConfig.apiUrl) {
    const url = `${baseUrl}${path}`;
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          detail = body.detail || JSON.stringify(body);
        } catch {
          detail = await response.text() || detail;
        }
        throw new Error(detail);
      }
      return await response.json();
    } catch (err) {
      console.error(`API error ${path}:`, err);
      throw err;
    }
  }

  window.MXApi = {
    get: (path) => apiFetch(path),
    post: (path, body) => apiFetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    }),
    training: {
      get: (path) => {
        if (!MXConfig.trainingUrl) {
          return Promise.reject(new Error('Training monitor is only available when running training_server.py locally.'));
        }
        return apiFetch(path, {}, MXConfig.trainingUrl);
      },
      post: (path, body) => {
        if (!MXConfig.trainingUrl) {
          return Promise.reject(new Error('Training monitor is only available when running training_server.py locally.'));
        }
        return apiFetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body || {}),
        }, MXConfig.trainingUrl);
      },
    },
  };

  // -------------------------------------------------------------------------
  // Notifications
  // -------------------------------------------------------------------------

  const toastContainer = document.createElement('div');
  toastContainer.id = 'mx-toast-container';
  toastContainer.setAttribute('aria-live', 'polite');
  toastContainer.setAttribute('aria-atomic', 'true');
  document.body.appendChild(toastContainer);

  window.MXNotify = {
    show(message, type = 'info', durationMs = 4000) {
      const el = document.createElement('div');
      el.className = `mx-toast mx-toast--${type}`;
      el.setAttribute('role', 'status');
      const icon = ICONS[type] || ICONS.info;
      el.innerHTML = `
        <span class="material-symbols-outlined" aria-hidden="true">${icon}</span>
        <span class="mx-toast__message">${escapeHtml(String(message))}</span>
      `;
      toastContainer.appendChild(el);
      requestAnimationFrame(() => el.classList.add('mx-toast--visible'));
      setTimeout(() => {
        el.classList.remove('mx-toast--visible');
        setTimeout(() => el.remove(), 300);
      }, durationMs);
    },
    info(msg) { this.show(msg, 'info'); },
    success(msg) { this.show(msg, 'success'); },
    warning(msg) { this.show(msg, 'warning'); },
    error(msg) { this.show(msg, 'error', 6000); },
  };

  // -------------------------------------------------------------------------
  // UI helpers
  // -------------------------------------------------------------------------

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  window.MXEscape = escapeHtml;

  window.MXFormat = {
    pct(value, decimals = 1) {
      if (value === undefined || value === null || Number.isNaN(value)) return '-';
      return `${(Number(value) * 100).toFixed(decimals)}%`;
    },
    number(value, decimals = 0) {
      if (value === undefined || value === null || Number.isNaN(value)) return '-';
      return Number(value).toLocaleString('es-AR', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    },
    date(value) {
      if (!value) return '-';
      try {
        return new Date(value).toLocaleDateString('es-AR');
      } catch {
        return value;
      }
    },
  };

  window.MXUI = {
    setLoading(elementId, loading = true) {
      const el = document.getElementById(elementId);
      if (!el) return;
      if (loading) {
        el.classList.add('mx-loading');
      } else {
        el.classList.remove('mx-loading');
      }
    },
    setError(elementId, message) {
      const el = document.getElementById(elementId);
      if (!el) return;
      el.innerHTML = `
        <div class="mx-state mx-state--error" role="alert">
          <span class="material-symbols-outlined" aria-hidden="true">${ICONS.error}</span>
          <p>${escapeHtml(message)}</p>
        </div>`;
    },
    setEmpty(elementId, message) {
      const el = document.getElementById(elementId);
      if (!el) return;
      el.innerHTML = `
        <div class="mx-state mx-state--empty">
          <span class="material-symbols-outlined" aria-hidden="true">${ICONS.info}</span>
          <p>${escapeHtml(message)}</p>
        </div>`;
    },
  };

  // -------------------------------------------------------------------------
  // Tab navigation (accessible)
  // -------------------------------------------------------------------------

  const TABS = [
    { id: 'dashboard', label: 'Panel', icon: ICONS.dashboard },
    { id: 'training', label: 'Entrenamiento', icon: ICONS.training },
    { id: 'predictions', label: 'Predicciones', icon: ICONS.predictions },
    { id: 'models', label: 'Modelos', icon: ICONS.models },
    { id: 'features', label: 'Features', icon: ICONS.features },
  ];

  function renderNav() {
    const nav = document.getElementById('mx-nav');
    if (!nav) return;
    nav.setAttribute('role', 'tablist');
    nav.setAttribute('aria-label', 'Secciones del dashboard');

    nav.innerHTML = TABS.map((tab, index) => `
      <button
        id="mx-tab-${tab.id}"
        class="mx-nav-item ${index === 0 ? 'active' : ''}"
        role="tab"
        aria-selected="${index === 0 ? 'true' : 'false'}"
        aria-controls="mx-panel-${tab.id}"
        tabindex="${index === 0 ? '0' : '-1'}"
        data-tab="${tab.id}"
      >
        <span class="material-symbols-outlined" aria-hidden="true">${tab.icon}</span>
        <span>${escapeHtml(tab.label)}</span>
      </button>
    `).join('');
  }

  function switchTab(tabId) {
    const panels = document.querySelectorAll('.mx-tab-panel');
    const buttons = document.querySelectorAll('[role="tab"]');

    panels.forEach((panel) => {
      const isTarget = panel.id === `mx-panel-${tabId}`;
      panel.classList.toggle('active', isTarget);
      panel.setAttribute('aria-hidden', isTarget ? 'false' : 'true');
      if (isTarget) panel.removeAttribute('hidden');
      else panel.setAttribute('hidden', '');
    });

    buttons.forEach((btn) => {
      const isTarget = btn.dataset.tab === tabId;
      btn.classList.toggle('active', isTarget);
      btn.setAttribute('aria-selected', isTarget ? 'true' : 'false');
      btn.setAttribute('tabindex', isTarget ? '0' : '-1');
    });

    window.dispatchEvent(new CustomEvent('mx:tab:switch', { detail: { tabId } }));
  }

  function handleNavClick(e) {
    const btn = e.target.closest('[role="tab"]');
    if (!btn) return;
    switchTab(btn.dataset.tab);
  }

  function handleNavKeydown(e) {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const current = tabs.findIndex((t) => t.getAttribute('aria-selected') === 'true');
    let next = current;

    if (e.key === 'ArrowRight') next = (current + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = tabs.length - 1;
    else return;

    e.preventDefault();
    tabs[next].focus();
    switchTab(tabs[next].dataset.tab);
  }

  function initTabs() {
    renderNav();
    const nav = document.getElementById('mx-nav');
    if (nav) {
      nav.addEventListener('click', handleNavClick);
      nav.addEventListener('keydown', handleNavKeydown);
    }

    // Hide all panels except the first.
    const panels = document.querySelectorAll('.mx-tab-panel');
    panels.forEach((panel, index) => {
      if (index !== 0) {
        panel.setAttribute('hidden', '');
        panel.setAttribute('aria-hidden', 'true');
      } else {
        panel.classList.add('active');
        panel.setAttribute('aria-hidden', 'false');
      }
    });
  }

  // -------------------------------------------------------------------------
  // Health / status
  // -------------------------------------------------------------------------

  async function updateApiStatus() {
    const dot = document.getElementById('mx-status-dot');
    const text = document.getElementById('mx-status-text');
    if (!dot || !text) return;

    try {
      await MXApi.get('/health');
      dot.className = 'mx-status-dot mx-status-dot--ok';
      text.textContent = 'Conectado';
    } catch {
      dot.className = 'mx-status-dot mx-status-dot--error';
      text.textContent = 'Desconectado';
    }
  }

  // -------------------------------------------------------------------------
  // Mobile drawer
  // -------------------------------------------------------------------------

  function initDrawer() {
    const toggle = document.getElementById('mx-drawer-toggle');
    const sidebar = document.getElementById('mx-sidebar');
    if (!toggle || !sidebar) return;

    toggle.addEventListener('click', () => {
      const isOpen = sidebar.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(isOpen));
    });
  }

  // -------------------------------------------------------------------------
  // Bootstrap
  // -------------------------------------------------------------------------

  function init() {
    initTabs();
    initDrawer();
    updateApiStatus();
    setInterval(updateApiStatus, MXConfig.pollIntervalMs);

    // Switch to the tab targeted by the URL hash (#dashboard, #training, ...).
    const hash = window.location.hash.replace('#', '');
    if (hash && TABS.find((t) => t.id === hash)) {
      switchTab(hash);
    }

    window.addEventListener('hashchange', () => {
      const h = window.location.hash.replace('#', '');
      if (h && TABS.find((t) => t.id === h)) switchTab(h);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
