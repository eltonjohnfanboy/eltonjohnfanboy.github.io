// Theme toggle — respects system preference, persists choice
(function() {
  const STORAGE_KEY = 'theme-preference';
  const root = document.documentElement;

  // Determine initial theme: stored preference > system preference > light
  function getPreferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    const btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.textContent = theme === 'dark' ? '☀ light' : '☾ dark';
    }
  }

  // Apply on load (before DOM ready, to avoid flash)
  applyTheme(getPreferredTheme());

  // Wire up toggle once DOM is ready
  document.addEventListener('DOMContentLoaded', function() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;

    // Set initial label
    applyTheme(getPreferredTheme());

    btn.addEventListener('click', function() {
      const current = root.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      localStorage.setItem(STORAGE_KEY, next);
      applyTheme(next);
    });
  });
})();
