const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const errors = [];
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    if (text.includes('localhost:8765') || text.includes('Training monitor')) return;
    if (type === 'error' || type === 'warning') errors.push(`[${type}] ${text}`);
  });
  page.on('pageerror', err => errors.push(`[pageerror] ${err.message}`));

  try {
    await page.goto('http://127.0.0.1:8000/', { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('#mx-nav', { timeout: 10000 });
    await page.waitForTimeout(2000);

    const champion = await page.textContent('#tournament-champion-name');
    console.log('Champion rendered:', champion);
    const finalVisible = await page.isVisible('#tournament-final');
    console.log('Final visible:', finalVisible);

    if (errors.length > 0) {
      console.log('Console issues:', errors.join('\n'));
      process.exit(1);
    }
    process.exit(champion && champion !== '-' ? 0 : 1);
  } catch (e) {
    console.error('Test failed:', e.message);
    console.error('Console issues:', errors.join('\n'));
    process.exit(1);
  } finally {
    await browser.close();
  }
})();
