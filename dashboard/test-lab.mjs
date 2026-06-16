import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
const page = await context.newPage();

const errors = [];
page.on('console', msg => {
  const type = msg.type();
  if (type === 'error' || type === 'warning') {
    errors.push(`[${type}] ${msg.text()}`);
  }
});
page.on('pageerror', err => errors.push(`[pageerror] ${err.message}`));

try {
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForSelector('#mx-nav', { timeout: 10000 });

  // Click Lab tab
  const labTab = await page.locator('button[data-tab="lab"]');
  await labTab.click();
  await page.waitForTimeout(2000);

  // Click train XGBoost (don't wait for completion)
  const trainBtn = await page.locator('#lab-train-xgboost');
  if (await trainBtn.isVisible()) {
    // Do not actually train to save time; just verify it exists.
    console.log('Lab tab rendered and train button visible.');
  }

  console.log('Console issues:', errors.length > 0 ? errors.join('\n') : 'none');
  process.exit(errors.length > 0 ? 1 : 0);
} catch (e) {
  console.error('Test failed:', e.message);
  console.error('Console issues:', errors.join('\n'));
  process.exit(1);
} finally {
  await browser.close();
}
