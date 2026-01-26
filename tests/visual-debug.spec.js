// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('AudioSep Visual Debug', () => {
  test.setTimeout(300000);

  test('verify category heatmap renders', async ({ page }) => {
    page.on('console', msg => {
      console.log(`[BROWSER] ${msg.text()}`);
    });

    await page.goto('http://localhost:8000');
    await page.waitForSelector('#file-list', { timeout: 10000 });

    const files = await page.locator('.file-item').count();
    if (files === 0) { console.log('No files'); return; }

    await page.locator('.file-item').first().click();
    await page.waitForSelector('.classification-item', { timeout: 60000 });
    await page.locator('.separator-tab.audiosep').click();
    await page.waitForTimeout(1000);

    // Check for existing AudioSep results or trigger one
    const wsCount = await page.locator('.audio-visualization .wavesurfer-container').count();
    if (wsCount === 0) {
      const btn = page.locator('.classification-item .separate-btn').first();
      if (await btn.count() > 0) {
        console.log('Triggering AudioSep separation...');
        await btn.click();
        await page.waitForSelector('.audio-visualization .wavesurfer-container', { timeout: 120000 });
      }
    }

    // Wait for waveform and heatmap to render
    await page.waitForTimeout(3000);

    // Check if category heatmap exists
    const heatmapInfo = await page.evaluate(() => {
      const heatmaps = document.querySelectorAll('.category-heatmap-container');
      const results = [];

      heatmaps.forEach((hm, idx) => {
        const canvas = hm.querySelector('canvas');
        const category = hm.dataset.category;

        let pixels = 0;
        if (canvas) {
          try {
            const ctx = canvas.getContext('2d');
            const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (let i = 3; i < data.data.length; i += 4) if (data.data[i] > 0) pixels++;
          } catch (e) {}
        }

        results.push({
          index: idx,
          category,
          hasCanvas: !!canvas,
          canvasWidth: canvas?.width || 0,
          canvasHeight: canvas?.height || 0,
          pixels
        });
      });

      return results;
    });

    console.log('\n=== CATEGORY HEATMAP CHECK ===');
    console.log(`Found ${heatmapInfo.length} heatmap containers`);
    heatmapInfo.forEach(info => {
      console.log(`  [${info.index}] Category: "${info.category}"`);
      console.log(`      Canvas: ${info.hasCanvas ? `${info.canvasWidth}x${info.canvasHeight}` : 'MISSING'}`);
      console.log(`      Pixels: ${info.pixels}`);
    });

    // Check playhead
    const playheadInfo = await page.evaluate(() => {
      const playheads = document.querySelectorAll('.unified-playhead');
      return {
        count: playheads.length,
        visible: Array.from(playheads).filter(p => getComputedStyle(p).opacity !== '0').length
      };
    });
    console.log(`\nUnified playheads: ${playheadInfo.count} (${playheadInfo.visible} visible)`);

    // Take screenshot
    await page.screenshot({ path: 'tests/screenshots/category-heatmap.png', fullPage: true });
    console.log('\nScreenshot saved to tests/screenshots/category-heatmap.png');

    // Test playback with playhead sync
    const playBtn = page.locator('.audio-visualization .stem-play-btn').first();
    if (await playBtn.count() > 0) {
      console.log('\n=== TESTING PLAYHEAD SYNC ===');
      await playBtn.click();

      for (let i = 1; i <= 5; i++) {
        await page.waitForTimeout(500);
        const pos = await page.evaluate(() => {
          const playhead = document.querySelector('.unified-playhead');
          return playhead ? {
            left: playhead.style.left,
            opacity: getComputedStyle(playhead).opacity
          } : null;
        });
        console.log(`[T+${i * 0.5}s] Playhead: ${JSON.stringify(pos)}`);
      }

      await page.screenshot({ path: 'tests/screenshots/playhead-sync.png', fullPage: true });
      await playBtn.click(); // Pause
    }

    console.log('\n=== TEST COMPLETE ===');
    await page.waitForTimeout(3000);
  });
});
