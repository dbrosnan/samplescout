// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('AudioSep Visual Debug', () => {
  test.setTimeout(300000);

  test('debug waveform rendering during playback', async ({ page }) => {
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

    const wsCount = await page.locator('.audio-visualization .wavesurfer-container').count();
    if (wsCount === 0) {
      const btn = page.locator('.classification-item .separate-btn').first();
      if (await btn.count() > 0) {
        await btn.click();
        await page.waitForSelector('.audio-visualization .wavesurfer-container', { timeout: 120000 });
      }
    }

    // Wait for waveform to fully render
    await page.waitForTimeout(3000);

    // Function to count pixels in shadow DOM canvas
    const countPixels = async () => {
      return await page.evaluate(() => {
        const container = document.querySelector('.audio-visualization .wavesurfer-container');
        if (!container) return { error: 'no container' };

        // Find the div that holds the shadow DOM (first non-playhead child)
        let shadowHost = null;
        for (const child of container.children) {
          if (child.shadowRoot) {
            shadowHost = child;
            break;
          }
        }

        if (!shadowHost || !shadowHost.shadowRoot) {
          return { error: 'no shadow root', children: container.children.length };
        }

        const canvas = shadowHost.shadowRoot.querySelector('canvas');
        if (!canvas) return { error: 'no canvas in shadow DOM' };

        try {
          const ctx = canvas.getContext('2d');
          const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
          let pixels = 0;
          for (let i = 3; i < data.data.length; i += 4) if (data.data[i] > 0) pixels++;
          return { pixels, width: canvas.width, height: canvas.height };
        } catch (e) {
          return { error: e.message };
        }
      });
    };

    console.log('\n=== PRE-PLAYBACK STATE ===');
    const prePlaying = await countPixels();
    console.log('Pixels before play:', JSON.stringify(prePlaying));

    // Take screenshot before playback
    await page.screenshot({ path: 'tests/screenshots/before-play.png', fullPage: true });

    // Find and click play button (stem players use stem-play-btn class)
    const playBtn = page.locator('.audio-visualization .stem-play-btn').first();
    if (await playBtn.count() > 0) {
      console.log('\n=== STARTING PLAYBACK ===');
      await playBtn.click();

      // Check pixels during playback
      for (let i = 1; i <= 5; i++) {
        await page.waitForTimeout(500);
        const during = await countPixels();
        console.log(`[T+${i * 0.5}s] Pixels: ${JSON.stringify(during)}`);
      }

      // Take screenshot during playback
      await page.screenshot({ path: 'tests/screenshots/during-play.png', fullPage: true });

      // Pause
      await playBtn.click();
      await page.waitForTimeout(500);

      console.log('\n=== POST-PAUSE STATE ===');
      const postPause = await countPixels();
      console.log('Pixels after pause:', JSON.stringify(postPause));

      // Take screenshot after pause
      await page.screenshot({ path: 'tests/screenshots/after-pause.png', fullPage: true });

      // Final verdict
      console.log('\n=== VISUAL INTEGRITY CHECK ===');
      if (prePlaying.pixels && postPause.pixels) {
        const ratio = postPause.pixels / prePlaying.pixels;
        console.log(`Pre-play pixels: ${prePlaying.pixels}`);
        console.log(`Post-pause pixels: ${postPause.pixels}`);
        console.log(`Ratio: ${ratio.toFixed(2)}`);
        if (ratio >= 0.9) {
          console.log('✓ PASS: Visuals preserved during playback');
        } else {
          console.log('✗ FAIL: Visuals degraded during playback');
        }
      }
    }

    console.log('\n=== TEST COMPLETE ===');
    await page.waitForTimeout(5000);
  });
});
