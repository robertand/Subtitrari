import { test, expect } from '@playwright/test';

test('verify timeline interactions and localized UI', async ({ page }) => {
  // Go to the app
  await page.goto('http://localhost:5000');

  // Handle Login
  await page.fill('#loginUsername', 'TestUser');
  await page.click('button:has-text("Intră")');
  await expect(page.locator('#loginModal')).toBeHidden();

  // Verify localization and default settings
  const isolateLabel = page.locator('label', { hasText: 'Izolare Voce' });
  await expect(isolateLabel).toBeVisible();

  const isolateCheckbox = page.locator('#isolateVoice');
  await expect(isolateCheckbox).toBeChecked();

  const vadCheckbox = page.locator('#useVAD');
  await expect(vadCheckbox).toBeChecked();

  const dedupCheckbox = page.locator('#deduplicate');
  await expect(dedupCheckbox).toBeChecked();

  // Verify that WhisperX is gone
  const whisperXOption = page.locator('option[value="whisperx"]');
  await expect(whisperXOption).not.toBeAttached();

  // Mock a task completion to see the timeline
  // In a real scenario we'd upload a file, but here we can inject state via JS for UI verification
  await page.evaluate(() => {
    state.segments = [
      { start: 0, end: 5, text: 'Primul segment de test' },
      { start: 4, end: 9, text: 'Al doilea segment suprapus' }
    ];
    state.taskId = 'test-task';
    // Mock video duration
    Object.defineProperty(elements.mainVideoPlayer, 'duration', { value: 60 });
    showResults({ segments: state.segments, translations: {} });
  });

  // Check if timeline is visible
  const timeline = page.locator('#timelineSection');
  await expect(timeline).toBeVisible();

  // Check segments on timeline
  const timelineSegments = page.locator('.timeline-segment-block');
  await expect(timelineSegments).toHaveCount(2);

  // Check for playhead handle
  const playheadHandle = page.locator('#playheadHandle');
  await expect(playheadHandle).toBeVisible();

  // Take a screenshot of the populated timeline
  await page.screenshot({ path: '/home/jules/verification/timeline_populated.png' });

  // Test zoom interaction (trigger wheel event)
  await page.mouse.move(500, 200); // Over timeline
  await page.mouse.wheel(0, -100); // Zoom in

  // Wait a bit for render
  await page.waitForTimeout(500);
  const zoomText = await page.locator('#zoomLevel').textContent();
  console.log('Zoom Level after wheel:', zoomText);

  // Test deletion synchronization
  // Click delete on the first segment in the list
  await page.click('.segment-item[data-index="0"] .segment-delete-btn');

  // Handle confirmation dialog
  page.on('dialog', dialog => dialog.accept());

  // The first segment should be gone from both list and timeline
  // Wait for the change to reflect
  await page.waitForTimeout(500);
  await expect(page.locator('.segment-item')).toHaveCount(1);
  await expect(page.locator('.timeline-segment-block')).toHaveCount(1);

  await page.screenshot({ path: '/home/jules/verification/timeline_after_delete.png' });
});
