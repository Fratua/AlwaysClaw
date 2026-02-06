/**
 * integrations.test.mjs
 *
 * Tests for integration policy JSON files:
 * - gmail-policy.json has watchLifecycle, notificationProcessing, security, reliability
 * - twilio-policy.json has webhookSecurity, smsOperations, voiceOperations
 * - browser-policy.json has profileIsolation, operationGating, artifactVerification
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = process.cwd();

function loadJSON(relPath) {
  const abs = resolve(ROOT, relPath);
  return JSON.parse(readFileSync(abs, 'utf-8'));
}

describe('Gmail Policy', () => {
  it('should be valid JSON', () => {
    const data = loadJSON('integrations/gmail-policy.json');
    assert.ok(data, 'gmail-policy.json should parse');
    assert.equal(typeof data, 'object');
  });

  it('should have watchLifecycle section', () => {
    const data = loadJSON('integrations/gmail-policy.json');
    assert.ok(data.watchLifecycle !== undefined,
      'gmail-policy.json must have watchLifecycle');
  });

  it('should have notificationProcessing section', () => {
    const data = loadJSON('integrations/gmail-policy.json');
    assert.ok(data.notificationProcessing !== undefined,
      'gmail-policy.json must have notificationProcessing');
  });

  it('should have security section', () => {
    const data = loadJSON('integrations/gmail-policy.json');
    assert.ok(data.security !== undefined,
      'gmail-policy.json must have security');
  });

  it('should have reliability section', () => {
    const data = loadJSON('integrations/gmail-policy.json');
    assert.ok(data.reliability !== undefined,
      'gmail-policy.json must have reliability');
  });

  it('watchLifecycle should define renewal cadence', () => {
    const data = loadJSON('integrations/gmail-policy.json');
    const wl = data.watchLifecycle;
    if (typeof wl === 'object') {
      const hasRenewal = wl.renewalCadence !== undefined ||
        wl.maxWatchLifetimeDays !== undefined ||
        wl.renewalPolicy !== undefined;
      assert.ok(hasRenewal, 'watchLifecycle should define renewal cadence');
    }
  });
});

describe('Twilio Policy', () => {
  it('should be valid JSON', () => {
    const data = loadJSON('integrations/twilio-policy.json');
    assert.ok(data, 'twilio-policy.json should parse');
    assert.equal(typeof data, 'object');
  });

  it('should have webhookSecurity section', () => {
    const data = loadJSON('integrations/twilio-policy.json');
    assert.ok(data.webhookSecurity !== undefined,
      'twilio-policy.json must have webhookSecurity');
  });

  it('should have smsOperations section', () => {
    const data = loadJSON('integrations/twilio-policy.json');
    assert.ok(data.smsOperations !== undefined,
      'twilio-policy.json must have smsOperations');
  });

  it('should have voiceOperations section', () => {
    const data = loadJSON('integrations/twilio-policy.json');
    assert.ok(data.voiceOperations !== undefined,
      'twilio-policy.json must have voiceOperations');
  });

  it('webhookSecurity should reference X-Twilio-Signature validation', () => {
    const data = loadJSON('integrations/twilio-policy.json');
    const ws = data.webhookSecurity;
    if (typeof ws === 'object') {
      const json = JSON.stringify(ws).toLowerCase();
      const hasSignature = json.includes('signature') || json.includes('x-twilio');
      assert.ok(hasSignature, 'webhookSecurity should reference signature validation');
    }
  });
});

describe('Browser Policy', () => {
  it('should be valid JSON', () => {
    const data = loadJSON('integrations/browser-policy.json');
    assert.ok(data, 'browser-policy.json should parse');
    assert.equal(typeof data, 'object');
  });

  it('should have profileIsolation section', () => {
    const data = loadJSON('integrations/browser-policy.json');
    assert.ok(data.profileIsolation !== undefined,
      'browser-policy.json must have profileIsolation');
  });

  it('should have operationGating section', () => {
    const data = loadJSON('integrations/browser-policy.json');
    assert.ok(data.operationGating !== undefined,
      'browser-policy.json must have operationGating');
  });

  it('should have artifactVerification section', () => {
    const data = loadJSON('integrations/browser-policy.json');
    assert.ok(data.artifactVerification !== undefined,
      'browser-policy.json must have artifactVerification');
  });

  it('profileIsolation should mandate per-agent profiles', () => {
    const data = loadJSON('integrations/browser-policy.json');
    const pi = data.profileIsolation;
    if (typeof pi === 'object') {
      const json = JSON.stringify(pi).toLowerCase();
      const hasIsolation = json.includes('isolat') || json.includes('per-agent') || json.includes('managed');
      assert.ok(hasIsolation,
        'profileIsolation should reference isolation or per-agent profiles');
    }
  });
});
