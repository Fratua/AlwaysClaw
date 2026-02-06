/**
 * policies.test.mjs
 *
 * Tests for policy JSON files:
 * - approval-matrix.json has entries for all 105 loops
 * - tool-tier-matrix.json has exactly 3 tiers
 * - session-scope-policy.json has all 4 strategies
 * - No tool appears in multiple tiers
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

describe('Approval Matrix Policy', () => {
  it('should be valid JSON', () => {
    const data = loadJSON('policies/approval-matrix.json');
    assert.ok(data, 'approval-matrix.json should parse');
  });

  it('should have entries for all 105 loops', () => {
    const data = loadJSON('policies/approval-matrix.json');
    // The matrix may store entries as an array or as an object keyed by loopId
    const entries = data.loops || data.entries || data;
    let count;
    if (Array.isArray(entries)) {
      count = entries.length;
    } else if (typeof entries === 'object') {
      count = Object.keys(entries).length;
    } else {
      count = 0;
    }
    assert.ok(count >= 105,
      `Expected at least 105 loop entries in approval matrix, got ${count}`);
  });

  it('should have required top-level structure', () => {
    const data = loadJSON('policies/approval-matrix.json');
    assert.ok(typeof data === 'object', 'Should be an object');
    const keys = Object.keys(data);
    assert.ok(keys.length > 0, 'Should have at least one top-level key');
  });
});

describe('Tool Tier Matrix Policy', () => {
  it('should be valid JSON', () => {
    const data = loadJSON('policies/tool-tier-matrix.json');
    assert.ok(data, 'tool-tier-matrix.json should parse');
  });

  it('should have exactly 3 tiers', () => {
    const data = loadJSON('policies/tool-tier-matrix.json');
    const tiers = data.tiers || data;
    let tierKeys;
    if (Array.isArray(tiers)) {
      tierKeys = tiers;
    } else if (typeof tiers === 'object') {
      tierKeys = Object.keys(tiers);
    } else {
      tierKeys = [];
    }
    assert.equal(tierKeys.length, 3,
      `Expected exactly 3 tiers, got ${tierKeys.length}`);
  });

  it('should have no tool appearing in multiple tiers', () => {
    const data = loadJSON('policies/tool-tier-matrix.json');
    const tiers = data.tiers || data;

    if (typeof tiers === 'object' && !Array.isArray(tiers)) {
      const seenTools = new Map();
      const duplicates = [];

      for (const [tierName, tierDef] of Object.entries(tiers)) {
        const tools = tierDef.tools || tierDef.allowedTools || (Array.isArray(tierDef) ? tierDef : []);
        for (const tool of tools) {
          if (seenTools.has(tool)) {
            duplicates.push(`"${tool}" in both ${seenTools.get(tool)} and ${tierName}`);
          } else {
            seenTools.set(tool, tierName);
          }
        }
      }

      assert.equal(duplicates.length, 0,
        `Tools in multiple tiers: ${duplicates.slice(0, 5).join('; ')}`);
    }
  });
});

describe('Session Scope Policy', () => {
  it('should be valid JSON', () => {
    const data = loadJSON('policies/session-scope-policy.json');
    assert.ok(data, 'session-scope-policy.json should parse');
  });

  it('should have all 4 session strategies', () => {
    const data = loadJSON('policies/session-scope-policy.json');
    const strategies = data.strategies || data.sessionStrategies || data;
    let strategyKeys;
    if (Array.isArray(strategies)) {
      strategyKeys = strategies;
    } else if (typeof strategies === 'object') {
      strategyKeys = Object.keys(strategies);
    } else {
      strategyKeys = [];
    }

    const required = ['main', 'per-peer', 'per-channel-peer', 'per-account-channel-peer'];
    const missing = required.filter(s => !strategyKeys.includes(s));
    assert.equal(missing.length, 0,
      `Missing strategies: ${missing.join(', ')}`);
  });

  it('should define a default strategy', () => {
    const data = loadJSON('policies/session-scope-policy.json');
    // The policy should indicate a default strategy
    const hasDefault = data.default !== undefined ||
      data.defaultStrategy !== undefined ||
      (data.strategies && typeof data.strategies === 'object' &&
       Object.values(data.strategies).some(s => s.default === true || s.isDefault === true));
    assert.ok(hasDefault, 'Should define a default strategy');
  });
});
