/**
 * structure.test.mjs
 *
 * Tests that all expected directories and files exist in the artifact pack.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, statSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = process.cwd();

function dirExists(relPath) {
  const abs = resolve(ROOT, relPath);
  return existsSync(abs) && statSync(abs).isDirectory();
}

function fileExists(relPath) {
  return existsSync(resolve(ROOT, relPath));
}

describe('Directory Structure', () => {
  const requiredDirs = [
    'schema',
    'loops',
    'loops/contracts',
    'policies',
    'integrations',
    'ops',
  ];

  for (const dir of requiredDirs) {
    it(`directory "${dir}" should exist`, () => {
      assert.ok(dirExists(dir), `Directory ${dir} must exist`);
    });
  }
});

describe('Required Files', () => {
  const requiredFiles = [
    'schema/loop.contract.schema.json',
    'loops/registry.json',
    'policies/approval-matrix.json',
    'policies/tool-tier-matrix.json',
    'policies/session-scope-policy.json',
    'integrations/gmail-policy.json',
    'integrations/twilio-policy.json',
    'integrations/browser-policy.json',
    'ops/schedules.json',
    'ops/slo-targets.json',
    'ops/incident-runbook.md',
  ];

  for (const file of requiredFiles) {
    it(`file "${file}" should exist`, () => {
      assert.ok(fileExists(file), `File ${file} must exist`);
    });
  }
});

describe('Contract Files Count', () => {
  it('loops/contracts/ should contain exactly 105 JSON files', () => {
    const contractDir = resolve(ROOT, 'loops', 'contracts');
    assert.ok(existsSync(contractDir), 'loops/contracts/ must exist');
    const jsonFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    assert.equal(jsonFiles.length, 105,
      `Expected 105 contract files, got ${jsonFiles.length}`);
  });
});

describe('Schema File', () => {
  it('schema/loop.contract.schema.json should be non-empty valid JSON', () => {
    const abs = resolve(ROOT, 'schema', 'loop.contract.schema.json');
    assert.ok(existsSync(abs), 'Schema file must exist');
    const { readFileSync } = await import('node:fs');
    const raw = readFileSync(abs, 'utf-8');
    assert.ok(raw.trim().length > 0, 'Schema file should not be empty');
    const parsed = JSON.parse(raw);
    assert.ok(typeof parsed === 'object', 'Schema should parse as object');
  });
});

describe('Registry File', () => {
  it('loops/registry.json should be non-empty valid JSON', () => {
    const abs = resolve(ROOT, 'loops', 'registry.json');
    assert.ok(existsSync(abs), 'Registry file must exist');
    const { readFileSync } = await import('node:fs');
    const raw = readFileSync(abs, 'utf-8');
    assert.ok(raw.trim().length > 0, 'Registry file should not be empty');
    const parsed = JSON.parse(raw);
    assert.ok(typeof parsed === 'object', 'Registry should parse as object');
  });
});

describe('Ops Files', () => {
  it('ops/schedules.json should be valid JSON', () => {
    const abs = resolve(ROOT, 'ops', 'schedules.json');
    assert.ok(existsSync(abs), 'schedules.json must exist');
    const { readFileSync } = await import('node:fs');
    const data = JSON.parse(readFileSync(abs, 'utf-8'));
    assert.ok(typeof data === 'object', 'schedules.json should parse as object');
  });

  it('ops/slo-targets.json should be valid JSON', () => {
    const abs = resolve(ROOT, 'ops', 'slo-targets.json');
    assert.ok(existsSync(abs), 'slo-targets.json must exist');
    const { readFileSync } = await import('node:fs');
    const data = JSON.parse(readFileSync(abs, 'utf-8'));
    assert.ok(typeof data === 'object', 'slo-targets.json should parse as object');
  });

  it('ops/incident-runbook.md should exist and be non-empty', () => {
    const abs = resolve(ROOT, 'ops', 'incident-runbook.md');
    assert.ok(existsSync(abs), 'incident-runbook.md must exist');
    const { readFileSync } = await import('node:fs');
    const raw = readFileSync(abs, 'utf-8');
    assert.ok(raw.trim().length > 0, 'incident-runbook.md should not be empty');
  });
});
