/**
 * contracts.test.mjs
 *
 * Tests for loops/contracts/*.json:
 * - Each of 105 contracts validates against schema (using ajv)
 * - No two contracts have identical entryCriteria
 * - No two contracts have identical successCriteria
 * - All fallbackLoopChain references point to valid loopIds
 * - Core loops have richer fields than derived
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';

const ROOT = process.cwd();

function loadJSON(relPath) {
  const abs = resolve(ROOT, relPath);
  return JSON.parse(readFileSync(abs, 'utf-8'));
}

function getLoopArray(registry) {
  return registry.loops || (Array.isArray(registry) ? registry : []);
}

describe('Loop Contracts', () => {
  let contractFiles;
  let contracts;
  let schemaValidate;
  let registryIds;

  it('should have exactly 105 contract files', () => {
    const contractDir = resolve(ROOT, 'loops', 'contracts');
    assert.ok(existsSync(contractDir), 'loops/contracts/ directory must exist');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    assert.equal(contractFiles.length, 105,
      `Expected 105 contract files, got ${contractFiles.length}`);
  });

  it('should all be valid JSON and parseable', () => {
    const contractDir = resolve(ROOT, 'loops', 'contracts');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    contracts = [];
    const failures = [];
    for (const file of contractFiles) {
      try {
        const data = JSON.parse(readFileSync(join(contractDir, file), 'utf-8'));
        contracts.push({ file, data });
      } catch (e) {
        failures.push(`${file}: ${e.message}`);
      }
    }
    assert.equal(failures.length, 0,
      `JSON parse failures: ${failures.join('; ')}`);
  });

  it('should all validate against the loop contract schema', () => {
    const schema = loadJSON('schema/loop.contract.schema.json');
    const ajv = new Ajv({ strict: false, allErrors: true });
    addFormats(ajv);
    schemaValidate = ajv.compile(schema);

    const contractDir = resolve(ROOT, 'loops', 'contracts');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    contracts = contractFiles.map(file => ({
      file,
      data: JSON.parse(readFileSync(join(contractDir, file), 'utf-8')),
    }));

    const failures = [];
    for (const { file, data } of contracts) {
      const valid = schemaValidate(data);
      if (!valid) {
        const errs = schemaValidate.errors.map(e => `${e.instancePath} ${e.message}`).join(', ');
        failures.push(`${file}: ${errs}`);
      }
    }
    assert.equal(failures.length, 0,
      `Schema validation failures (first 5): ${failures.slice(0, 5).join('; ')}`);
  });

  it('should have no two contracts with identical entryCriteria', () => {
    const contractDir = resolve(ROOT, 'loops', 'contracts');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    contracts = contractFiles.map(file => ({
      file,
      data: JSON.parse(readFileSync(join(contractDir, file), 'utf-8')),
    }));

    const seen = new Map();
    const duplicates = [];
    for (const { file, data } of contracts) {
      const key = JSON.stringify(data.entryCriteria || []);
      if (seen.has(key)) {
        duplicates.push(`${file} duplicates ${seen.get(key)}`);
      } else {
        seen.set(key, file);
      }
    }
    assert.equal(duplicates.length, 0,
      `Duplicate entryCriteria: ${duplicates.slice(0, 5).join('; ')}`);
  });

  it('should have no two contracts with identical successCriteria', () => {
    const contractDir = resolve(ROOT, 'loops', 'contracts');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    contracts = contractFiles.map(file => ({
      file,
      data: JSON.parse(readFileSync(join(contractDir, file), 'utf-8')),
    }));

    const seen = new Map();
    const duplicates = [];
    for (const { file, data } of contracts) {
      const key = JSON.stringify(data.successCriteria || []);
      if (seen.has(key)) {
        duplicates.push(`${file} duplicates ${seen.get(key)}`);
      } else {
        seen.set(key, file);
      }
    }
    assert.equal(duplicates.length, 0,
      `Duplicate successCriteria: ${duplicates.slice(0, 5).join('; ')}`);
  });

  it('should have all fallbackLoopChain references point to valid loopIds', () => {
    const registry = loadJSON('loops/registry.json');
    const loops = getLoopArray(registry);
    registryIds = new Set(loops.map(l => l.loopId));

    const contractDir = resolve(ROOT, 'loops', 'contracts');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    contracts = contractFiles.map(file => ({
      file,
      data: JSON.parse(readFileSync(join(contractDir, file), 'utf-8')),
    }));

    const invalid = [];
    for (const { file, data } of contracts) {
      const chain = data.fallbackLoopChain || [];
      for (const ref of chain) {
        if (!registryIds.has(ref)) {
          invalid.push(`${file}: references unknown loopId "${ref}"`);
        }
      }
    }
    assert.equal(invalid.length, 0,
      `Invalid fallbackLoopChain refs: ${invalid.slice(0, 5).join('; ')}`);
  });

  it('should have core loops with richer field sets than derived loops', () => {
    const registry = loadJSON('loops/registry.json');
    const loops = getLoopArray(registry);
    const coreIds = new Set(
      loops.filter(l => l.category === 'core' || l.loopClass === 'core').map(l => l.loopId)
    );
    const derivedIds = new Set(
      loops.filter(l => l.category === 'derived' || l.loopClass === 'derived').map(l => l.loopId)
    );

    const contractDir = resolve(ROOT, 'loops', 'contracts');
    contractFiles = readdirSync(contractDir).filter(f => f.endsWith('.json'));
    contracts = contractFiles.map(file => ({
      file,
      data: JSON.parse(readFileSync(join(contractDir, file), 'utf-8')),
    }));

    let coreFieldCounts = [];
    let derivedFieldCounts = [];

    for (const { data } of contracts) {
      const fieldCount = Object.keys(data).length;
      if (coreIds.has(data.loopId)) {
        coreFieldCounts.push(fieldCount);
      } else if (derivedIds.has(data.loopId)) {
        derivedFieldCounts.push(fieldCount);
      }
    }

    if (coreFieldCounts.length > 0 && derivedFieldCounts.length > 0) {
      const avgCore = coreFieldCounts.reduce((a, b) => a + b, 0) / coreFieldCounts.length;
      const avgDerived = derivedFieldCounts.reduce((a, b) => a + b, 0) / derivedFieldCounts.length;
      assert.ok(avgCore >= avgDerived,
        `Core loops avg fields (${avgCore.toFixed(1)}) should be >= derived (${avgDerived.toFixed(1)})`);
    }
  });
});
