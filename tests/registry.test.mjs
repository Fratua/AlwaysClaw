/**
 * registry.test.mjs
 *
 * Tests for loops/registry.json:
 * - Registry has exactly 105 entries
 * - Each entry has loopId, displayName, tier, category, riskTier, triggerClass
 * - No duplicate loopIds
 * - All contractFile paths exist (relative check)
 * - Dependency graph has no cycles (topological sort)
 * - Tier counts: A=43, B=47
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = process.cwd();

function loadJSON(relPath) {
  const abs = resolve(ROOT, relPath);
  return JSON.parse(readFileSync(abs, 'utf-8'));
}

function getLoopArray(registry) {
  return registry.loops || (Array.isArray(registry) ? registry : []);
}

describe('Loop Registry', () => {
  let registry;
  let loops;

  it('should be valid JSON', () => {
    registry = loadJSON('loops/registry.json');
    assert.ok(registry, 'Registry should parse as valid JSON');
  });

  it('should have exactly 105 entries', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    assert.equal(loops.length, 105, `Expected 105 entries, got ${loops.length}`);
  });

  it('should have loopId on every entry', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    for (let i = 0; i < loops.length; i++) {
      assert.ok(loops[i].loopId != null, `Entry ${i} missing loopId`);
      assert.ok(typeof loops[i].loopId === 'string', `Entry ${i} loopId should be a string`);
    }
  });

  it('should have displayName on every entry', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    for (let i = 0; i < loops.length; i++) {
      assert.ok(loops[i].displayName != null, `Entry ${i} missing displayName`);
    }
  });

  it('should have tier on every entry', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    for (let i = 0; i < loops.length; i++) {
      assert.ok(loops[i].tier != null, `Entry ${i} (${loops[i].loopId}) missing tier`);
    }
  });

  it('should have category on every entry', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    for (let i = 0; i < loops.length; i++) {
      assert.ok(loops[i].category != null, `Entry ${i} (${loops[i].loopId}) missing category`);
    }
  });

  it('should have riskTier on every entry', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    for (let i = 0; i < loops.length; i++) {
      assert.ok(loops[i].riskTier != null, `Entry ${i} (${loops[i].loopId}) missing riskTier`);
    }
  });

  it('should have triggerClass on every entry', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    for (let i = 0; i < loops.length; i++) {
      assert.ok(loops[i].triggerClass != null, `Entry ${i} (${loops[i].loopId}) missing triggerClass`);
    }
  });

  it('should have no duplicate loopIds', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    const ids = loops.map(l => l.loopId);
    const unique = new Set(ids);
    assert.equal(ids.length, unique.size,
      `Found ${ids.length - unique.size} duplicate loopIds`);
  });

  it('should have all contractFile paths pointing to existing files', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    const missing = [];
    for (const entry of loops) {
      if (entry.contractFile) {
        const fullPath = resolve(ROOT, entry.contractFile);
        if (!existsSync(fullPath)) {
          missing.push(entry.contractFile);
        }
      }
    }
    assert.equal(missing.length, 0,
      `Missing contract files: ${missing.slice(0, 10).join(', ')}`);
  });

  it('should have a dependency graph with no cycles (topological sort)', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);

    // Build adjacency list from dependency edges
    const adj = new Map();
    const idSet = new Set(loops.map(l => l.loopId));

    for (const entry of loops) {
      if (!adj.has(entry.loopId)) adj.set(entry.loopId, []);
      const deps = entry.dependencyEdges || entry.dependencies || [];
      for (const dep of deps) {
        const depId = typeof dep === 'string' ? dep : dep.target || dep.loopId;
        if (depId && idSet.has(depId)) {
          adj.get(entry.loopId).push(depId);
        }
      }
    }

    // Kahn's algorithm for cycle detection
    const inDegree = new Map();
    for (const id of idSet) inDegree.set(id, 0);
    for (const [, neighbors] of adj) {
      for (const n of neighbors) {
        inDegree.set(n, (inDegree.get(n) || 0) + 1);
      }
    }

    const queue = [];
    for (const [id, deg] of inDegree) {
      if (deg === 0) queue.push(id);
    }

    let visited = 0;
    while (queue.length > 0) {
      const node = queue.shift();
      visited++;
      for (const neighbor of (adj.get(node) || [])) {
        const newDeg = inDegree.get(neighbor) - 1;
        inDegree.set(neighbor, newDeg);
        if (newDeg === 0) queue.push(neighbor);
      }
    }

    assert.equal(visited, idSet.size,
      `Dependency graph has cycles (visited ${visited}/${idSet.size} nodes)`);
  });

  it('should have Tier A count of 43', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    const tierA = loops.filter(l => l.tier === 'A').length;
    assert.equal(tierA, 43, `Expected 43 Tier A loops, got ${tierA}`);
  });

  it('should have Tier B count of 47', () => {
    registry = loadJSON('loops/registry.json');
    loops = getLoopArray(registry);
    const tierB = loops.filter(l => l.tier === 'B').length;
    assert.equal(tierB, 47, `Expected 47 Tier B loops, got ${tierB}`);
  });
});
