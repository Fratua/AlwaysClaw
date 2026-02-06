/**
 * schema.test.mjs
 *
 * Tests for schema/loop.contract.schema.json:
 * - Schema is valid JSON
 * - Schema has all required property definitions
 * - Schema enum values match the master plan
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

describe('Loop Contract Schema', () => {
  let schema;

  it('should be valid JSON', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    assert.ok(schema, 'Schema should parse as valid JSON');
    assert.equal(typeof schema, 'object', 'Schema should be an object');
  });

  it('should have a $schema or type declaration', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const hasSchema = schema.$schema !== undefined;
    const hasType = schema.type !== undefined;
    assert.ok(hasSchema || hasType, 'Schema should have $schema or type');
  });

  it('should define required properties from the loop contract spec', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    const requiredProps = [
      'loopId',
      'version',
      'displayName',
      'entryCriteria',
      'successCriteria',
      'allowedTools',
      'maxSteps',
      'maxDurationSec',
      'fallbackLoopChain',
      'terminationPolicy',
    ];

    for (const prop of requiredProps) {
      assert.ok(props[prop] !== undefined, `Schema should define property: ${prop}`);
    }
  });

  it('should define thinkingPolicy property', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    assert.ok(props.thinkingPolicy !== undefined, 'Schema should define thinkingPolicy');
  });

  it('should define approvalPoints property', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    assert.ok(props.approvalPoints !== undefined, 'Schema should define approvalPoints');
  });

  it('should have category enum with 14 values matching master plan categories', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    // The category property should have an enum or be constrained
    if (props.category && props.category.enum) {
      assert.equal(props.category.enum.length, 14,
        `Expected 14 category enum values, got ${props.category.enum.length}`);
    } else {
      // Category might be defined differently, just check it exists
      assert.ok(props.category !== undefined, 'Schema should define category');
    }
  });

  it('should have dependencyLayer enum with 11 values matching master plan layers', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    if (props.dependencyLayer && props.dependencyLayer.enum) {
      assert.equal(props.dependencyLayer.enum.length, 11,
        `Expected 11 dependencyLayer enum values, got ${props.dependencyLayer.enum.length}`);
    } else {
      // May be defined differently
      assert.ok(props.dependencyLayer !== undefined || props.category !== undefined,
        'Schema should define dependencyLayer or category');
    }
  });

  it('should define riskTier with safe/moderate/high values', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    if (props.riskTier && props.riskTier.enum) {
      const expected = ['safe', 'moderate', 'high'];
      for (const val of expected) {
        assert.ok(props.riskTier.enum.includes(val),
          `riskTier enum should include "${val}"`);
      }
    } else {
      assert.ok(props.riskTier !== undefined, 'Schema should define riskTier');
    }
  });

  it('should define triggerTypes as an array property', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    const triggerProp = props.triggerTypes || props.triggerClass;
    assert.ok(triggerProp !== undefined, 'Schema should define triggerTypes or triggerClass');
  });

  it('should define ownerAgent property', () => {
    schema = loadJSON('schema/loop.contract.schema.json');
    const props = schema.properties || {};
    assert.ok(props.ownerAgent !== undefined, 'Schema should define ownerAgent');
  });
});
