#!/usr/bin/env node

/**
 * validate_artifacts.mjs
 *
 * Production validation script for the AlwaysClaw artifact pack.
 * Loads the loop contract schema, validates all contracts, the registry,
 * policies, integrations, and directory structure.
 *
 * Exit code 0 on all-pass, 1 on any failure.
 */

import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ROOT = process.cwd();

function loadJSON(relPath) {
  const abs = resolve(ROOT, relPath);
  const raw = readFileSync(abs, 'utf-8');
  return JSON.parse(raw);
}

function fileExists(relPath) {
  return existsSync(resolve(ROOT, relPath));
}

function dirExists(relPath) {
  const abs = resolve(ROOT, relPath);
  return existsSync(abs) && statSync(abs).isDirectory();
}

// ---------------------------------------------------------------------------
// Report accumulator
// ---------------------------------------------------------------------------

const report = {
  sections: [],
  totalChecks: 0,
  passed: 0,
  failed: 0,
  errors: [],
};

function section(name) {
  const s = { name, checks: [] };
  report.sections.push(s);
  return s;
}

function check(sec, description, passed, detail) {
  report.totalChecks++;
  if (passed) {
    report.passed++;
  } else {
    report.failed++;
    report.errors.push(`[${sec.name}] ${description}: ${detail || 'FAILED'}`);
  }
  sec.checks.push({ description, passed, detail: detail || '' });
}

// ---------------------------------------------------------------------------
// 1. Schema validation setup
// ---------------------------------------------------------------------------

function validateSchema(sec) {
  let schema;
  try {
    schema = loadJSON('schema/loop.contract.schema.json');
    check(sec, 'Schema file is valid JSON', true);
  } catch (e) {
    check(sec, 'Schema file is valid JSON', false, e.message);
    return null;
  }

  const ajv = new Ajv({ strict: false, allErrors: true });
  addFormats(ajv);

  let validate;
  try {
    validate = ajv.compile(schema);
    check(sec, 'Schema compiles with ajv', true);
  } catch (e) {
    check(sec, 'Schema compiles with ajv', false, e.message);
    return null;
  }

  return validate;
}

// ---------------------------------------------------------------------------
// 2. Registry validation
// ---------------------------------------------------------------------------

function validateRegistry(sec) {
  let registry;
  try {
    registry = loadJSON('loops/registry.json');
    check(sec, 'Registry is valid JSON', true);
  } catch (e) {
    check(sec, 'Registry is valid JSON', false, e.message);
    return null;
  }

  const loops = registry.loops || registry;
  const loopArray = Array.isArray(loops) ? loops : [];

  check(sec, 'Registry total count is 105', loopArray.length === 105,
    `got ${loopArray.length}`);

  const coreCount = loopArray.filter(l => l.category === 'core' || l.loopClass === 'core').length;
  const derivedCount = loopArray.filter(l => l.category === 'derived' || l.loopClass === 'derived').length;
  check(sec, 'Core loop count is 15', coreCount === 15, `got ${coreCount}`);
  check(sec, 'Derived loop count is 90', derivedCount === 90, `got ${derivedCount}`);

  const tierA = loopArray.filter(l => l.tier === 'A').length;
  const tierB = loopArray.filter(l => l.tier === 'B').length;
  check(sec, 'Tier A count is 43', tierA === 43, `got ${tierA}`);
  check(sec, 'Tier B count is 47', tierB === 47, `got ${tierB}`);

  // Check required fields on each entry
  const requiredFields = ['loopId', 'displayName', 'tier', 'category', 'riskTier', 'triggerClass'];
  let missingFieldEntries = [];
  for (const entry of loopArray) {
    for (const field of requiredFields) {
      if (entry[field] === undefined || entry[field] === null) {
        missingFieldEntries.push(`${entry.loopId || 'unknown'} missing ${field}`);
      }
    }
  }
  check(sec, 'All registry entries have required fields',
    missingFieldEntries.length === 0,
    missingFieldEntries.length > 0 ? missingFieldEntries.slice(0, 5).join('; ') : '');

  // Check no duplicate loopIds
  const ids = loopArray.map(l => l.loopId);
  const uniqueIds = new Set(ids);
  check(sec, 'No duplicate loopIds', ids.length === uniqueIds.size,
    `${ids.length - uniqueIds.size} duplicates found`);

  // Check contract file references
  let missingContracts = [];
  for (const entry of loopArray) {
    if (entry.contractFile) {
      if (!fileExists(entry.contractFile)) {
        missingContracts.push(entry.contractFile);
      }
    }
  }
  check(sec, 'All contractFile references exist',
    missingContracts.length === 0,
    missingContracts.length > 0 ? `missing: ${missingContracts.slice(0, 5).join(', ')}` : '');

  // Validate dependency edges reference valid loop IDs
  let invalidDeps = [];
  for (const entry of loopArray) {
    const deps = entry.dependencyEdges || entry.dependencies || [];
    for (const dep of deps) {
      const depId = typeof dep === 'string' ? dep : dep.target || dep.loopId;
      if (depId && !uniqueIds.has(depId)) {
        invalidDeps.push(`${entry.loopId} -> ${depId}`);
      }
    }
  }
  check(sec, 'All dependency edges reference valid loop IDs',
    invalidDeps.length === 0,
    invalidDeps.length > 0 ? invalidDeps.slice(0, 5).join('; ') : '');

  return { loopArray, idSet: uniqueIds };
}

// ---------------------------------------------------------------------------
// 3. Contract validation
// ---------------------------------------------------------------------------

function validateContracts(sec, schemaValidate, registryData) {
  const contractDir = resolve(ROOT, 'loops', 'contracts');
  if (!existsSync(contractDir)) {
    check(sec, 'loops/contracts/ directory exists', false);
    return;
  }

  const files = readdirSync(contractDir).filter(f => f.endsWith('.json'));
  check(sec, 'Contract file count is 105', files.length === 105,
    `got ${files.length}`);

  let schemaFailures = [];
  let contracts = [];

  for (const file of files) {
    const filePath = join(contractDir, file);
    try {
      const contract = JSON.parse(readFileSync(filePath, 'utf-8'));
      contracts.push({ file, contract });
      if (schemaValidate) {
        const valid = schemaValidate(contract);
        if (!valid) {
          schemaFailures.push(`${file}: ${schemaValidate.errors.map(e => e.message).join(', ')}`);
        }
      }
    } catch (e) {
      schemaFailures.push(`${file}: ${e.message}`);
    }
  }

  check(sec, 'All contracts validate against schema',
    schemaFailures.length === 0,
    schemaFailures.length > 0 ? schemaFailures.slice(0, 5).join('; ') : '');

  // Check no duplicate entryCriteria
  const entryStrs = contracts.map(c => JSON.stringify(c.contract.entryCriteria || []));
  const uniqueEntries = new Set(entryStrs);
  check(sec, 'No two contracts have identical entryCriteria',
    entryStrs.length === uniqueEntries.size,
    `${entryStrs.length - uniqueEntries.size} duplicates`);

  // Check no duplicate successCriteria
  const successStrs = contracts.map(c => JSON.stringify(c.contract.successCriteria || []));
  const uniqueSuccess = new Set(successStrs);
  check(sec, 'No two contracts have identical successCriteria',
    successStrs.length === uniqueSuccess.size,
    `${successStrs.length - uniqueSuccess.size} duplicates`);

  // Check fallbackLoopChain references
  if (registryData) {
    let invalidFallbacks = [];
    for (const { file, contract } of contracts) {
      const chain = contract.fallbackLoopChain || [];
      for (const ref of chain) {
        if (!registryData.idSet.has(ref)) {
          invalidFallbacks.push(`${file}: ${ref}`);
        }
      }
    }
    check(sec, 'All fallbackLoopChain references point to valid loopIds',
      invalidFallbacks.length === 0,
      invalidFallbacks.length > 0 ? invalidFallbacks.slice(0, 5).join('; ') : '');
  }
}

// ---------------------------------------------------------------------------
// 4. Policy validation
// ---------------------------------------------------------------------------

function validatePolicies(sec) {
  // approval-matrix.json
  try {
    const approval = loadJSON('policies/approval-matrix.json');
    const entries = approval.loops || approval.entries || approval;
    const entryArray = Array.isArray(entries) ? entries : Object.keys(entries);
    check(sec, 'approval-matrix.json has entries for all 105 loops',
      entryArray.length >= 105, `got ${entryArray.length}`);
    const topKeys = Object.keys(approval);
    check(sec, 'approval-matrix.json has required top-level keys',
      topKeys.length > 0, '');
  } catch (e) {
    check(sec, 'approval-matrix.json is valid JSON', false, e.message);
  }

  // tool-tier-matrix.json
  try {
    const toolTiers = loadJSON('policies/tool-tier-matrix.json');
    const tiers = toolTiers.tiers || toolTiers;
    const tierKeys = Array.isArray(tiers) ? tiers : Object.keys(tiers);
    check(sec, 'tool-tier-matrix.json has exactly 3 tiers',
      tierKeys.length === 3, `got ${tierKeys.length}`);

    // Check no tool appears in multiple tiers
    if (!Array.isArray(tiers)) {
      const allToolSets = Object.values(tiers).map(t => {
        const tools = t.tools || t.allowedTools || t;
        return Array.isArray(tools) ? tools : [];
      });
      const seen = new Set();
      let duplicates = [];
      for (const toolSet of allToolSets) {
        for (const tool of toolSet) {
          if (seen.has(tool)) duplicates.push(tool);
          seen.add(tool);
        }
      }
      check(sec, 'No tool appears in multiple tiers',
        duplicates.length === 0,
        duplicates.length > 0 ? duplicates.slice(0, 5).join(', ') : '');
    }
  } catch (e) {
    check(sec, 'tool-tier-matrix.json is valid JSON', false, e.message);
  }

  // session-scope-policy.json
  try {
    const sessionPolicy = loadJSON('policies/session-scope-policy.json');
    const strategies = sessionPolicy.strategies || sessionPolicy.sessionStrategies || sessionPolicy;
    const strategyKeys = Array.isArray(strategies) ? strategies : Object.keys(strategies);
    const requiredStrategies = ['main', 'per-peer', 'per-channel-peer', 'per-account-channel-peer'];
    const missing = requiredStrategies.filter(s => !strategyKeys.includes(s));
    check(sec, 'session-scope-policy.json has all 4 strategies',
      missing.length === 0, missing.length > 0 ? `missing: ${missing.join(', ')}` : '');
  } catch (e) {
    check(sec, 'session-scope-policy.json is valid JSON', false, e.message);
  }
}

// ---------------------------------------------------------------------------
// 5. Integration validation
// ---------------------------------------------------------------------------

function validateIntegrations(sec) {
  // gmail-policy.json
  try {
    const gmail = loadJSON('integrations/gmail-policy.json');
    const requiredKeys = ['watchLifecycle', 'notificationProcessing', 'security', 'reliability'];
    const topKeys = Object.keys(gmail);
    const missing = requiredKeys.filter(k => !topKeys.includes(k));
    check(sec, 'gmail-policy.json has required sections',
      missing.length === 0, missing.length > 0 ? `missing: ${missing.join(', ')}` : '');
  } catch (e) {
    check(sec, 'gmail-policy.json is valid JSON', false, e.message);
  }

  // twilio-policy.json
  try {
    const twilio = loadJSON('integrations/twilio-policy.json');
    const requiredKeys = ['webhookSecurity', 'smsOperations', 'voiceOperations'];
    const topKeys = Object.keys(twilio);
    const missing = requiredKeys.filter(k => !topKeys.includes(k));
    check(sec, 'twilio-policy.json has required sections',
      missing.length === 0, missing.length > 0 ? `missing: ${missing.join(', ')}` : '');
  } catch (e) {
    check(sec, 'twilio-policy.json is valid JSON', false, e.message);
  }

  // browser-policy.json
  try {
    const browser = loadJSON('integrations/browser-policy.json');
    const requiredKeys = ['profileIsolation', 'operationGating', 'artifactVerification'];
    const topKeys = Object.keys(browser);
    const missing = requiredKeys.filter(k => !topKeys.includes(k));
    check(sec, 'browser-policy.json has required sections',
      missing.length === 0, missing.length > 0 ? `missing: ${missing.join(', ')}` : '');
  } catch (e) {
    check(sec, 'browser-policy.json is valid JSON', false, e.message);
  }
}

// ---------------------------------------------------------------------------
// 6. Directory structure validation
// ---------------------------------------------------------------------------

function validateStructure(sec) {
  const requiredDirs = [
    'schema',
    'loops',
    'loops/contracts',
    'policies',
    'integrations',
    'ops',
  ];

  for (const dir of requiredDirs) {
    check(sec, `Directory exists: ${dir}`, dirExists(dir));
  }

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
    check(sec, `File exists: ${file}`, fileExists(file));
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function printReport() {
  console.log('\n========================================');
  console.log('  AlwaysClaw Artifact Validation Report');
  console.log('========================================\n');

  for (const s of report.sections) {
    const passed = s.checks.filter(c => c.passed).length;
    const failed = s.checks.filter(c => !c.passed).length;
    const icon = failed === 0 ? 'PASS' : 'FAIL';
    console.log(`[${icon}] ${s.name} (${passed}/${s.checks.length} passed)`);
    for (const c of s.checks) {
      const mark = c.passed ? '  + ' : '  - ';
      const suffix = !c.passed && c.detail ? ` (${c.detail})` : '';
      console.log(`${mark}${c.description}${suffix}`);
    }
    console.log('');
  }

  console.log('----------------------------------------');
  console.log(`Total: ${report.totalChecks} checks, ${report.passed} passed, ${report.failed} failed`);
  console.log('----------------------------------------\n');

  if (report.failed > 0) {
    console.log('FAILURES:');
    for (const err of report.errors) {
      console.log(`  - ${err}`);
    }
    console.log('');
  }

  // Output structured JSON summary
  console.log('--- JSON SUMMARY ---');
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    totalChecks: report.totalChecks,
    passed: report.passed,
    failed: report.failed,
    result: report.failed === 0 ? 'PASS' : 'FAIL',
    sections: report.sections.map(s => ({
      name: s.name,
      passed: s.checks.filter(c => c.passed).length,
      failed: s.checks.filter(c => !c.passed).length,
    })),
    errors: report.errors,
  }, null, 2));
}

function main() {
  // 1. Schema
  const schemaSec = section('Schema Validation');
  const schemaValidate = validateSchema(schemaSec);

  // 2. Registry
  const registrySec = section('Registry Validation');
  const registryData = validateRegistry(registrySec);

  // 3. Contracts
  const contractsSec = section('Contract Validation');
  validateContracts(contractsSec, schemaValidate, registryData);

  // 4. Policies
  const policiesSec = section('Policy Validation');
  validatePolicies(policiesSec);

  // 5. Integrations
  const integrationsSec = section('Integration Validation');
  validateIntegrations(integrationsSec);

  // 6. Structure
  const structureSec = section('Directory Structure');
  validateStructure(structureSec);

  // Report
  printReport();

  process.exit(report.failed === 0 ? 0 : 1);
}

main();
