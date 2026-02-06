# AlwaysClaw Validation Test Suite

Test suite for validating the AlwaysClaw artifact pack (schema, loop contracts, registry, policies, integrations, and directory structure).

## Prerequisites

- Node.js >= 18 (uses `node:test` built-in test runner)
- Install dependencies in both `scripts/` and `tests/` directories

```bash
cd scripts && npm install
cd ../tests && npm install
```

## Running Tests

All tests must be run from the **project root** directory (where `schema/`, `loops/`, `policies/`, etc. live).

### Run all tests

```bash
node --test tests/*.test.mjs
```

### Run individual test suites

```bash
node --test tests/schema.test.mjs
node --test tests/registry.test.mjs
node --test tests/contracts.test.mjs
node --test tests/policies.test.mjs
node --test tests/integrations.test.mjs
node --test tests/structure.test.mjs
```

### Run via npm scripts (from project root)

```bash
npm --prefix tests test
npm --prefix tests run test:schema
npm --prefix tests run test:registry
npm --prefix tests run test:contracts
npm --prefix tests run test:policies
npm --prefix tests run test:integrations
npm --prefix tests run test:structure
```

## Running the Production Validator

The `scripts/validate_artifacts.mjs` validator performs a comprehensive check and outputs a structured report.

```bash
node scripts/validate_artifacts.mjs
```

- Exit code `0` = all checks passed
- Exit code `1` = one or more checks failed

The validator outputs both a human-readable report and a JSON summary.

## Test Coverage

| File | What It Tests |
|---|---|
| `schema.test.mjs` | Schema is valid JSON, has all required property definitions, enum values match master plan |
| `registry.test.mjs` | 105 entries, required fields, no duplicates, contract file refs, no dependency cycles, tier counts |
| `contracts.test.mjs` | All 105 validate against schema, unique entry/success criteria, valid fallback refs, core vs derived richness |
| `policies.test.mjs` | Approval matrix covers 105 loops, 3 tool tiers, 4 session strategies, no cross-tier tool overlap |
| `integrations.test.mjs` | Gmail/Twilio/Browser policy files have all required sections |
| `structure.test.mjs` | All expected directories and files exist in the artifact pack |

## Dependencies

- `ajv` - JSON Schema validator (used by contracts.test.mjs and validate_artifacts.mjs)
- `ajv-formats` - Format validation plugin for ajv
- `node:test` - Built-in Node.js test runner (no external test framework needed)
- `node:assert` - Built-in Node.js assertion library
