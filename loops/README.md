# AlwaysClaw Loop Registry and Contracts

## Overview

This directory contains the complete loop contract registry and individual contract files for all 105 hardcoded loops in the AlwaysClaw loop kernel.

## Structure

```
loops/
  registry.json              # Master registry with all 105 loop entries
  contracts/
    ralph-loop-v1.json       # Individual contract files (105 total)
    research-loop-v1.json
    ...
```

## Registry (`registry.json`)

The registry provides:

- **version**: Schema version of the registry
- **generatedAt**: ISO timestamp of generation
- **counts**: Total, core, derived, and tier distribution
- **dependencyLayers**: Loops grouped by dependency layer
- **loops**: Array of loop summary entries with loopId, displayName, tier, category, riskTier, triggerClass, dependsOn, blocks, and contractFile path

## Contracts (`contracts/*.json`)

Each contract file is a standalone JSON document validated against `../schema/loop.contract.schema.json`. It contains the full execution policy for one loop including:

- Entry and exit criteria
- Tool permissions and restrictions
- Thinking level policy
- Approval requirements
- Max steps and duration limits
- Success criteria and fallback chains
- Writeback and output contracts
- Composition rules (which loops can call/be called by this loop)

## Loop Categories

- **Core Loops (1-15)**: Ralph, Research, Discovery, Planning, Execution, Reflection, Meta-Cognition, Memory Consolidation, Context Refresh, Heartbeat Synthesis, Self-Driven, Approval Gate, Incident Response, Skill Acquisition, Persona Calibration
- **Runtime Loops (16-35)**: Queue, session, cron, heartbeat, routing, and orchestration
- **Context/Memory Loops (36-46)**: Memory write/search/compact, context compilation, persona loading
- **Integration Loops (47-65)**: Gmail, Twilio, Browser, Voice
- **Security Loops (66-75)**: Auth, channel gates, tool policy, secrets, incident detection
- **Reliability Loops (76-90)**: Health, SLO, error budget, chaos, failover, deployment
- **Quality Loops (91-95)**: Test runner, config drift, dependency scanning, code quality
- **Governance Loops (96-105)**: Proposals, approvals, goal rebalancing, policy updates

## Tier Distribution

- **Tier A** (43 loops): Mandatory for v1 OpenClaw parity and safety
- **Tier B** (47 loops): Advanced autonomy and optimization, phased in after v1 stability

## Regeneration

Run the generation script to regenerate all contracts:

```bash
node scripts/generate_artifacts.mjs
```
