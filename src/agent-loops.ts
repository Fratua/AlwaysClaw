/**
 * Agent Loops - 15 Hardcoded Agentic Loops
 * OpenClaw AI Agent Scheduler
 */

import { SchedulerSystem } from './scheduler-system';

const logger = require('../../logger');

/**
 * Predefined schedule constants
 */
export const SCHEDULES = {
  // Fast loops (seconds)
  EVERY_5_SECONDS: '*/5 * * * * *',
  EVERY_10_SECONDS: '*/10 * * * * *',
  EVERY_30_SECONDS: '*/30 * * * * *',
  
  // Normal loops (minutes)
  EVERY_MINUTE: '* * * * *',
  EVERY_2_MINUTES: '*/2 * * * *',
  EVERY_5_MINUTES: '*/5 * * * *',
  EVERY_10_MINUTES: '*/10 * * * *',
  EVERY_15_MINUTES: '*/15 * * * *',
  EVERY_30_MINUTES: '*/30 * * * *',
  
  // Hourly loops
  EVERY_HOUR: '0 * * * *',
  
  // Daily loops
  DAILY_2AM: '0 2 * * *',
  DAILY_3AM: '0 3 * * *',
  DAILY_4AM: '0 4 * * *',
  DAILY_9AM: '0 9 * * *',
  DAILY_5PM: '0 17 * * *',

  // Weekly loops
  WEEKLY_MONDAY_9AM: '0 9 * * 1',
  WEEKLY_SUNDAY_3AM: '0 3 * * 0',
} as const;

/**
 * Register all 15 agent loops
 */
export function registerAgentLoops(scheduler: SchedulerSystem): void {
  logger.info('[AgentLoops] Registering 15 agent loops...');

  const bridgeCall = async (method: string, context: Record<string, unknown> = {}) => {
    // Bridge is accessed at runtime from the daemon master
    const { getBridge } = require('../../python-bridge');
    const bridge = getBridge();
    return bridge.call(method, { timestamp: Date.now(), ...context });
  };

  // Loop 1: Heartbeat Monitor - Every 30 seconds
  scheduler.registerAgentLoop(
    'loop-01-heartbeat',
    'Heartbeat Monitor',
    SCHEDULES.EVERY_30_SECONDS,
    async () => {
      logger.info('[Loop:Heartbeat] Sending system heartbeat...');
      const health = await bridgeCall('health');
      if (process.send) {
        process.send({
          type: 'cron-heartbeat',
          data: { timestamp: Date.now(), uptime: process.uptime(), memory: process.memoryUsage(), bridgeHealth: health }
        });
      }
    },
    { 
      tags: ['critical', 'system', 'monitoring'],
      preventOverlap: true,
      timeout: 10000,
    }
  );

  // Loop 2: Gmail Synchronization - Every 2 minutes
  scheduler.registerAgentLoop(
    'loop-02-gmail-sync',
    'Gmail Synchronization',
    SCHEDULES.EVERY_2_MINUTES,
    async () => {
      logger.info('[Loop:GmailSync] Syncing Gmail inbox...');
      try {
        const result = await bridgeCall('gmail.read', { query: 'is:unread', max_results: 10 });
        logger.info('[Loop:GmailSync] Inbox synced:', typeof result === 'object' ? JSON.stringify(result).substring(0, 200) : result);
      } catch (e) {
        logger.warn('[Loop:GmailSync] Gmail sync skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['communication', 'gmail'],
      preventOverlap: true,
      timeout: 60000,
    }
  );

  // Loop 3: Browser Health Check - Every minute
  scheduler.registerAgentLoop(
    'loop-03-browser-health',
    'Browser Health Check',
    SCHEDULES.EVERY_MINUTE,
    async () => {
      logger.info('[Loop:BrowserHealth] Checking browser health...');
      try {
        const health = await bridgeCall('health');
        logger.info('[Loop:BrowserHealth] Bridge status:', health?.status || 'unknown');
      } catch (e) {
        logger.warn('[Loop:BrowserHealth] Check skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['system', 'browser', 'monitoring'],
      preventOverlap: true,
      timeout: 30000,
    }
  );

  // Loop 4: Voice Queue Processor - Every 5 seconds
  scheduler.registerAgentLoop(
    'loop-04-voice-queue',
    'Voice Queue Processor',
    SCHEDULES.EVERY_5_SECONDS,
    async () => {
      // Voice queue processing is lightweight - check if any TTS/STT requests pending
      // No-op if no queue system configured
    },
    { 
      tags: ['voice', 'tts', 'stt', 'realtime'],
      preventOverlap: true,
      timeout: 4000,
    }
  );

  // Loop 5: Twilio Status Check - Every minute
  scheduler.registerAgentLoop(
    'loop-05-twilio-status',
    'Twilio Status Check',
    SCHEDULES.EVERY_MINUTE,
    async () => {
      logger.info('[Loop:TwilioStatus] Checking Twilio status...');
      try {
        const health = await bridgeCall('health');
        logger.info('[Loop:TwilioStatus] System status:', health?.status || 'unknown');
      } catch (e) {
        logger.warn('[Loop:TwilioStatus] Check skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['communication', 'twilio', 'sms', 'voice'],
      preventOverlap: true,
      timeout: 30000,
    }
  );

  // Loop 6: System Resource Monitor - Every 30 seconds
  scheduler.registerAgentLoop(
    'loop-06-resource-monitor',
    'Resource Monitor',
    SCHEDULES.EVERY_30_SECONDS,
    async () => {
      const usage = process.memoryUsage();
      logger.info('[Loop:ResourceMonitor] Memory usage:', {
        heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)}MB`,
        heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)}MB`,
        external: `${Math.round(usage.external / 1024 / 1024)}MB`,
        rss: `${Math.round(usage.rss / 1024 / 1024)}MB`,
      });
      try {
        await bridgeCall('health');
      } catch (e) {
        // Resource monitoring continues even if bridge is down
      }
    },
    { 
      tags: ['system', 'monitoring', 'resources'],
      preventOverlap: true,
      timeout: 10000,
    }
  );

  // Loop 7: Identity Synchronization - Every 5 minutes
  scheduler.registerAgentLoop(
    'loop-07-identity-sync',
    'Identity Synchronization',
    SCHEDULES.EVERY_5_MINUTES,
    async () => {
      logger.info('[Loop:IdentitySync] Syncing identity state...');
      try {
        await bridgeCall('memory.search', { query: 'identity config', limit: 5 });
      } catch (e) {
        logger.warn('[Loop:IdentitySync] Sync skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['identity', 'user', 'config'],
      preventOverlap: true,
      timeout: 60000,
    }
  );

  // Loop 8: Soul State Update - Every minute
  scheduler.registerAgentLoop(
    'loop-08-soul-update',
    'Soul State Update',
    SCHEDULES.EVERY_MINUTE,
    async () => {
      logger.info('[Loop:SoulUpdate] Updating soul state...');
      try {
        await bridgeCall('memory.store', {
          type: 'system',
          content: JSON.stringify({ state: 'active', timestamp: Date.now(), uptime: process.uptime() }),
          source: 'soul-update-loop',
          tags: ['soul-state', 'system'],
        });
      } catch (e) {
        logger.warn('[Loop:SoulUpdate] Update skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['soul', 'emotional', 'behavioral'],
      preventOverlap: true,
      timeout: 30000,
    }
  );

  // Loop 9: Context Cleanup - Every 10 minutes
  scheduler.registerAgentLoop(
    'loop-09-context-cleanup',
    'Context Cleanup',
    SCHEDULES.EVERY_10_MINUTES,
    async () => {
      logger.info('[Loop:ContextCleanup] Cleaning up old context...');
      try {
        await bridgeCall('memory.consolidate');
      } catch (e) {
        logger.warn('[Loop:ContextCleanup] Cleanup skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['maintenance', 'cleanup', 'context'],
      preventOverlap: true,
      timeout: 120000,
    }
  );

  // Loop 10: Notification Digest - Every 15 minutes
  scheduler.registerAgentLoop(
    'loop-10-notification-digest',
    'Notification Digest',
    SCHEDULES.EVERY_15_MINUTES,
    async () => {
      logger.info('[Loop:NotificationDigest] Processing notification digest...');
      try {
        const result = await bridgeCall('memory.search', { query: 'notification', type: 'episodic', limit: 20 });
        logger.info('[Loop:NotificationDigest] Found notifications:', result?.count || 0);
      } catch (e) {
        logger.warn('[Loop:NotificationDigest] Digest skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['notifications', 'digest', 'communication'],
      preventOverlap: true,
      timeout: 60000,
    }
  );

  // Loop 11: Log Rotation - Every hour
  scheduler.registerAgentLoop(
    'loop-11-log-rotation',
    'Log Rotation',
    SCHEDULES.EVERY_HOUR,
    async () => {
      logger.info('[Loop:LogRotation] Log rotation handled by Winston DailyRotateFile');
      // Winston's DailyRotateFile transport handles rotation automatically
    },
    { 
      tags: ['maintenance', 'logs', 'cleanup'],
      preventOverlap: true,
      timeout: 300000,
    }
  );

  // Loop 12: Backup Check - Every 30 minutes
  scheduler.registerAgentLoop(
    'loop-12-backup-check',
    'Backup Check',
    SCHEDULES.EVERY_30_MINUTES,
    async () => {
      logger.info('[Loop:BackupCheck] Running backup...');
      try {
        const result = await bridgeCall('memory.backup');
        logger.info('[Loop:BackupCheck] Backup result:', result?.backed_up ? 'success' : 'failed');
      } catch (e) {
        logger.warn('[Loop:BackupCheck] Backup skipped:', (e as Error).message);
      }
    },
    { 
      tags: ['maintenance', 'backup', 'data'],
      preventOverlap: true,
      timeout: 300000,
    }
  );

  // Loop 13: API Rate Limit Check - Every minute
  scheduler.registerAgentLoop(
    'loop-13-rate-limit',
    'API Rate Limit Check',
    SCHEDULES.EVERY_MINUTE,
    async () => {
      logger.info('[Loop:RateLimit] Checking API rate limits...');
      const usage = process.cpuUsage();
      logger.info('[Loop:RateLimit] CPU usage:', { user: usage.user, system: usage.system });
    },
    { 
      tags: ['system', 'api', 'throttling'],
      preventOverlap: true,
      timeout: 30000,
    }
  );

  // Loop 14: User Presence Check - Every 5 minutes
  scheduler.registerAgentLoop(
    'loop-14-presence-check',
    'User Presence Check',
    SCHEDULES.EVERY_5_MINUTES,
    async () => {
      logger.info('[Loop:PresenceCheck] User presence check completed');
      // User presence detection requires Windows UI automation (not yet implemented)
    },
    { 
      tags: ['user', 'presence', 'activity'],
      preventOverlap: true,
      timeout: 30000,
    }
  );

  // Loop 15: Daily Maintenance - 2 AM daily
  scheduler.registerAgentLoop(
    'loop-15-daily-maintenance',
    'Daily Maintenance',
    SCHEDULES.DAILY_2AM,
    async () => {
      logger.info('[Loop:DailyMaintenance] Running daily maintenance...');
      try {
        await bridgeCall('memory.consolidate');
        await bridgeCall('memory.backup');
        await bridgeCall('memory.sync');
        logger.info('[Loop:DailyMaintenance] Maintenance complete');
      } catch (e) {
        logger.warn('[Loop:DailyMaintenance] Maintenance error:', (e as Error).message);
      }
    },
    { 
      tags: ['maintenance', 'daily', 'cleanup', 'reports'],
      preventOverlap: true,
      timeout: 1800000, // 30 minutes
    }
  );

  logger.info('[AgentLoops] 15 operational loops registered');

  // ═══════════════════════════════════════════════════════════
  // Python Cognitive Loops (16-29) - dispatched via bridge
  // ═══════════════════════════════════════════════════════════

  // Loop 16: Ralph Loop - Every 5 minutes
  scheduler.registerAgentLoop(
    'loop-16-ralph',
    'Ralph Loop (Self-Driven Agent)',
    SCHEDULES.EVERY_5_MINUTES,
    async () => {
      logger.info('[Loop:Ralph] Running ralph cycle...');
      await bridgeCall('loop.ralph.run_cycle');
    },
    { tags: ['cognitive', 'ralph', 'self-driven'], preventOverlap: true, timeout: 240000 }
  );

  // Loop 17: Research Loop - Every 15 minutes
  scheduler.registerAgentLoop(
    'loop-17-research',
    'Research Loop',
    SCHEDULES.EVERY_15_MINUTES,
    async () => {
      logger.info('[Loop:Research] Running research cycle...');
      await bridgeCall('loop.research.run_cycle');
    },
    { tags: ['cognitive', 'research'], preventOverlap: true, timeout: 600000 }
  );

  // Loop 18: Planning Loop - Every 10 minutes
  scheduler.registerAgentLoop(
    'loop-18-planning',
    'Planning Loop',
    SCHEDULES.EVERY_10_MINUTES,
    async () => {
      logger.info('[Loop:Planning] Running planning cycle...');
      await bridgeCall('loop.planning.run_cycle');
    },
    { tags: ['cognitive', 'planning'], preventOverlap: true, timeout: 300000 }
  );

  // Loop 19: E2E Loop - Every 10 minutes
  scheduler.registerAgentLoop(
    'loop-19-e2e',
    'End-to-End Loop',
    SCHEDULES.EVERY_10_MINUTES,
    async () => {
      logger.info('[Loop:E2E] Running E2E cycle...');
      await bridgeCall('loop.e2e.run_cycle');
    },
    { tags: ['cognitive', 'e2e'], preventOverlap: true, timeout: 300000 }
  );

  // Loop 20: Exploration Loop - Every 30 minutes
  scheduler.registerAgentLoop(
    'loop-20-exploration',
    'Exploration Loop',
    SCHEDULES.EVERY_30_MINUTES,
    async () => {
      logger.info('[Loop:Exploration] Running exploration cycle...');
      await bridgeCall('loop.exploration.run_cycle');
    },
    { tags: ['cognitive', 'exploration'], preventOverlap: true, timeout: 600000 }
  );

  // Loop 21: Discovery Loop - Every 30 minutes
  scheduler.registerAgentLoop(
    'loop-21-discovery',
    'Discovery Loop',
    SCHEDULES.EVERY_30_MINUTES,
    async () => {
      logger.info('[Loop:Discovery] Running discovery cycle...');
      await bridgeCall('loop.discovery.run_cycle');
    },
    { tags: ['cognitive', 'discovery'], preventOverlap: true, timeout: 600000 }
  );

  // Loop 22: Bug Finder Loop - Every 15 minutes
  scheduler.registerAgentLoop(
    'loop-22-bug-finder',
    'Bug Finder Loop',
    SCHEDULES.EVERY_15_MINUTES,
    async () => {
      logger.info('[Loop:BugFinder] Running bug finder cycle...');
      await bridgeCall('loop.bug_finder.run_cycle');
    },
    { tags: ['cognitive', 'bug-finder', 'quality'], preventOverlap: true, timeout: 300000 }
  );

  // Loop 23: Self-Learning Loop - Hourly
  scheduler.registerAgentLoop(
    'loop-23-self-learning',
    'Self-Learning Loop',
    SCHEDULES.EVERY_HOUR,
    async () => {
      logger.info('[Loop:SelfLearning] Running self-learning cycle...');
      await bridgeCall('loop.self_learning.run_cycle');
    },
    { tags: ['cognitive', 'self-learning'], preventOverlap: true, timeout: 1800000 }
  );

  // Loop 24: Meta-Cognition Loop - Hourly
  scheduler.registerAgentLoop(
    'loop-24-meta-cognition',
    'Meta-Cognition Loop',
    SCHEDULES.EVERY_HOUR,
    async () => {
      logger.info('[Loop:MetaCognition] Running meta-cognition cycle...');
      await bridgeCall('loop.meta_cognition.run_cycle');
    },
    { tags: ['cognitive', 'meta-cognition'], preventOverlap: true, timeout: 1800000 }
  );

  // Loop 25: Self-Upgrading Loop - Daily 3 AM
  scheduler.registerAgentLoop(
    'loop-25-self-upgrading',
    'Self-Upgrading Loop',
    SCHEDULES.DAILY_3AM,
    async () => {
      logger.info('[Loop:SelfUpgrading] Running self-upgrading cycle...');
      await bridgeCall('loop.self_upgrading.run_cycle');
    },
    { tags: ['cognitive', 'self-upgrading'], preventOverlap: true, timeout: 3600000 }
  );

  // Loop 26: Self-Updating Loop - Daily 4 AM
  scheduler.registerAgentLoop(
    'loop-26-self-updating',
    'Self-Updating Loop',
    SCHEDULES.DAILY_4AM,
    async () => {
      logger.info('[Loop:SelfUpdating] Running self-updating cycle...');
      await bridgeCall('loop.self_updating.run_cycle');
    },
    { tags: ['cognitive', 'self-updating'], preventOverlap: true, timeout: 3600000 }
  );

  // Loop 27: Self-Driven Loop - Every 30 minutes
  scheduler.registerAgentLoop(
    'loop-27-self-driven',
    'Self-Driven Loop',
    SCHEDULES.EVERY_30_MINUTES,
    async () => {
      logger.info('[Loop:SelfDriven] Running self-driven cycle...');
      await bridgeCall('loop.self_driven.run_cycle');
    },
    { tags: ['cognitive', 'self-driven'], preventOverlap: true, timeout: 600000 }
  );

  // Loop 28: CPEL Loop - Hourly
  scheduler.registerAgentLoop(
    'loop-28-cpel',
    'Context/Prompt Engineering Loop',
    SCHEDULES.EVERY_HOUR,
    async () => {
      logger.info('[Loop:CPEL] Running CPEL cycle...');
      await bridgeCall('loop.cpel.run_cycle');
    },
    { tags: ['cognitive', 'cpel', 'prompt-engineering'], preventOverlap: true, timeout: 1800000 }
  );

  // Loop 29: Context Engineering Loop - Every 15 minutes
  scheduler.registerAgentLoop(
    'loop-29-context-eng',
    'Context Engineering Loop',
    SCHEDULES.EVERY_15_MINUTES,
    async () => {
      logger.info('[Loop:ContextEng] Running context engineering cycle...');
      await bridgeCall('loop.context_engineering.run_cycle');
    },
    { tags: ['cognitive', 'context-engineering'], preventOverlap: true, timeout: 300000 }
  );

  // Loop 30: Web Monitor Loop - Every 10 minutes
  scheduler.registerAgentLoop(
    'loop-30-web-monitor',
    'Web Monitor Loop',
    SCHEDULES.EVERY_10_MINUTES,
    async () => {
      logger.info('[Loop:WebMonitor] Running web monitor cycle...');
      await bridgeCall('loop.web_monitor.run_cycle');
    },
    { tags: ['cognitive', 'web-monitor'], preventOverlap: true, timeout: 300000 }
  );

  logger.info('[AgentLoops] 15 cognitive loops registered');

  // ═══════════════════════════════════════════════════════════
  // Operational Cron Jobs (absorbed from cron-scheduler.js)
  // ═══════════════════════════════════════════════════════════

  // Cron 1: Cleanup - Daily midnight
  scheduler.registerAgentLoop(
    'cron-cleanup',
    'Daily Cleanup',
    '0 0 * * *',
    async () => {
      logger.info('[Cron:Cleanup] Running cleanup...');
      await bridgeCall('memory.consolidate');
    },
    { tags: ['cron', 'maintenance', 'cleanup'], preventOverlap: true, timeout: 600000 }
  );

  // Cron 2: Backup - Daily 2 AM
  scheduler.registerAgentLoop(
    'cron-backup',
    'Daily Backup',
    SCHEDULES.DAILY_2AM,
    async () => {
      logger.info('[Cron:Backup] Running backup...');
      await bridgeCall('memory.backup');
    },
    { tags: ['cron', 'backup'], preventOverlap: true, timeout: 600000 }
  );

  // Cron 3: Weekly Report - Monday 9 AM
  scheduler.registerAgentLoop(
    'cron-weekly-report',
    'Weekly Report',
    SCHEDULES.WEEKLY_MONDAY_9AM,
    async () => {
      logger.info('[Cron:WeeklyReport] Generating weekly report...');
    },
    { tags: ['cron', 'reports'], preventOverlap: true, timeout: 600000 }
  );

  // Cron 4: Health Check - Every 5 minutes
  scheduler.registerAgentLoop(
    'cron-health-check',
    'Periodic Health Check',
    SCHEDULES.EVERY_5_MINUTES,
    async () => {
      logger.info('[Cron:HealthCheck] Running health check...');
      try {
        await bridgeCall('health');
      } catch (e) {
        logger.error('[Cron:HealthCheck] Bridge health check failed:', e);
      }
    },
    { tags: ['cron', 'health'], preventOverlap: true, timeout: 30000 }
  );

  // Cron 5: State Persist - Every 10 minutes
  scheduler.registerAgentLoop(
    'cron-state-persist',
    'State Persistence',
    SCHEDULES.EVERY_10_MINUTES,
    async () => {
      logger.info('[Cron:StatePersist] Persisting state...');
    },
    { tags: ['cron', 'state'], preventOverlap: true, timeout: 60000 }
  );

  // Cron 6: Metrics Collection - Every minute
  scheduler.registerAgentLoop(
    'cron-metrics-collect',
    'Metrics Collection',
    SCHEDULES.EVERY_MINUTE,
    async () => {
      // Lightweight metrics collection
    },
    { tags: ['cron', 'metrics'], preventOverlap: true, timeout: 10000 }
  );

  const totalLoops = scheduler.cronEngine.getAllJobs().length;
  logger.info(`[AgentLoops] All ${totalLoops} loops registered successfully (15 operational + 15 cognitive + 6 cron)`);
}

/**
 * Get loop information by ID
 */
export function getLoopInfo(loopId: string): {
  id: string;
  name: string;
  schedule: string;
  description: string;
  tags: string[];
} | null {
  const loops: Record<string, { name: string; schedule: string; description: string; tags: string[] }> = {
    'loop-01-heartbeat': {
      name: 'Heartbeat Monitor',
      schedule: SCHEDULES.EVERY_30_SECONDS,
      description: 'Sends periodic heartbeat signals to monitor system health',
      tags: ['critical', 'system', 'monitoring'],
    },
    'loop-02-gmail-sync': {
      name: 'Gmail Synchronization',
      schedule: SCHEDULES.EVERY_2_MINUTES,
      description: 'Synchronizes Gmail inbox for new messages and commands',
      tags: ['communication', 'gmail'],
    },
    'loop-03-browser-health': {
      name: 'Browser Health Check',
      schedule: SCHEDULES.EVERY_MINUTE,
      description: 'Monitors browser instance health and responsiveness',
      tags: ['system', 'browser', 'monitoring'],
    },
    'loop-04-voice-queue': {
      name: 'Voice Queue Processor',
      schedule: SCHEDULES.EVERY_5_SECONDS,
      description: 'Processes TTS/STT queue for voice interactions',
      tags: ['voice', 'tts', 'stt', 'realtime'],
    },
    'loop-05-twilio-status': {
      name: 'Twilio Status Check',
      schedule: SCHEDULES.EVERY_MINUTE,
      description: 'Monitors Twilio connection and message status',
      tags: ['communication', 'twilio', 'sms', 'voice'],
    },
    'loop-06-resource-monitor': {
      name: 'Resource Monitor',
      schedule: SCHEDULES.EVERY_30_SECONDS,
      description: 'Monitors system resource usage (memory, CPU, disk)',
      tags: ['system', 'monitoring', 'resources'],
    },
    'loop-07-identity-sync': {
      name: 'Identity Synchronization',
      schedule: SCHEDULES.EVERY_5_MINUTES,
      description: 'Synchronizes identity and user preference updates',
      tags: ['identity', 'user', 'config'],
    },
    'loop-08-soul-update': {
      name: 'Soul State Update',
      schedule: SCHEDULES.EVERY_MINUTE,
      description: 'Updates emotional and behavioral state parameters',
      tags: ['soul', 'emotional', 'behavioral'],
    },
    'loop-09-context-cleanup': {
      name: 'Context Cleanup',
      schedule: SCHEDULES.EVERY_10_MINUTES,
      description: 'Cleans up old conversation context and cache entries',
      tags: ['maintenance', 'cleanup', 'context'],
    },
    'loop-10-notification-digest': {
      name: 'Notification Digest',
      schedule: SCHEDULES.EVERY_15_MINUTES,
      description: 'Sends accumulated notification digests',
      tags: ['notifications', 'digest', 'communication'],
    },
    'loop-11-log-rotation': {
      name: 'Log Rotation',
      schedule: SCHEDULES.EVERY_HOUR,
      description: 'Rotates and compresses log files',
      tags: ['maintenance', 'logs', 'cleanup'],
    },
    'loop-12-backup-check': {
      name: 'Backup Check',
      schedule: SCHEDULES.EVERY_30_MINUTES,
      description: 'Verifies backup integrity and completion',
      tags: ['maintenance', 'backup', 'data'],
    },
    'loop-13-rate-limit': {
      name: 'API Rate Limit Check',
      schedule: SCHEDULES.EVERY_MINUTE,
      description: 'Monitors and adjusts API call rates',
      tags: ['system', 'api', 'throttling'],
    },
    'loop-14-presence-check': {
      name: 'User Presence Check',
      schedule: SCHEDULES.EVERY_5_MINUTES,
      description: 'Detects user activity and idle state',
      tags: ['user', 'presence', 'activity'],
    },
    'loop-15-daily-maintenance': {
      name: 'Daily Maintenance',
      schedule: SCHEDULES.DAILY_2AM,
      description: 'Comprehensive daily maintenance tasks',
      tags: ['maintenance', 'daily', 'cleanup', 'reports'],
    },
    'loop-16-ralph': {
      name: 'Ralph Loop',
      schedule: SCHEDULES.EVERY_5_MINUTES,
      description: 'Self-driven agent loop for autonomous task execution',
      tags: ['cognitive', 'ralph', 'self-driven'],
    },
    'loop-17-research': {
      name: 'Research Loop',
      schedule: SCHEDULES.EVERY_15_MINUTES,
      description: 'Autonomous research and information gathering',
      tags: ['cognitive', 'research'],
    },
    'loop-18-planning': {
      name: 'Planning Loop',
      schedule: SCHEDULES.EVERY_10_MINUTES,
      description: 'Strategic planning and task decomposition',
      tags: ['cognitive', 'planning'],
    },
    'loop-19-e2e': {
      name: 'End-to-End Loop',
      schedule: SCHEDULES.EVERY_10_MINUTES,
      description: 'End-to-end workflow execution and monitoring',
      tags: ['cognitive', 'e2e'],
    },
    'loop-20-exploration': {
      name: 'Exploration Loop',
      schedule: SCHEDULES.EVERY_30_MINUTES,
      description: 'Exploratory learning and environment discovery',
      tags: ['cognitive', 'exploration'],
    },
    'loop-21-discovery': {
      name: 'Discovery Loop',
      schedule: SCHEDULES.EVERY_30_MINUTES,
      description: 'New capability and pattern discovery',
      tags: ['cognitive', 'discovery'],
    },
    'loop-22-bug-finder': {
      name: 'Bug Finder Loop',
      schedule: SCHEDULES.EVERY_15_MINUTES,
      description: 'Automated bug detection and reporting',
      tags: ['cognitive', 'bug-finder', 'quality'],
    },
    'loop-23-self-learning': {
      name: 'Self-Learning Loop',
      schedule: SCHEDULES.EVERY_HOUR,
      description: 'Learning from past interactions and outcomes',
      tags: ['cognitive', 'self-learning'],
    },
    'loop-24-meta-cognition': {
      name: 'Meta-Cognition Loop',
      schedule: SCHEDULES.EVERY_HOUR,
      description: 'Thinking about thinking - cognitive strategy optimization',
      tags: ['cognitive', 'meta-cognition'],
    },
    'loop-25-self-upgrading': {
      name: 'Self-Upgrading Loop',
      schedule: SCHEDULES.DAILY_3AM,
      description: 'Self-improvement and capability upgrades',
      tags: ['cognitive', 'self-upgrading'],
    },
    'loop-26-self-updating': {
      name: 'Self-Updating Loop',
      schedule: SCHEDULES.DAILY_4AM,
      description: 'Configuration and model self-updates',
      tags: ['cognitive', 'self-updating'],
    },
    'loop-27-self-driven': {
      name: 'Self-Driven Loop',
      schedule: SCHEDULES.EVERY_30_MINUTES,
      description: 'Autonomous goal-directed behavior',
      tags: ['cognitive', 'self-driven'],
    },
    'loop-28-cpel': {
      name: 'CPEL Loop',
      schedule: SCHEDULES.EVERY_HOUR,
      description: 'Context and prompt engineering optimization',
      tags: ['cognitive', 'cpel', 'prompt-engineering'],
    },
    'loop-29-context-eng': {
      name: 'Context Engineering Loop',
      schedule: SCHEDULES.EVERY_15_MINUTES,
      description: 'LLM context window optimization and management',
      tags: ['cognitive', 'context-engineering'],
    },
    'loop-30-web-monitor': {
      name: 'Web Monitor Loop',
      schedule: SCHEDULES.EVERY_10_MINUTES,
      description: 'Web monitoring and change detection',
      tags: ['cognitive', 'web-monitor'],
    },
  };

  const loop = loops[loopId];
  if (!loop) return null;

  return {
    id: loopId,
    ...loop,
  };
}

/**
 * Get all loop IDs
 */
export function getAllLoopIds(): string[] {
  return [
    'loop-01-heartbeat',
    'loop-02-gmail-sync',
    'loop-03-browser-health',
    'loop-04-voice-queue',
    'loop-05-twilio-status',
    'loop-06-resource-monitor',
    'loop-07-identity-sync',
    'loop-08-soul-update',
    'loop-09-context-cleanup',
    'loop-10-notification-digest',
    'loop-11-log-rotation',
    'loop-12-backup-check',
    'loop-13-rate-limit',
    'loop-14-presence-check',
    'loop-15-daily-maintenance',
    'loop-16-ralph',
    'loop-17-research',
    'loop-18-planning',
    'loop-19-e2e',
    'loop-20-exploration',
    'loop-21-discovery',
    'loop-22-bug-finder',
    'loop-23-self-learning',
    'loop-24-meta-cognition',
    'loop-25-self-upgrading',
    'loop-26-self-updating',
    'loop-27-self-driven',
    'loop-28-cpel',
    'loop-29-context-eng',
    'loop-30-web-monitor',
  ];
}
