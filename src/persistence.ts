/**
 * Persistence Manager - State Storage
 * OpenClaw AI Agent Scheduler
 */

import * as sqlite3 from 'sqlite3';
import { open, Database } from 'sqlite';
import * as path from 'path';
import * as fs from 'fs';

export interface PersistedJob {
  id: string;
  name: string;
  schedule: string;
  options: string; // JSON
  status: string; // JSON
  metrics: string; // JSON
  lastUpdated: number;
  createdAt: number;
}

export interface ExecutionLog {
  id: string;
  jobId: string;
  startTime: number;
  endTime: number;
  duration: number;
  success: boolean;
  error?: string;
}

export class PersistenceManager {
  private db: Database<sqlite3.Database, sqlite3.Statement> | null = null;
  private readonly dbPath: string;
  private readonly backupPath: string;
  private autoSaveInterval: NodeJS.Timeout | null = null;

  constructor(dataDir: string = './data') {
    this.dbPath = path.join(dataDir, 'scheduler.db');
    this.backupPath = path.join(dataDir, 'backups');
    
    // Ensure directories exist
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    if (!fs.existsSync(this.backupPath)) {
      fs.mkdirSync(this.backupPath, { recursive: true });
    }
  }

  /**
   * Initialize the database
   */
  async initialize(): Promise<void> {
    this.db = await open({
      filename: this.dbPath,
      driver: sqlite3.Database,
    });

    // Create tables
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        schedule TEXT NOT NULL,
        options TEXT NOT NULL,
        status TEXT NOT NULL,
        metrics TEXT NOT NULL,
        lastUpdated INTEGER NOT NULL,
        createdAt INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS execution_log (
        id TEXT PRIMARY KEY,
        jobId TEXT NOT NULL,
        startTime INTEGER NOT NULL,
        endTime INTEGER NOT NULL,
        duration INTEGER NOT NULL,
        success INTEGER NOT NULL,
        error TEXT,
        FOREIGN KEY (jobId) REFERENCES jobs(id)
      );

      CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updatedAt INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_execution_log_jobId ON execution_log(jobId);
      CREATE INDEX IF NOT EXISTS idx_execution_log_startTime ON execution_log(startTime);
      CREATE INDEX IF NOT EXISTS idx_jobs_lastUpdated ON jobs(lastUpdated);
    `);

    // Start auto-save
    this.startAutoSave();

    console.log('[Persistence] Database initialized at', this.dbPath);
  }

  /**
   * Save a job to the database
   */
  async saveJob(job: PersistedJob): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`
      INSERT OR REPLACE INTO jobs (id, name, schedule, options, status, metrics, lastUpdated, createdAt)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      job.id,
      job.name,
      job.schedule,
      job.options,
      job.status,
      job.metrics,
      Date.now(),
      job.createdAt || Date.now()
    ]);
  }

  /**
   * Load all jobs from the database
   */
  async loadJobs(): Promise<PersistedJob[]> {
    if (!this.db) throw new Error('Database not initialized');

    return await this.db.all<PersistedJob[]>(`SELECT * FROM jobs`);
  }

  /**
   * Get a specific job
   */
  async getJob(jobId: string): Promise<PersistedJob | undefined> {
    if (!this.db) throw new Error('Database not initialized');

    return await this.db.get<PersistedJob>(
      `SELECT * FROM jobs WHERE id = ?`,
      [jobId]
    );
  }

  /**
   * Delete a job from the database
   */
  async deleteJob(jobId: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`DELETE FROM jobs WHERE id = ?`, [jobId]);
    await this.db.run(`DELETE FROM execution_log WHERE jobId = ?`, [jobId]);
  }

  /**
   * Log a job execution
   */
  async logExecution(log: ExecutionLog): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`
      INSERT INTO execution_log (id, jobId, startTime, endTime, duration, success, error)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `, [
      log.id,
      log.jobId,
      log.startTime,
      log.endTime,
      log.duration,
      log.success ? 1 : 0,
      log.error
    ]);
  }

  /**
   * Get execution history for a job
   */
  async getExecutionHistory(jobId: string, limit: number = 100): Promise<ExecutionLog[]> {
    if (!this.db) throw new Error('Database not initialized');

    return await this.db.all<ExecutionLog[]>(`
      SELECT * FROM execution_log 
      WHERE jobId = ? 
      ORDER BY startTime DESC 
      LIMIT ?
    `, [jobId, limit]);
  }

  /**
   * Get execution statistics for a job
   */
  async getJobStats(jobId: string): Promise<{
    totalExecutions: number;
    successfulExecutions: number;
    failedExecutions: number;
    averageDuration: number;
    lastExecution: number | null;
  }> {
    if (!this.db) throw new Error('Database not initialized');

    const result = await this.db.get<{
      total: number;
      successful: number;
      failed: number;
      avgDuration: number;
      lastExecution: number;
    }>(`
      SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
        AVG(duration) as avgDuration,
        MAX(startTime) as lastExecution
      FROM execution_log
      WHERE jobId = ?
    `, [jobId]);

    return {
      totalExecutions: result?.total || 0,
      successfulExecutions: result?.successful || 0,
      failedExecutions: result?.failed || 0,
      averageDuration: Math.round(result?.avgDuration || 0),
      lastExecution: result?.lastExecution || null,
    };
  }

  /**
   * Save system state
   */
  async saveSystemState(key: string, value: any): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    await this.db.run(`
      INSERT OR REPLACE INTO system_state (key, value, updatedAt)
      VALUES (?, ?, ?)
    `, [key, JSON.stringify(value), Date.now()]);
  }

  /**
   * Load system state
   */
  async loadSystemState(key: string): Promise<any | null> {
    if (!this.db) throw new Error('Database not initialized');

    const row = await this.db.get<{ value: string }>(`
      SELECT value FROM system_state WHERE key = ?
    `, [key]);

    return row ? JSON.parse(row.value) : null;
  }

  /**
   * Create a backup
   */
  async createBackup(): Promise<string> {
    const timestamp = Date.now();
    const backupFile = path.join(
      this.backupPath,
      `scheduler_backup_${timestamp}.db`
    );

    await this.db?.exec(`VACUUM INTO '${backupFile}'`);
    console.log('[Persistence] Backup created:', backupFile);
    return backupFile;
  }

  /**
   * Restore from backup
   */
  async restoreFromBackup(backupFile: string): Promise<void> {
    if (!fs.existsSync(backupFile)) {
      throw new Error(`Backup file not found: ${backupFile}`);
    }

    // Close current connection
    await this.close();

    // Copy backup to main database
    fs.copyFileSync(backupFile, this.dbPath);

    // Reinitialize
    await this.initialize();
    console.log('[Persistence] Restored from backup:', backupFile);
  }

  /**
   * List available backups
   */
  listBackups(): string[] {
    if (!fs.existsSync(this.backupPath)) {
      return [];
    }

    return fs.readdirSync(this.backupPath)
      .filter(f => f.startsWith('scheduler_backup_') && f.endsWith('.db'))
      .map(f => path.join(this.backupPath, f))
      .sort((a, b) => fs.statSync(b).mtime.getTime() - fs.statSync(a).mtime.getTime());
  }

  /**
   * Clean old execution logs
   */
  async cleanOldLogs(olderThanDays: number = 30): Promise<number> {
    if (!this.db) throw new Error('Database not initialized');

    const cutoff = Date.now() - (olderThanDays * 24 * 60 * 60 * 1000);
    
    const result = await this.db.run(`
      DELETE FROM execution_log WHERE startTime < ?
    `, [cutoff]);

    const deleted = result.changes || 0;
    if (deleted > 0) {
      console.log(`[Persistence] Cleaned ${deleted} old log entries`);
    }
    return deleted;
  }

  /**
   * Clean old backups (keep only last N)
   */
  cleanOldBackups(keepCount: number = 10): void {
    const backups = this.listBackups();
    const toDelete = backups.slice(keepCount);
    
    for (const backup of toDelete) {
      try {
        fs.unlinkSync(backup);
        console.log('[Persistence] Deleted old backup:', backup);
      } catch (error) {
        console.error('[Persistence] Failed to delete backup:', backup, error);
      }
    }
  }

  /**
   * Start auto-save interval
   */
  private startAutoSave(): void {
    // Create backup every 5 minutes
    this.autoSaveInterval = setInterval(async () => {
      try {
        await this.createBackup();
        await this.cleanOldLogs(30);
        this.cleanOldBackups(10);
      } catch (error) {
        console.error('[Persistence] Auto-save error:', error);
      }
    }, 300000); // Every 5 minutes
  }

  /**
   * Close the database connection
   */
  async close(): Promise<void> {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
      this.autoSaveInterval = null;
    }
    await this.db?.close();
    this.db = null;
    console.log('[Persistence] Database connection closed');
  }
}
