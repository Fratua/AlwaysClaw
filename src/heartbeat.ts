/**
 * Heartbeat System - Health Monitoring
 * OpenClaw AI Agent Scheduler
 */

import { EventEmitter } from 'events';

export interface HeartbeatConfig {
  interval: number;
  timeout: number;
  missedThreshold: number;
  nodeId: string;
  metadata?: Record<string, any>;
}

export interface HeartbeatMessage {
  nodeId: string;
  timestamp: number;
  sequence: number;
  metadata: Record<string, any>;
  status: 'healthy' | 'degraded' | 'unhealthy';
}

export interface NodeHealth {
  nodeId: string;
  lastHeartbeat: number;
  missedCount: number;
  status: 'online' | 'suspect' | 'offline';
  healthScore: number;
  metadata: Record<string, any>;
}

export class HeartbeatSender extends EventEmitter {
  private sequence = 0;
  private intervalId: NodeJS.Timeout | null = null;
  private isRunning = false;

  constructor(private config: HeartbeatConfig) {
    super();
  }

  /**
   * Start sending heartbeats
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.emit('started', { nodeId: this.config.nodeId });

    this.intervalId = setInterval(() => {
      this.sendHeartbeat();
    }, this.config.interval);
  }

  /**
   * Stop sending heartbeats
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isRunning = false;
    this.emit('stopped', { nodeId: this.config.nodeId });
  }

  /**
   * Send a heartbeat message
   */
  private sendHeartbeat(): void {
    const heartbeat: HeartbeatMessage = {
      nodeId: this.config.nodeId,
      timestamp: Date.now(),
      sequence: this.sequence++,
      metadata: {
        ...this.config.metadata,
        memory: process.memoryUsage(),
        uptime: process.uptime(),
      },
      status: this.calculateStatus(),
    };

    this.emit('heartbeat', heartbeat);
  }

  /**
   * Calculate current status based on system health
   */
  private calculateStatus(): HeartbeatMessage['status'] {
    const memUsage = process.memoryUsage();
    const heapUsedPercent = memUsage.heapUsed / memUsage.heapTotal;

    if (heapUsedPercent > 0.9) {
      return 'unhealthy';
    } else if (heapUsedPercent > 0.75) {
      return 'degraded';
    }
    return 'healthy';
  }

  /**
   * Update metadata dynamically
   */
  updateMetadata(metadata: Record<string, any>): void {
    this.config.metadata = { ...this.config.metadata, ...metadata };
  }

  /**
   * Check if heartbeat is running
   */
  isActive(): boolean {
    return this.isRunning;
  }
}

export class HeartbeatMonitor extends EventEmitter {
  private nodes: Map<string, NodeHealth> = new Map();
  private checkIntervalId: NodeJS.Timeout | null = null;

  constructor(
    private config: {
      checkInterval: number;
      missedThreshold: number;
    }
  ) {
    super();
  }

  /**
   * Start monitoring
   */
  start(): void {
    this.checkIntervalId = setInterval(() => {
      this.checkNodes();
    }, this.config.checkInterval);
    this.emit('started');
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.checkIntervalId) {
      clearInterval(this.checkIntervalId);
      this.checkIntervalId = null;
    }
    this.emit('stopped');
  }

  /**
   * Receive a heartbeat from a node
   */
  receiveHeartbeat(heartbeat: HeartbeatMessage): void {
    const existing = this.nodes.get(heartbeat.nodeId);

    if (existing) {
      // Update existing node
      existing.lastHeartbeat = heartbeat.timestamp;
      existing.missedCount = 0;
      existing.status = 'online';
      existing.metadata = heartbeat.metadata;
      
      // Calculate health score
      existing.healthScore = this.calculateHealthScore(existing, heartbeat);
    } else {
      // Register new node
      this.nodes.set(heartbeat.nodeId, {
        nodeId: heartbeat.nodeId,
        lastHeartbeat: heartbeat.timestamp,
        missedCount: 0,
        status: 'online',
        healthScore: 100,
        metadata: heartbeat.metadata,
      });

      this.emit('node:registered', { nodeId: heartbeat.nodeId });
    }

    this.emit('heartbeat:received', heartbeat);
  }

  /**
   * Check all nodes for timeouts
   */
  private checkNodes(): void {
    const now = Date.now();

    for (const [nodeId, node] of this.nodes) {
      const timeSinceLastHeartbeat = now - node.lastHeartbeat;

      if (timeSinceLastHeartbeat > this.config.missedThreshold * 1000) {
        node.missedCount++;

        if (node.missedCount >= 3) {
          if (node.status !== 'offline') {
            node.status = 'offline';
            this.emit('node:offline', { nodeId, missedCount: node.missedCount });
          }
        } else if (node.missedCount >= 1) {
          if (node.status !== 'suspect') {
            node.status = 'suspect';
            this.emit('node:suspect', { nodeId, missedCount: node.missedCount });
          }
        }

        node.healthScore = Math.max(0, node.healthScore - 20);
      }
    }
  }

  /**
   * Calculate health score based on various factors
   */
  private calculateHealthScore(
    node: NodeHealth,
    heartbeat: HeartbeatMessage
  ): number {
    let score = 100;

    // Deduct for missed heartbeats
    score -= node.missedCount * 10;

    // Deduct for degraded status
    if (heartbeat.status === 'degraded') {
      score -= 15;
    } else if (heartbeat.status === 'unhealthy') {
      score -= 30;
    }

    return Math.max(0, score);
  }

  /**
   * Get node health
   */
  getNodeHealth(nodeId: string): NodeHealth | undefined {
    return this.nodes.get(nodeId);
  }

  /**
   * Get all node healths
   */
  getAllNodes(): NodeHealth[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Get healthy nodes only
   */
  getHealthyNodes(): NodeHealth[] {
    return this.getAllNodes().filter(n => n.status === 'online');
  }

  /**
   * Remove a node
   */
  removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.emit('node:removed', { nodeId });
  }

  /**
   * Get system overview
   */
  getOverview(): {
    total: number;
    online: number;
    suspect: number;
    offline: number;
    averageHealthScore: number;
  } {
    const nodes = this.getAllNodes();
    const total = nodes.length;
    const online = nodes.filter(n => n.status === 'online').length;
    const suspect = nodes.filter(n => n.status === 'suspect').length;
    const offline = nodes.filter(n => n.status === 'offline').length;
    const averageHealthScore = total > 0
      ? nodes.reduce((sum, n) => sum + n.healthScore, 0) / total
      : 0;

    return { total, online, suspect, offline, averageHealthScore };
  }
}
