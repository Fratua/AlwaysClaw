// src/politeness/RateLimiter.ts
// Rate limiting and request throttling

import { EventEmitter } from 'events';
import { RateLimitConfig, RequestRecord } from './types';

export class RateLimiter extends EventEmitter {
  private config: RateLimitConfig;
  private requestHistory: RequestRecord[] = [];
  private activeRequests = 0;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<RateLimitConfig> = {}) {
    super();
    this.config = {
      requestsPerSecond: 1,
      requestsPerMinute: 20,
      requestsPerHour: 200,
      requestsPerDay: 2000,
      concurrentRequests: 3,
      ...config
    };
    this.cleanupInterval = setInterval(() => this.cleanOldRecords(), 60000);
  }

  async acquire(domain: string, weight: number = 1): Promise<void> {
    const domainConfig = this.getDomainConfig(domain);

    while (this.activeRequests >= domainConfig.concurrentRequests) {
      await this.delay(100);
    }

    await this.waitForRateLimit(domain, weight, domainConfig);
    this.recordRequest(domain, weight);
    this.activeRequests++;
    this.emit('acquired', { domain, weight, active: this.activeRequests });
  }

  release(domain: string, weight: number = 1): void {
    this.activeRequests = Math.max(0, this.activeRequests - 1);
    this.emit('released', { domain, weight, active: this.activeRequests });
  }

  private async waitForRateLimit(domain: string, weight: number, config: RateLimitConfig): Promise<void> {
    const now = Date.now();

    while (true) {
      const limits = [
        { limit: config.requestsPerSecond, window: 1000 },
        { limit: config.requestsPerMinute, window: 60000 },
        { limit: config.requestsPerHour, window: 3600000 },
        { limit: config.requestsPerDay, window: 86400000 }
      ];

      let shouldWait = false;
      let waitTime = 0;

      for (const { limit, window } of limits) {
        const count = this.countRequests(domain, window);
        if (count + weight > limit) {
          const oldestRequest = this.getOldestRequest(domain, window);
          if (oldestRequest) {
            const timeToWait = window - (now - oldestRequest.timestamp);
            if (timeToWait > waitTime) {
              waitTime = timeToWait;
              shouldWait = true;
            }
          }
        }
      }

      if (!shouldWait) break;
      this.emit('rateLimited', { domain, waitTime });
      await this.delay(Math.max(100, Math.min(waitTime, 5000)));
    }
  }

  private countRequests(domain: string, windowMs: number): number {
    const cutoff = Date.now() - windowMs;
    return this.requestHistory
      .filter(r => r.domain === domain && r.timestamp > cutoff)
      .reduce((sum, r) => sum + r.weight, 0);
  }

  private getOldestRequest(domain: string, windowMs: number): RequestRecord | null {
    const cutoff = Date.now() - windowMs;
    const requests = this.requestHistory.filter(r => r.domain === domain && r.timestamp > cutoff);
    return requests.length > 0 ? requests.sort((a, b) => a.timestamp - b.timestamp)[0] : null;
  }

  private recordRequest(domain: string, weight: number): void {
    this.requestHistory.push({ timestamp: Date.now(), domain, weight });
  }

  private cleanOldRecords(): void {
    const cutoff = Date.now() - 86400000;
    this.requestHistory = this.requestHistory.filter(r => r.timestamp > cutoff);
  }

  private getDomainConfig(domain: string): RateLimitConfig {
    const specific = this.config.domainSpecific?.[domain];
    return specific ? { ...this.config, ...specific } : this.config;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  shutdown(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }

  getStats(domain?: string) {
    const now = Date.now();
    const relevant = domain ? this.requestHistory.filter(r => r.domain === domain) : this.requestHistory;

    return {
      totalRequests: relevant.length,
      activeRequests: this.activeRequests,
      requestsPerSecond: relevant.filter(r => r.timestamp > now - 1000).length,
      requestsPerMinute: relevant.filter(r => r.timestamp > now - 60000).length
    };
  }
}
