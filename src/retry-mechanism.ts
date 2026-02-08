/**
 * Retry Mechanism - Fault Tolerance
 * OpenClaw AI Agent Scheduler
 */

export type RetryStrategy = 'fixed' | 'exponential' | 'linear' | 'custom';

export interface RetryConfig {
  maxAttempts: number;
  strategy: RetryStrategy;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors?: string[];
  nonRetryableErrors?: string[];
  onRetry?: (attempt: number, error: Error, delay: number) => void;
  onFailed?: (error: Error, attempts: number) => void;
}

export interface RetryResult<T> {
  success: boolean;
  result?: T;
  error?: Error;
  attempts: number;
  totalDuration: number;
}

export class RetryMechanism {
  private static readonly defaultConfig: RetryConfig = {
    maxAttempts: 3,
    strategy: 'exponential',
    baseDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
  };

  private config: RetryConfig;

  constructor(config: Partial<RetryConfig> = {}) {
    this.config = { ...RetryMechanism.defaultConfig, ...config };
  }

  /**
   * Execute with retry logic
   */
  async execute<T>(
    fn: () => Promise<T>,
    customConfig?: Partial<RetryConfig>
  ): Promise<RetryResult<T>> {
    const config: RetryConfig = { ...this.config, ...customConfig };
    const startTime = Date.now();
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
      try {
        const result = await fn();
        return {
          success: true,
          result,
          attempts: attempt,
          totalDuration: Date.now() - startTime,
        };
      } catch (error) {
        lastError = error as Error;

        // Check if error is non-retryable
        if (this.isNonRetryable(error as Error, config)) {
          break;
        }

        // Check if error is retryable (if whitelist specified)
        if (!this.isRetryable(error as Error, config)) {
          break;
        }

        // Don't retry on last attempt
        if (attempt === config.maxAttempts) {
          break;
        }

        // Calculate delay
        const delay = this.calculateDelay(attempt, config);

        // Call retry callback
        config.onRetry?.(attempt, lastError, delay);

        // Wait before retry
        await this.sleep(delay);
      }
    }

    // All attempts failed
    config.onFailed?.(lastError!, config.maxAttempts);

    return {
      success: false,
      error: lastError,
      attempts: config.maxAttempts,
      totalDuration: Date.now() - startTime,
    };
  }

  /**
   * Calculate retry delay based on strategy
   */
  private calculateDelay(attempt: number, config: RetryConfig): number {
    let delay: number;

    switch (config.strategy) {
      case 'fixed':
        delay = config.baseDelay;
        break;
      case 'linear':
        delay = config.baseDelay * attempt;
        break;
      case 'exponential':
        delay = config.baseDelay * Math.pow(config.backoffMultiplier, attempt - 1);
        break;
      case 'custom':
        delay = this.customDelay(attempt, config);
        break;
      default:
        delay = config.baseDelay;
    }

    // Cap at max delay
    return Math.min(delay, config.maxDelay);
  }

  /**
   * Custom delay calculation (override for custom strategy)
   */
  protected customDelay(attempt: number, config: RetryConfig): number {
    return config.baseDelay * attempt;
  }

  /**
   * Check if error is retryable
   */
  private isRetryable(error: Error, config: RetryConfig): boolean {
    if (!config.retryableErrors || config.retryableErrors.length === 0) {
      return true; // Retry all by default
    }

    return config.retryableErrors.some(pattern =>
      error.message.includes(pattern) || error.name.includes(pattern)
    );
  }

  /**
   * Check if error is non-retryable
   */
  private isNonRetryable(error: Error, config: RetryConfig): boolean {
    if (!config.nonRetryableErrors || config.nonRetryableErrors.length === 0) {
      return false;
    }

    return config.nonRetryableErrors.some(pattern =>
      error.message.includes(pattern) || error.name.includes(pattern)
    );
  }

  /**
   * Sleep helper
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Circuit breaker pattern for failure handling
 */
export class CircuitBreaker {
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  private failureCount = 0;
  private lastFailureTime: number = 0;
  private nextAttempt: number = 0;
  private halfOpenCalls = 0;

  constructor(
    private config: {
      failureThreshold: number;
      resetTimeout: number;
      halfOpenMaxCalls: number;
    }
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() < this.nextAttempt) {
        throw new Error(`Circuit breaker is OPEN - retry after ${new Date(this.nextAttempt).toISOString()}`);
      }
      this.state = 'half-open';
      this.halfOpenCalls = 0;
    }

    if (this.state === 'half-open' && this.halfOpenCalls >= this.config.halfOpenMaxCalls) {
      throw new Error('Circuit breaker half-open call limit reached');
    }

    if (this.state === 'half-open') {
      this.halfOpenCalls++;
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;
    this.state = 'closed';
    this.halfOpenCalls = 0;
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.state === 'half-open') {
      // Failed during half-open, go back to open
      this.state = 'open';
      this.nextAttempt = Date.now() + this.config.resetTimeout;
    } else if (this.failureCount >= this.config.failureThreshold) {
      this.state = 'open';
      this.nextAttempt = Date.now() + this.config.resetTimeout;
    }
  }

  getState(): { state: string; failureCount: number; nextAttempt?: Date } {
    return {
      state: this.state,
      failureCount: this.failureCount,
      nextAttempt: this.state === 'open' ? new Date(this.nextAttempt) : undefined,
    };
  }

  /**
   * Force reset the circuit breaker
   */
  reset(): void {
    this.state = 'closed';
    this.failureCount = 0;
    this.halfOpenCalls = 0;
  }
}
