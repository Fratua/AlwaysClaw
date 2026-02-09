/**
 * OpenClawAgent - I/O Worker
 * Handles external service integrations (Gmail, Browser, Twilio, TTS, STT)
 * Wired to Python bridge for Gmail, Twilio, TTS/STT.
 * Browser uses compiled TypeScript BrowserPool/PipelineOrchestrator.
 */

const WorkerBase = require('./worker-base');
const logger = require('./logger');
const { getBridge } = require('./python-bridge');
const fs = require('fs');
const path = require('path');
const os = require('os');

class IOWorker extends WorkerBase {
  constructor() {
    super();
    this.ioService = process.env.IO_SERVICE;
    this.service = null;
    this.requestQueue = [];
    this.isProcessing = false;
  }

  async onInitialize() {
    logger.info(`[IO Worker ${this.workerId}] Initializing ${this.ioService} service...`);

    // Initialize the specific I/O service
    switch(this.ioService) {
      case 'gmail':
        this.service = new GmailService();
        break;
      case 'browser':
        this.service = new BrowserService();
        break;
      case 'twilio':
        this.service = new TwilioService();
        break;
      case 'tts':
        this.service = new TTSService();
        break;
      case 'stt':
        this.service = new STTService();
        break;
      default:
        throw new Error(`Unknown I/O service: ${this.ioService}`);
    }

    await this.service.initialize();
    logger.info(`[IO Worker ${this.workerId}] ${this.ioService} service initialized`);
  }

  onMessage(msg) {
    if (msg.type === 'io-request') {
      const request = msg.data;
      this.service.execute(request).then(result => {
        this.sendToMaster({
          type: 'io-response',
          data: {
            requestId: request.requestId,
            service: request.service,
            action: request.action,
            result,
          }
        });
      }).catch(error => {
        this.sendToMaster({
          type: 'io-response',
          data: {
            requestId: request.requestId,
            service: request.service,
            action: request.action,
            error: error.message,
          }
        });
      });
      return;
    }
    super.onMessage(msg);
  }

  async onTask(data) {
    return await this.service.execute(data);
  }

  async onShutdown() {
    if (this.service) {
      await this.service.shutdown();
    }
  }
}

// Gmail Service - delegates to Python bridge
class GmailService {
  constructor() {
    this.bridge = null;
  }

  async initialize() {
    logger.info('[GmailService] Initializing via Python bridge...');
    this.bridge = getBridge();
    await this.bridge.start();
    logger.info('[GmailService] Bridge connected');
  }

  async execute(request) {
    switch(request.action) {
      case 'send':
        return this.bridge.call('gmail.send', request.data || {});
      case 'read':
        return this.bridge.call('gmail.read', request.data || {});
      case 'search':
        return this.bridge.call('gmail.search', request.data || {});
      case 'watch':
        return this.bridge.call('gmail.watch', request.data || {});
      default:
        throw new Error(`Unknown Gmail action: ${request.action}`);
    }
  }

  async shutdown() {
    logger.info('[GmailService] Shutting down...');
    if (this.bridge) await this.bridge.stop();
  }
}

// Browser Service - uses compiled TS BrowserPool + PipelineOrchestrator
class BrowserService {
  constructor() {
    this.browserPool = null;
    this.pipelineOrchestrator = null;
    this._sessionPages = new Map();
    this._pendingAcquisitions = new Map();
  }

  /**
   * Acquire a page for a session, serializing concurrent acquisitions per sessionId.
   */
  async _acquireSessionPage(sessionId) {
    // If already have a page for this session, return it
    const existing = this._sessionPages.get(sessionId);
    if (existing) return existing;

    // If there's already a pending acquisition for this session, await it
    const pending = this._pendingAcquisitions.get(sessionId);
    if (pending) {
      return pending;
    }

    // Start a new acquisition
    const acquisition = (async () => {
      const pooledPage = await this.browserPool.acquirePage();
      this._sessionPages.set(sessionId, pooledPage);
      this._pendingAcquisitions.delete(sessionId);
      return pooledPage;
    })();

    this._pendingAcquisitions.set(sessionId, acquisition);
    return acquisition;
  }

  async initialize() {
    logger.info('[BrowserService] Initializing...');

    try {
      // Try loading compiled TS modules
      const { BrowserPool } = require('./dist/BrowserPool');
      const { PipelineOrchestrator } = require('./dist/PipelineOrchestrator');
      const { RateLimiter } = require('./dist/RateLimiter');
      const { DeduplicationEngine } = require('./dist/DeduplicationEngine');

      this.browserPool = new BrowserPool({
        maxBrowsers: parseInt(process.env.BROWSER_MAX_INSTANCES || '3', 10),
        maxPagesPerBrowser: parseInt(process.env.BROWSER_MAX_PAGES_PER || '5', 10),
        headless: process.env.BROWSER_HEADLESS !== 'false',
        stealth: true,
      });

      const rateLimiter = new RateLimiter({
        requestsPerSecond: 1,
        requestsPerMinute: 20,
        requestsPerHour: 200,
        requestsPerDay: 2000,
        concurrentRequests: 3,
      });

      const dedupEngine = new DeduplicationEngine({
        method: 'hybrid',
        threshold: 0.85,
        storage: 'memory',
      });

      this.pipelineOrchestrator = new PipelineOrchestrator(
        this.browserPool,
        rateLimiter,
        dedupEngine
      );

      await this.browserPool.initialize();
      logger.info('[BrowserService] Browser pool and pipeline initialized');
    } catch (error) {
      logger.warn('[BrowserService] Compiled TS modules not available, using stub mode:', error.message);
      this.stubMode = true;
    }
  }

  async execute(request) {
    switch(request.action) {
      case 'navigate':
        return this.navigate(request.data);
      case 'click':
        return this.click(request.data);
      case 'type':
        return this.type(request.data);
      case 'screenshot':
        return this.screenshot(request.data);
      case 'evaluate':
        return this.evaluate(request.data);
      case 'scrape':
        return this.scrape(request.data);
      case 'close':
        return this.closePage(request.data);
      default:
        throw new Error(`Unknown browser action: ${request.action}`);
    }
  }

  async navigate(data) {
    logger.info(`[BrowserService] Navigating to ${data.url}`);
    if (this.browserPool) {
      const sessionId = data.sessionId || `session-${Date.now()}`;
      let pooledPage = await this._acquireSessionPage(sessionId);

      try {
        await pooledPage.page.goto(data.url, {
          waitUntil: data.waitUntil || 'domcontentloaded',
          timeout: data.timeout || 30000,
        });
        const title = await pooledPage.page.title();
        const content = await pooledPage.page.content();
        return { navigated: true, url: data.url, title, contentLength: content.length, sessionId };
      } catch (error) {
        // Release the page on error
        this._sessionPages.delete(sessionId);
        await this.browserPool.releasePage(pooledPage);
        throw error;
      }
    }
    return { navigated: true, url: data.url, stubMode: true, warning: 'BrowserService running in stub mode - no real browser available' };
  }

  async click(data) {
    logger.info(`[BrowserService] Clicking element: ${data.selector}`);
    if (this.browserPool) {
      const sessionId = data.sessionId;
      let pooledPage = sessionId && this._sessionPages.get(sessionId);
      const isTemporary = !pooledPage;

      if (!pooledPage) {
        pooledPage = await this.browserPool.acquirePage();
      }

      try {
        await pooledPage.page.click(data.selector, { timeout: data.timeout || 10000 });
        const result = { clicked: true, selector: data.selector, sessionId };
        // Release temporary pages that aren't tracked by a session
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        return result;
      } catch (error) {
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        throw error;
      }
    }
    return { clicked: true, stubMode: true, warning: 'BrowserService running in stub mode - no real browser available' };
  }

  async type(data) {
    logger.info(`[BrowserService] Typing into element: ${data.selector}`);
    if (this.browserPool) {
      const sessionId = data.sessionId;
      let pooledPage = sessionId && this._sessionPages.get(sessionId);
      const isTemporary = !pooledPage;

      if (!pooledPage) {
        pooledPage = await this.browserPool.acquirePage();
      }

      try {
        await pooledPage.page.fill(data.selector, data.text || '');
        const result = { typed: true, sessionId };
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        return result;
      } catch (error) {
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        throw error;
      }
    }
    return { typed: true, stubMode: true, warning: 'BrowserService running in stub mode - no real browser available' };
  }

  async screenshot(data) {
    logger.info('[BrowserService] Taking screenshot...');
    if (this.browserPool) {
      const sessionId = data.sessionId;
      let pooledPage = sessionId && this._sessionPages.get(sessionId);
      const isTemporary = !pooledPage;

      if (!pooledPage) {
        pooledPage = await this.browserPool.acquirePage();
      }

      try {
        const buffer = await pooledPage.page.screenshot({
          fullPage: data.fullPage || false,
          type: 'png',
        });
        // Write to temp file to avoid large base64 strings in IPC
        const tmpDir = os.tmpdir();
        const tmpFile = path.join(tmpDir, `screenshot-${Date.now()}-${Math.random().toString(36).substring(2, 8)}.png`);
        fs.writeFileSync(tmpFile, buffer);
        const result = { screenshotPath: tmpFile, size: buffer.length, sessionId };
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        return result;
      } catch (error) {
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        throw error;
      }
    }
    return { screenshot: null, stubMode: true, warning: 'BrowserService running in stub mode - no real browser available' };
  }

  async evaluate(data) {
    logger.info('[BrowserService] Evaluating script...');

    // Validate script against allowlist of known safe operations
    const ALLOWED_SCRIPTS = new Map([
      ['getTitle', '(() => document.title)()'],
      ['getUrl', '(() => window.location.href)()'],
      ['getBodyText', '(() => document.body?.innerText || "")()'],
      ['getLinks', '(() => Array.from(document.querySelectorAll("a[href]")).map(a => a.href))()'],
      ['getMetaTags', '(() => Array.from(document.querySelectorAll("meta")).map(m => ({name: m.name, content: m.content})))()'],
    ]);

    let scriptToRun;
    if (data.scriptName && ALLOWED_SCRIPTS.has(data.scriptName)) {
      scriptToRun = ALLOWED_SCRIPTS.get(data.scriptName);
    } else if (data.script) {
      // Reject scripts containing dangerous patterns
      const DANGEROUS_PATTERNS = /\b(fetch|XMLHttpRequest|document\.cookie|localStorage|sessionStorage|indexedDB|eval|Function|import\s*\(|require\s*\(|process\.|child_process|__proto__|constructor\s*\[)\b/i;
      if (DANGEROUS_PATTERNS.test(data.script)) {
        throw new Error('Script rejected: contains disallowed patterns. Use scriptName with an allowed script name instead.');
      }
      scriptToRun = data.script;
    } else {
      scriptToRun = '(() => null)()';
    }

    if (this.browserPool) {
      const sessionId = data.sessionId;
      let pooledPage = sessionId && this._sessionPages.get(sessionId);
      const isTemporary = !pooledPage;

      if (!pooledPage) {
        pooledPage = await this.browserPool.acquirePage();
      }

      try {
        const result = await pooledPage.page.evaluate(scriptToRun);
        const response = { result, sessionId };
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        return response;
      } catch (error) {
        if (isTemporary) {
          await this.browserPool.releasePage(pooledPage);
        }
        throw error;
      }
    }
    return { result: null, stubMode: true, warning: 'BrowserService running in stub mode - no real browser available' };
  }

  async scrape(data) {
    logger.info(`[BrowserService] Scraping: ${data.url}`);
    if (this.pipelineOrchestrator) {
      return this.pipelineOrchestrator.execute(data.url, data.pipeline);
    }
    return { success: false, error: 'Pipeline not initialized' };
  }

  async closePage(data) {
    logger.info('[BrowserService] Closing page...');
    const sessionId = data.sessionId;
    if (sessionId && this._sessionPages.has(sessionId)) {
      const pooledPage = this._sessionPages.get(sessionId);
      this._sessionPages.delete(sessionId);
      if (this.browserPool) {
        await this.browserPool.releasePage(pooledPage);
      }
      return { closed: true, sessionId };
    }
    return { closed: true };
  }

  async shutdown() {
    logger.info('[BrowserService] Shutting down...');
    // Release all session pages
    for (const [sessionId, pooledPage] of this._sessionPages) {
      if (this.browserPool) {
        try {
          await this.browserPool.releasePage(pooledPage);
        } catch (e) {
          logger.debug(`[BrowserService] Error releasing page ${sessionId}: ${e.message}`);
        }
      }
    }
    this._sessionPages.clear();

    if (this.browserPool) {
      await this.browserPool.shutdown();
    }
  }
}

// Twilio Service - delegates to Python bridge
class TwilioService {
  constructor() {
    this.bridge = null;
  }

  async initialize() {
    logger.info('[TwilioService] Initializing via Python bridge...');
    this.bridge = getBridge();
    await this.bridge.start();
    logger.info('[TwilioService] Bridge connected');
  }

  async execute(request) {
    switch(request.action) {
      case 'call':
        return this.bridge.call('twilio.call', request.data || {});
      case 'sms':
        return this.bridge.call('twilio.sms', request.data || {});
      case 'status':
        return this.bridge.call('twilio.status', request.data || {});
      default:
        throw new Error(`Unknown Twilio action: ${request.action}`);
    }
  }

  async shutdown() {
    logger.info('[TwilioService] Shutting down...');
    if (this.bridge) await this.bridge.stop();
  }
}

// TTS Service - delegates to Python bridge
class TTSService {
  constructor() {
    this.bridge = null;
  }

  async initialize() {
    logger.info('[TTSService] Initializing via Python bridge...');
    this.bridge = getBridge();
    await this.bridge.start();
    logger.info('[TTSService] Bridge connected');
  }

  async execute(request) {
    switch(request.action) {
      case 'speak':
        return this.bridge.call('tts.speak', request.data || {});
      case 'save':
        return this.bridge.call('tts.speak', {
          ...(request.data || {}),
          output_path: request.data?.outputPath,
        });
      default:
        throw new Error(`Unknown TTS action: ${request.action}`);
    }
  }

  async shutdown() {
    logger.info('[TTSService] Shutting down...');
    if (this.bridge) await this.bridge.stop();
  }
}

// STT Service - delegates to Python bridge (stub for now)
class STTService {
  constructor() {
    this.bridge = null;
  }

  async initialize() {
    logger.info('[STTService] Initializing via Python bridge...');
    this.bridge = getBridge();
    await this.bridge.start();
    logger.info('[STTService] Bridge connected');
  }

  async execute(request) {
    switch(request.action) {
      case 'transcribe':
        return this.bridge.call('stt.transcribe', request.data || {});
      case 'start-listening':
        return this.bridge.call('stt.start_listening', request.data || {});
      case 'stop-listening':
        return this.bridge.call('stt.stop_listening', request.data || {});
      default:
        throw new Error(`Unknown STT action: ${request.action}`);
    }
  }

  async shutdown() {
    logger.info('[STTService] Shutting down...');
    if (this.bridge) await this.bridge.stop();
  }
}

// Initialize worker if this file is run directly
if (require.main === module) {
  const worker = new IOWorker();
  worker.initialize().catch(error => {
    logger.error('[IO Worker] Initialization failed:', error);
    process.exit(1);
  });
}

module.exports = IOWorker;
