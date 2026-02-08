// src/renderers/BrowserPool.ts
// Playwright browser pool with stealth and fingerprinting

import { chromium, Browser, BrowserContext, Page } from 'playwright';
import { EventEmitter } from 'events';
import { PoolConfig, PooledPage } from './types';

export class BrowserPool extends EventEmitter {
  private browsers: Browser[] = [];
  private contexts: Map<Browser, BrowserContext[]> = new Map();
  private pages: PooledPage[] = [];
  private config: PoolConfig;
  private healthCheckInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<PoolConfig> = {}) {
    super();
    this.config = {
      maxBrowsers: 5,
      maxPagesPerBrowser: 10,
      headless: true,
      stealth: true,
      ...config
    };
    this.startHealthCheck();
  }

  async initialize(): Promise<void> {
    for (let i = 0; i < this.config.maxBrowsers; i++) {
      await this.createBrowser();
    }
    this.emit('initialized', { browsers: this.browsers.length });
  }

  private async createBrowser(): Promise<Browser> {
    const launchOptions: any = {
      headless: this.config.headless,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
        '--window-size=1920,1080',
        '--disable-blink-features=AutomationControlled'
      ]
    };

    if (this.config.proxy) {
      launchOptions.proxy = { server: this.config.proxy };
    }

    const browser = await chromium.launch(launchOptions);
    this.browsers.push(browser);
    this.contexts.set(browser, []);

    if (this.config.stealth) {
      await this.applyStealthPatches(browser);
    }

    this.emit('browserCreated', { browserId: browser.toString() });
    return browser;
  }

  private async applyStealthPatches(browser: Browser): Promise<void> {
    const context = await browser.newContext({
      userAgent: this.config.userAgent || this.getRandomUserAgent(),
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 1,
      locale: 'en-US',
      timezoneId: 'America/New_York',
      permissions: ['notifications']
    });

    await context.addInitScript(() => {
      Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
      Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
      Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
      
      const originalQuery = window.navigator.permissions.query;
      window.navigator.permissions.query = (parameters: any) => 
        parameters.name === 'notifications' 
          ? Promise.resolve({ state: Notification.permission } as PermissionStatus)
          : originalQuery(parameters);
    });
  }

  async acquirePage(): Promise<PooledPage> {
    let pooledPage = this.pages.find(p => !p.inUse);

    if (!pooledPage) {
      const browser = this.browsers.find(b => {
        const contexts = this.contexts.get(b) || [];
        return contexts.length < this.config.maxPagesPerBrowser;
      });

      if (browser) {
        pooledPage = await this.createPooledPage(browser);
      } else if (this.browsers.length < this.config.maxBrowsers) {
        const newBrowser = await this.createBrowser();
        pooledPage = await this.createPooledPage(newBrowser);
      } else {
        pooledPage = await this.waitForPage();
      }
    }

    pooledPage.inUse = true;
    pooledPage.lastUsed = new Date();
    this.emit('pageAcquired', { pageId: pooledPage.page.toString() });
    return pooledPage;
  }

  private async createPooledPage(browser: Browser): Promise<PooledPage> {
    const context = await browser.newContext();
    const page = await context.newPage();
    
    const pooledPage: PooledPage = {
      page, context, browser,
      inUse: false,
      lastUsed: new Date()
    };

    this.pages.push(pooledPage);
    const contexts = this.contexts.get(browser) || [];
    contexts.push(context);
    this.contexts.set(browser, contexts);

    return pooledPage;
  }

  private async waitForPage(timeout: number = 30000): Promise<PooledPage> {
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        const page = this.pages.find(p => !p.inUse);
        if (page) {
          clearInterval(checkInterval);
          clearTimeout(timeoutId);
          resolve(page);
        }
      }, 100);

      const timeoutId = setTimeout(() => {
        clearInterval(checkInterval);
        reject(new Error('Timeout waiting for available page'));
      }, timeout);
    });
  }

  async releasePage(pooledPage: PooledPage): Promise<void> {
    pooledPage.inUse = false;
    await pooledPage.context.clearCookies();
    this.emit('pageReleased', { pageId: pooledPage.page.toString() });
  }

  private startHealthCheck(): void {
    this.healthCheckInterval = setInterval(async () => {
      const now = new Date();
      const staleThreshold = 5 * 60 * 1000;

      for (const pooledPage of this.pages) {
        if (!pooledPage.inUse && now.getTime() - pooledPage.lastUsed.getTime() > staleThreshold) {
          await this.closePooledPage(pooledPage);
        }
      }
    }, 60000);
  }

  private async closePooledPage(pooledPage: PooledPage): Promise<void> {
    await pooledPage.page.close();
    await pooledPage.context.close();
    this.pages = this.pages.filter(p => p !== pooledPage);
    const contexts = this.contexts.get(pooledPage.browser) || [];
    this.contexts.set(pooledPage.browser, contexts.filter(c => c !== pooledPage.context));
  }

  private getRandomUserAgent(): string {
    const userAgents = [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ];
    return userAgents[Math.floor(Math.random() * userAgents.length)];
  }

  async shutdown(): Promise<void> {
    if (this.healthCheckInterval) clearInterval(this.healthCheckInterval);
    for (const pooledPage of this.pages) await this.closePooledPage(pooledPage);
    for (const browser of this.browsers) await browser.close();
    this.emit('shutdown');
  }
}
