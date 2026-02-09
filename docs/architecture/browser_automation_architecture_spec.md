# OpenClaw Browser Automation & Control Architecture
## Technical Specification v1.0

**Project:** Windows 10 OpenClaw-inspired AI Agent System  
**Component:** Browser Automation & Control Module  
**Target Platform:** Windows 10  
**AI Engine:** GPT-5.2 with High Thinking Capability  
**Date:** January 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Framework Comparison Matrix](#framework-comparison-matrix)
4. [Puppeteer Integration](#puppeteer-integration)
5. [Playwright Integration](#playwright-integration)
6. [Selenium WebDriver Support](#selenium-webdriver-support)
7. [Browser Context & Page Management](#browser-context--page-management)
8. [Navigation & Page Loading Strategies](#navigation--page-loading-strategies)
9. [Element Selection & Interaction](#element-selection--interaction)
10. [Screenshot & PDF Generation](#screenshot--pdf-generation)
11. [Mobile Emulation & Responsive Testing](#mobile-emulation--responsive-testing)
12. [Network Interception & Session Management](#network-interception--session-management)
13. [AI Agent Integration Patterns](#ai-agent-integration-patterns)
14. [Security Considerations](#security-considerations)
15. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document provides a comprehensive technical specification for the Browser Automation & Control Architecture of the Windows 10 OpenClaw-inspired AI agent system. The architecture supports three major browser automation frameworks (Puppeteer, Playwright, Selenium) with a unified abstraction layer that enables seamless switching and fallback mechanisms.

### Key Capabilities

| Capability | Priority | Framework Support |
|------------|----------|-------------------|
| Cross-browser automation | High | Playwright, Selenium |
| Chrome/Chromium optimization | High | Puppeteer, Playwright |
| Form filling & data extraction | Critical | All frameworks |
| Screenshot & PDF generation | High | All frameworks |
| Mobile device emulation | Medium | All frameworks |
| Network interception | High | Playwright, Puppeteer |
| Session persistence | Critical | All frameworks |
| Parallel execution | Medium | Playwright, Selenium Grid |

---

## Architecture Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI Agent Core (GPT-5.2)                         │
│                    ┌─────────────────────────────┐                      │
│                    │   Agent Orchestration Layer  │                      │
│                    └──────────────┬──────────────┘                      │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                    Browser Automation Abstraction Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Puppeteer  │  │  Playwright  │  │   Selenium   │  │   Fallback   │ │
│  │   Adapter    │  │   Adapter    │  │   Adapter    │  │   Handler    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘ │
└─────────┼─────────────────┼─────────────────┼───────────────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                      Browser Instance Manager                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐ │
│  │   Chrome    │  │   Firefox   │  │    Edge     │  │  WebKit/Safari │ │
│  │  (Chromium) │  │   (Gecko)   │  │  (Chromium) │  │   (Playwright) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Automation Abstraction Layer**: Unified API for all browser frameworks
2. **Browser Instance Manager**: Lifecycle management for browser instances
3. **Context Manager**: Isolated browsing sessions with cookie/storage separation
4. **Action Executor**: High-level action composition and execution
5. **State Monitor**: Real-time browser state capture for AI reasoning
6. **Error Recovery**: Automatic retry and fallback mechanisms

---

## Framework Comparison Matrix

### Detailed Comparison

| Feature | Puppeteer | Playwright | Selenium |
|---------|-----------|------------|----------|
| **Created By** | Google Chrome Team | Microsoft (ex-Puppeteer team) | Selenium Project (2004) |
| **Primary Focus** | Chrome automation | Cross-browser testing | Multi-language automation |
| **Browser Support** | Chrome, Chromium, Firefox (experimental) | Chromium, Firefox, WebKit | Chrome, Firefox, Safari, Edge, Opera, IE |
| **Language Support** | JavaScript, TypeScript | JS, TS, Python, Java, C# | Java, Python, C#, Ruby, JS |
| **Protocol** | Chrome DevTools Protocol | CDP + Custom protocols | W3C WebDriver |
| **Speed** | Fast | Fastest | Slower (WebDriver overhead) |
| **Auto-wait** | Basic | Comprehensive | Manual |
| **Parallel Execution** | Manual (puppeteer-cluster) | Native support | Selenium Grid |
| **Mobile Emulation** | Built-in device profiles | Built-in + custom | Limited |
| **Network Interception** | Good | Excellent | Limited |
| **PDF Generation** | Excellent | Good | Basic |
| **Screenshot** | Excellent | Excellent | Good |
| **Trace Recording** | No | Yes (built-in) | No |
| **Package Size** | ~170 MB | ~150 MB | Varies by binding |
| **Community** | Large | Growing rapidly | Very large |
| **Documentation** | Good | Excellent | Extensive |

### Recommendation Matrix

| Use Case | Recommended Framework | Rationale |
|----------|----------------------|-----------|
| Primary AI agent browser | **Playwright** | Best balance of features, speed, reliability |
| Chrome-only operations | **Puppeteer** | Lighter weight, mature PDF/screenshot |
| Legacy browser support | **Selenium** | Only option for IE, Opera, older Safari |
| Parallel scraping at scale | **Playwright** | Native parallelization, better resource management |
| Enterprise integration | **Selenium** | Existing infrastructure, language flexibility |

---

## Puppeteer Integration

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Puppeteer Adapter                        │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Launcher   │  │   Page      │  │   Network Manager   │ │
│  │  Manager    │  │  Controller │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Device    │  │ Screenshot  │  │  Session Persistence│ │
│  │  Emulator   │  │   Engine    │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                            │
                    Chrome DevTools Protocol
                            │
┌───────────────────────────▼────────────────────────────────┐
│              Chrome/Chromium Browser                        │
└────────────────────────────────────────────────────────────┘
```

### Core Configuration

```javascript
// Puppeteer Configuration for OpenClaw Agent
const puppeteer = require('puppeteer');
const puppeteerExtra = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

// Apply stealth plugin to avoid detection
puppeteerExtra.use(StealthPlugin());

class PuppeteerAdapter {
  constructor(config = {}) {
    this.config = {
      headless: config.headless !== false, // Default to headless
      executablePath: config.executablePath || null,
      userDataDir: config.userDataDir || './browser-data',
      slowMo: config.slowMo || 0,
      defaultViewport: config.defaultViewport || { width: 1920, height: 1080 },
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
        '--window-size=1920,1080',
        '--disable-blink-features=AutomationControlled',
        ...(config.args || [])
      ],
      ignoreDefaultArgs: ['--enable-automation'],
      ...config
    };
    this.browser = null;
    this.pages = new Map();
  }

  async launch() {
    this.browser = await puppeteerExtra.launch(this.config);
    return this.browser;
  }

  async newPage(pageId = 'default') {
    if (!this.browser) await this.launch();
    const page = await this.browser.newPage();
    this.pages.set(pageId, page);
    return page;
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      this.pages.clear();
    }
  }
}

module.exports = { PuppeteerAdapter };
```

### Device Emulation

```javascript
// Device Emulation Module
const { KnownDevices } = require('puppeteer');

class DeviceEmulator {
  constructor(page) {
    this.page = page;
    this.devices = KnownDevices;
  }

  async emulate(deviceName) {
    const device = this.devices[deviceName];
    if (!device) {
      throw new Error(`Device "${deviceName}" not found. Available: ${Object.keys(this.devices).join(', ')}`);
    }
    await this.page.emulate(device);
  }

  async emulateCustom(config) {
    await this.page.setViewport({
      width: config.width,
      height: config.height,
      deviceScaleFactor: config.deviceScaleFactor || 1,
      isMobile: config.isMobile || false,
      hasTouch: config.hasTouch || false,
      isLandscape: config.isLandscape || false
    });

    if (config.userAgent) {
      await this.page.setUserAgent(config.userAgent);
    }
  }

  // Common device presets
  static get presets() {
    return {
      iPhone15Pro: 'iPhone 15 Pro',
      iPhone14ProMax: 'iPhone 14 Pro Max',
      iPadPro: 'iPad Pro',
      Pixel7: 'Pixel 7',
      GalaxyS23: 'Samsung Galaxy S23',
      DesktopHD: { width: 1920, height: 1080 },
      Desktop4K: { width: 3840, height: 2160 }
    };
  }
}
```

### Network Interception

```javascript
// Network Interception Module
class NetworkInterceptor {
  constructor(page) {
    this.page = page;
    this.interceptors = new Map();
  }

  async enable() {
    await this.page.setRequestInterception(true);
    this.page.on('request', this.handleRequest.bind(this));
  }

  async handleRequest(request) {
    const url = request.url();
    
    // Check for registered interceptors
    for (const [pattern, handler] of this.interceptors) {
      if (this.matchesPattern(url, pattern)) {
        await handler(request);
        return;
      }
    }
    
    // Default: continue request
    await request.continue();
  }

  matchesPattern(url, pattern) {
    if (typeof pattern === 'string') {
      return url.includes(pattern);
    }
    if (pattern instanceof RegExp) {
      return pattern.test(url);
    }
    return false;
  }

  addInterceptor(pattern, handler) {
    this.interceptors.set(pattern, handler);
  }

  // Mock API response
  mockResponse(pattern, responseData) {
    this.addInterceptor(pattern, async (request) => {
      await request.respond({
        status: responseData.status || 200,
        contentType: responseData.contentType || 'application/json',
        body: JSON.stringify(responseData.body)
      });
    });
  }

  // Block requests
  blockRequests(patterns) {
    patterns.forEach(pattern => {
      this.addInterceptor(pattern, async (request) => {
        await request.abort('blockedbyclient');
      });
    });
  }

  // Modify headers
  modifyHeaders(pattern, headerModifications) {
    this.addInterceptor(pattern, async (request) => {
      const headers = { ...request.headers(), ...headerModifications };
      await request.continue({ headers });
    });
  }
}
```

### Screenshot & PDF Generation

```javascript
// Screenshot and PDF Module
class MediaGenerator {
  constructor(page) {
    this.page = page;
  }

  async screenshot(options = {}) {
    const defaultOptions = {
      path: options.path || null,
      type: options.type || 'png',
      fullPage: options.fullPage !== false,
      clip: options.clip || null,
      omitBackground: options.omitBackground || false,
      encoding: options.encoding || 'binary'
    };

    return await this.page.screenshot(defaultOptions);
  }

  async screenshotElement(selector, options = {}) {
    const element = await this.page.$(selector);
    if (!element) {
      throw new Error(`Element "${selector}" not found`);
    }
    return await element.screenshot(options);
  }

  async generatePDF(options = {}) {
    const defaultOptions = {
      path: options.path || null,
      scale: options.scale || 1,
      displayHeaderFooter: options.displayHeaderFooter || false,
      headerTemplate: options.headerTemplate || '',
      footerTemplate: options.footerTemplate || '',
      printBackground: options.printBackground !== false,
      landscape: options.landscape || false,
      pageRanges: options.pageRanges || '',
      format: options.format || 'A4',
      width: options.width || null,
      height: options.height || null,
      margin: {
        top: options.margin?.top || '0.4in',
        right: options.margin?.right || '0.4in',
        bottom: options.margin?.bottom || '0.4in',
        left: options.margin?.left || '0.4in'
      },
      preferCSSPageSize: options.preferCSSPageSize || false
    };

    return await this.page.pdf(defaultOptions);
  }
}
```

---

## Playwright Integration

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Playwright Adapter                          │
├────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Browser   │  │   Context   │  │     Page Manager        │ │
│  │   Manager   │  │   Manager   │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    Route    │  │   Tracing   │  │    Video Recorder       │ │
│  │   Handler   │  │   Engine    │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    Chromium           Firefox             WebKit
   (CDP-based)       (Custom Protocol)   (Custom Protocol)
```

### Core Configuration

```javascript
// Playwright Configuration for OpenClaw Agent
const { chromium, firefox, webkit } = require('playwright');

class PlaywrightAdapter {
  constructor(config = {}) {
    this.config = {
      browserType: config.browserType || 'chromium', // chromium, firefox, webkit
      headless: config.headless !== false,
      executablePath: config.executablePath || undefined,
      slowMo: config.slowMo || 0,
      viewport: config.viewport || { width: 1920, height: 1080 },
      userAgent: config.userAgent || undefined,
      locale: config.locale || 'en-US',
      timezoneId: config.timezoneId || 'America/New_York',
      geolocation: config.geolocation || undefined,
      permissions: config.permissions || [],
      extraHTTPHeaders: config.extraHTTPHeaders || {},
      ignoreHTTPSErrors: config.ignoreHTTPSErrors || false,
      bypassCSP: config.bypassCSP || false,
      offline: config.offline || false,
      ...config
    };
    this.browser = null;
    this.context = null;
    this.pages = new Map();
  }

  getBrowserLauncher() {
    switch (this.config.browserType) {
      case 'firefox': return firefox;
      case 'webkit': return webkit;
      case 'chromium':
      default: return chromium;
    }
  }

  async launch() {
    const launcher = this.getBrowserLauncher();
    this.browser = await launcher.launch({
      headless: this.config.headless,
      executablePath: this.config.executablePath,
      slowMo: this.config.slowMo,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
      ]
    });
    return this.browser;
  }

  async newContext(contextOptions = {}) {
    if (!this.browser) await this.launch();
    
    this.context = await this.browser.newContext({
      viewport: contextOptions.viewport || this.config.viewport,
      userAgent: contextOptions.userAgent || this.config.userAgent,
      locale: contextOptions.locale || this.config.locale,
      timezoneId: contextOptions.timezoneId || this.config.timezoneId,
      geolocation: contextOptions.geolocation || this.config.geolocation,
      permissions: contextOptions.permissions || this.config.permissions,
      extraHTTPHeaders: contextOptions.extraHTTPHeaders || this.config.extraHTTPHeaders,
      ignoreHTTPSErrors: contextOptions.ignoreHTTPSErrors ?? this.config.ignoreHTTPSErrors,
      bypassCSP: contextOptions.bypassCSP ?? this.config.bypassCSP,
      offline: contextOptions.offline ?? this.config.offline,
      recordVideo: contextOptions.recordVideo || undefined,
      storageState: contextOptions.storageState || undefined
    });

    return this.context;
  }

  async newPage(pageId = 'default', contextOptions = {}) {
    if (!this.context) await this.newContext(contextOptions);
    const page = await this.context.newPage();
    this.pages.set(pageId, page);
    return page;
  }

  async close() {
    if (this.context) {
      await this.context.close();
      this.context = null;
    }
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
    this.pages.clear();
  }
}

module.exports = { PlaywrightAdapter };
```

### Advanced Network Routing

```javascript
// Playwright Network Routing Module
class PlaywrightRouter {
  constructor(page) {
    this.page = page;
    this.routes = [];
  }

  async route(pattern, handler) {
    await this.page.route(pattern, handler);
    this.routes.push({ pattern, handler });
  }

  // Mock API with JSON response
  async mockAPI(urlPattern, responseBody, status = 200) {
    await this.route(urlPattern, async (route) => {
      await route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(responseBody)
      });
    });
  }

  // Modify request before sending
  async modifyRequest(urlPattern, modifier) {
    await this.route(urlPattern, async (route) => {
      const request = route.request();
      const modified = await modifier(request);
      await route.continue(modified);
    });
  }

  // Fetch and modify response
  async modifyResponse(urlPattern, modifier) {
    await this.route(urlPattern, async (route) => {
      const response = await route.fetch();
      const modified = await modifier(response);
      await route.fulfill(modified);
    });
  }

  // Block specific resource types
  async blockResources(resourceTypes) {
    await this.route('**/*', async (route) => {
      if (resourceTypes.includes(route.request().resourceType())) {
        await route.abort();
      } else {
        await route.continue();
      }
    });
  }

  // Simulate network conditions
  async emulateNetworkConditions(conditions) {
    const context = this.page.context();
    await context.setOffline(conditions.offline || false);
    
    // Use CDP for network throttling (Chromium only)
    if (conditions.downloadThroughput !== undefined) {
      const session = await context.newCDPSession(this.page);
      await session.send('Network.emulateNetworkConditions', {
        offline: conditions.offline || false,
        downloadThroughput: conditions.downloadThroughput,
        uploadThroughput: conditions.uploadThroughput || conditions.downloadThroughput,
        latency: conditions.latency || 0
      });
    }
  }

  // Wait for specific network response
  async waitForResponse(urlPattern, options = {}) {
    return await this.page.waitForResponse(urlPattern, options);
  }

  // Wait for network idle
  async waitForNetworkIdle(idleTime = 500) {
    await this.page.waitForLoadState('networkidle', { timeout: idleTime });
  }
}
```

### Tracing and Recording

```javascript
// Playwright Tracing and Recording Module
class PlaywrightTracer {
  constructor(context) {
    this.context = context;
    this.tracing = false;
  }

  async startTracing(options = {}) {
    await this.context.tracing.start({
      screenshots: options.screenshots !== false,
      snapshots: options.snapshots !== false,
      sources: options.sources || false
    });
    this.tracing = true;
  }

  async stopTracing(outputPath) {
    if (!this.tracing) return;
    await this.context.tracing.stop({ path: outputPath });
    this.tracing = false;
  }

  // Enable video recording for context
  static getVideoConfig(options = {}) {
    return {
      dir: options.dir || './videos/',
      size: options.size || { width: 1280, height: 720 }
    };
  }
}
```

---

## Selenium WebDriver Support

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Selenium Adapter                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Driver    │  │   Wait      │  │   Action Chains     │  │
│  │   Manager   │  │   Engine    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Grid      │  │   Cookie    │  │   JavaScript Exec   │  │
│  │   Client    │  │   Manager   │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                    W3C WebDriver Protocol
                            │
┌───────────────────────────┼─────────────────────────────────┐
│              Browser-Specific WebDrivers                     │
│     ChromeDriver    GeckoDriver    EdgeDriver    SafariDriver│
└─────────────────────────────────────────────────────────────┘
```

### Core Configuration

```javascript
// Selenium Configuration for OpenClaw Agent
const { Builder, By, until, Key } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
const firefox = require('selenium-webdriver/firefox');

class SeleniumAdapter {
  constructor(config = {}) {
    this.config = {
      browser: config.browser || 'chrome', // chrome, firefox, edge, safari
      headless: config.headless !== false,
      windowSize: config.windowSize || { width: 1920, height: 1080 },
      implicitWait: config.implicitWait || 10000,
      pageLoadTimeout: config.pageLoadTimeout || 30000,
      scriptTimeout: config.scriptTimeout || 30000,
      userDataDir: config.userDataDir || null,
      ...config
    };
    this.driver = null;
  }

  buildChromeOptions() {
    const options = new chrome.Options();
    
    if (this.config.headless) {
      options.addArguments('--headless=new');
    }
    
    options.addArguments(
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      `--window-size=${this.config.windowSize.width},${this.config.windowSize.height}`,
      '--disable-blink-features=AutomationControlled'
    );

    if (this.config.userDataDir) {
      options.addArguments(`--user-data-dir=${this.config.userDataDir}`);
    }

    // Exclude automation switch
    options.excludeSwitches('enable-automation');
    options.setExperimentalOption('useAutomationExtension', false);

    return options;
  }

  buildFirefoxOptions() {
    const options = new firefox.Options();
    
    if (this.config.headless) {
      options.addArguments('--headless');
    }
    
    options.setPreference('dom.webdriver.enabled', false);
    options.setPreference('useAutomationExtension', false);
    
    return options;
  }

  async launch() {
    const builder = new Builder();

    switch (this.config.browser) {
      case 'chrome':
        builder.forBrowser('chrome').setChromeOptions(this.buildChromeOptions());
        break;
      case 'firefox':
        builder.forBrowser('firefox').setFirefoxOptions(this.buildFirefoxOptions());
        break;
      case 'edge':
        builder.forBrowser('MicrosoftEdge');
        break;
      case 'safari':
        builder.forBrowser('safari');
        break;
      default:
        throw new Error(`Unsupported browser: ${this.config.browser}`);
    }

    this.driver = await builder.build();
    
    // Set timeouts
    await this.driver.manage().setTimeouts({
      implicit: this.config.implicitWait,
      pageLoad: this.config.pageLoadTimeout,
      script: this.config.scriptTimeout
    });

    return this.driver;
  }

  async navigateTo(url) {
    await this.driver.get(url);
  }

  async findElement(selector, timeout = null) {
    const waitTime = timeout || this.config.implicitWait;
    return await this.driver.wait(
      until.elementLocated(By.css(selector)),
      waitTime
    );
  }

  async findElements(selector) {
    return await this.driver.findElements(By.css(selector));
  }

  async close() {
    if (this.driver) {
      await this.driver.quit();
      this.driver = null;
    }
  }
}

module.exports = { SeleniumAdapter, By, until, Key };
```

### Advanced Waits

```javascript
// Selenium Wait Utilities
class SeleniumWaits {
  constructor(driver) {
    this.driver = driver;
  }

  async waitForElementVisible(selector, timeout = 10000) {
    return await this.driver.wait(
      until.elementIsVisible(await this.driver.findElement(By.css(selector))),
      timeout
    );
  }

  async waitForElementClickable(selector, timeout = 10000) {
    return await this.driver.wait(
      until.elementIsEnabled(await this.driver.findElement(By.css(selector))),
      timeout
    );
  }

  async waitForElementText(selector, text, timeout = 10000) {
    return await this.driver.wait(
      until.elementTextContains(await this.driver.findElement(By.css(selector)), text),
      timeout
    );
  }

  async waitForUrlContains(substring, timeout = 10000) {
    return await this.driver.wait(
      until.urlContains(substring),
      timeout
    );
  }

  async waitForCustom(condition, timeout = 10000) {
    return await this.driver.wait(condition, timeout);
  }
}
```

---

## Browser Context & Page Management

### Context Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Browser Context Manager                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Browser Instance                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │  Context 1  │  │  Context 2  │  │  Context N  │      │  │
│  │  │  (Isolated) │  │  (Isolated) │  │  (Isolated) │      │  │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │      │  │
│  │  │ │ Page 1  │ │  │ │ Page 1  │ │  │ │ Page 1  │ │      │  │
│  │  │ │ Page 2  │ │  │ │ Page 2  │ │  │ │ Page 2  │ │      │  │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │      │  │
│  │  │ Cookies     │  │ Cookies     │  │ Cookies     │      │  │
│  │  │ Storage     │  │ Storage     │  │ Storage     │      │  │
│  │  │ Cache       │  │ Cache       │  │ Cache       │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Unified Context Manager

```javascript
// Unified Context Manager for all frameworks
class BrowserContextManager {
  constructor(adapter) {
    this.adapter = adapter;
    this.contexts = new Map();
    this.activeContextId = null;
  }

  async createContext(contextId, options = {}) {
    let context;

    if (this.adapter.type === 'playwright') {
      context = await this.adapter.newContext(options);
    } else if (this.adapter.type === 'puppeteer') {
      // Puppeteer uses browser-level isolation
      context = {
        pages: new Map(),
        cookies: [],
        ...options
      };
    } else if (this.adapter.type === 'selenium') {
      // Selenium uses separate driver instances
      context = await this.adapter.launch();
    }

    this.contexts.set(contextId, context);
    this.activeContextId = contextId;
    return context;
  }

  async switchContext(contextId) {
    if (!this.contexts.has(contextId)) {
      throw new Error(`Context "${contextId}" not found`);
    }
    this.activeContextId = contextId;
    return this.contexts.get(contextId);
  }

  async closeContext(contextId) {
    const context = this.contexts.get(contextId);
    if (context) {
      if (this.adapter.type === 'playwright' && context.close) {
        await context.close();
      }
      this.contexts.delete(contextId);
    }
  }

  getActiveContext() {
    return this.contexts.get(this.activeContextId);
  }

  // Session persistence
  async saveSession(contextId, filePath) {
    const context = this.contexts.get(contextId);
    if (!context) return;

    if (this.adapter.type === 'playwright') {
      const storageState = await context.storageState();
      require('fs').writeFileSync(filePath, JSON.stringify(storageState, null, 2));
    } else if (this.adapter.type === 'puppeteer') {
      const cookies = await context.cookies();
      require('fs').writeFileSync(filePath, JSON.stringify(cookies, null, 2));
    }
  }

  async loadSession(contextId, filePath) {
    const storageState = JSON.parse(require('fs').readFileSync(filePath, 'utf8'));
    
    if (this.adapter.type === 'playwright') {
      await this.createContext(contextId, { storageState });
    } else if (this.adapter.type === 'puppeteer') {
      const context = this.contexts.get(contextId);
      if (context && context.setCookie) {
        for (const cookie of storageState.cookies || storageState) {
          await context.setCookie(cookie);
        }
      }
    }
  }
}
```

### Page Lifecycle Management

```javascript
// Page Lifecycle Manager
class PageLifecycleManager {
  constructor(adapter) {
    this.adapter = adapter;
    this.pages = new Map();
    this.pageCounter = 0;
  }

  async createPage(contextId = 'default', options = {}) {
    const pageId = `page_${++this.pageCounter}`;
    let page;

    if (this.adapter.type === 'playwright') {
      const context = this.adapter.contexts?.get(contextId);
      page = await context.newPage();
    } else if (this.adapter.type === 'puppeteer') {
      page = await this.adapter.browser.newPage();
    } else if (this.adapter.type === 'selenium') {
      // Selenium opens new window
      await this.adapter.driver.executeScript('window.open()');
      const handles = await this.adapter.driver.getAllWindowHandles();
      await this.adapter.driver.switchTo().window(handles[handles.length - 1]);
      page = this.adapter.driver;
    }

    this.pages.set(pageId, { page, contextId, createdAt: Date.now() });
    return { pageId, page };
  }

  async switchPage(pageId) {
    const pageInfo = this.pages.get(pageId);
    if (!pageInfo) {
      throw new Error(`Page "${pageId}" not found`);
    }

    if (this.adapter.type === 'selenium') {
      // Find and switch to correct window handle
      const handles = await this.adapter.driver.getAllWindowHandles();
      // Implementation depends on tracking handles
    }

    return pageInfo.page;
  }

  async closePage(pageId) {
    const pageInfo = this.pages.get(pageId);
    if (!pageInfo) return;

    if (this.adapter.type === 'playwright') {
      await pageInfo.page.close();
    } else if (this.adapter.type === 'puppeteer') {
      await pageInfo.page.close();
    } else if (this.adapter.type === 'selenium') {
      await pageInfo.page.close();
    }

    this.pages.delete(pageId);
  }

  getPageInfo(pageId) {
    return this.pages.get(pageId);
  }

  getAllPages() {
    return Array.from(this.pages.entries()).map(([id, info]) => ({
      pageId: id,
      contextId: info.contextId,
      createdAt: info.createdAt
    }));
  }
}
```

---

## Navigation & Page Loading Strategies

### Navigation Manager

```javascript
// Navigation Manager with multiple loading strategies
class NavigationManager {
  constructor(adapter) {
    this.adapter = adapter;
    this.navigationHistory = [];
    this.defaultTimeout = 30000;
  }

  async navigate(page, url, options = {}) {
    const startTime = Date.now();
    const strategy = options.strategy || 'load'; // load, domcontentloaded, networkidle

    try {
      if (this.adapter.type === 'playwright') {
        await page.goto(url, {
          waitUntil: strategy,
          timeout: options.timeout || this.defaultTimeout
        });
      } else if (this.adapter.type === 'puppeteer') {
        const waitOptions = {
          waitUntil: strategy === 'networkidle' ? 'networkidle2' : strategy,
          timeout: options.timeout || this.defaultTimeout
        };
        await page.goto(url, waitOptions);
      } else if (this.adapter.type === 'selenium') {
        await page.get(url);
        // Wait for page load
        await page.waitForCondition(
          'return document.readyState === "complete"',
          options.timeout || this.defaultTimeout
        );
      }

      const duration = Date.now() - startTime;
      this.navigationHistory.push({
        url,
        timestamp: startTime,
        duration,
        strategy,
        success: true
      });

      return { success: true, duration };
    } catch (error) {
      this.navigationHistory.push({
        url,
        timestamp: startTime,
        strategy,
        success: false,
        error: error.message
      });
      throw error;
    }
  }

  async goBack(page) {
    if (this.adapter.type === 'playwright') {
      await page.goBack();
    } else if (this.adapter.type === 'puppeteer') {
      await page.goBack();
    } else if (this.adapter.type === 'selenium') {
      await page.navigate().back();
    }
  }

  async goForward(page) {
    if (this.adapter.type === 'playwright') {
      await page.goForward();
    } else if (this.adapter.type === 'puppeteer') {
      await page.goForward();
    } else if (this.adapter.type === 'selenium') {
      await page.navigate().forward();
    }
  }

  async reload(page, options = {}) {
    if (this.adapter.type === 'playwright') {
      await page.reload({
        waitUntil: options.strategy || 'load'
      });
    } else if (this.adapter.type === 'puppeteer') {
      await page.reload({
        waitUntil: options.strategy === 'networkidle' ? 'networkidle2' : options.strategy
      });
    } else if (this.adapter.type === 'selenium') {
      await page.navigate().refresh();
    }
  }

  getNavigationHistory() {
    return this.navigationHistory;
  }

  clearHistory() {
    this.navigationHistory = [];
  }
}
```

### Loading Strategy Options

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `load` | Wait for `load` event | Standard page navigation |
| `domcontentloaded` | Wait for `DOMContentLoaded` | Fast navigation, JS not required |
| `networkidle` | Wait for network idle (no requests for 500ms) | SPAs, dynamic content |
| `networkidle2` | Wait for 2+ connections idle | Puppeteer-specific, less strict |
| `commit` | Wait for response headers | Minimal wait, fastest |

---

## Element Selection & Interaction

### Unified Element API

```javascript
// Unified Element Selection and Interaction
class ElementController {
  constructor(adapter) {
    this.adapter = adapter;
  }

  // Selection methods
  async find(page, selector, options = {}) {
    if (this.adapter.type === 'playwright') {
      const locator = page.locator(selector);
      if (options.visible) await locator.waitFor({ state: 'visible' });
      return locator;
    } else if (this.adapter.type === 'puppeteer') {
      if (options.visible) {
        await page.waitForSelector(selector, { visible: true });
      }
      return await page.$(selector);
    } else if (this.adapter.type === 'selenium') {
      const { By, until } = require('selenium-webdriver');
      if (options.visible) {
        await page.wait(until.elementLocated(By.css(selector)), options.timeout || 10000);
      }
      return await page.findElement(By.css(selector));
    }
  }

  async findAll(page, selector) {
    if (this.adapter.type === 'playwright') {
      return await page.locator(selector).all();
    } else if (this.adapter.type === 'puppeteer') {
      return await page.$$(selector);
    } else if (this.adapter.type === 'selenium') {
      const { By } = require('selenium-webdriver');
      return await page.findElements(By.css(selector));
    }
  }

  // Interaction methods
  async click(element, options = {}) {
    if (this.adapter.type === 'playwright') {
      await element.click(options);
    } else if (this.adapter.type === 'puppeteer') {
      await element.click(options);
    } else if (this.adapter.type === 'selenium') {
      await element.click();
    }
  }

  async fill(element, text, options = {}) {
    if (this.adapter.type === 'playwright') {
      await element.fill(text, options);
    } else if (this.adapter.type === 'puppeteer') {
      await element.type(text, { delay: options.delay || 0 });
    } else if (this.adapter.type === 'selenium') {
      await element.clear();
      await element.sendKeys(text);
    }
  }

  async clear(element) {
    if (this.adapter.type === 'playwright') {
      await element.fill('');
    } else if (this.adapter.type === 'puppeteer') {
      await element.click({ clickCount: 3 });
      await element.type('');
    } else if (this.adapter.type === 'selenium') {
      await element.clear();
    }
  }

  async selectOption(element, value) {
    if (this.adapter.type === 'playwright') {
      await element.selectOption(value);
    } else if (this.adapter.type === 'puppeteer') {
      await element.select(value);
    } else if (this.adapter.type === 'selenium') {
      const { Select } = require('selenium-webdriver');
      const select = new Select(element);
      await select.selectByValue(value);
    }
  }

  async getText(element) {
    if (this.adapter.type === 'playwright') {
      return await element.textContent();
    } else if (this.adapter.type === 'puppeteer') {
      return await element.evaluate(el => el.textContent);
    } else if (this.adapter.type === 'selenium') {
      return await element.getText();
    }
  }

  async getAttribute(element, attribute) {
    if (this.adapter.type === 'playwright') {
      return await element.getAttribute(attribute);
    } else if (this.adapter.type === 'puppeteer') {
      return await element.evaluate((el, attr) => el.getAttribute(attr), attribute);
    } else if (this.adapter.type === 'selenium') {
      return await element.getAttribute(attribute);
    }
  }

  async isVisible(element) {
    if (this.adapter.type === 'playwright') {
      return await element.isVisible();
    } else if (this.adapter.type === 'puppeteer') {
      return await element.evaluate(el => {
        const style = window.getComputedStyle(el);
        return style.display !== 'none' && style.visibility !== 'hidden';
      });
    } else if (this.adapter.type === 'selenium') {
      return await element.isDisplayed();
    }
  }

  async scrollIntoView(element) {
    if (this.adapter.type === 'playwright') {
      await element.scrollIntoViewIfNeeded();
    } else if (this.adapter.type === 'puppeteer') {
      await element.evaluate(el => el.scrollIntoView({ behavior: 'smooth', block: 'center' }));
    } else if (this.adapter.type === 'selenium') {
      await this.adapter.driver.executeScript('arguments[0].scrollIntoView(true)', element);
    }
  }

  async hover(element) {
    if (this.adapter.type === 'playwright') {
      await element.hover();
    } else if (this.adapter.type === 'puppeteer') {
      await element.hover();
    } else if (this.adapter.type === 'selenium') {
      const { Actions } = require('selenium-webdriver');
      const actions = new Actions(this.adapter.driver);
      await actions.moveToElement(element).perform();
    }
  }
}
```

### Selector Strategies

```javascript
// Selector Strategy Factory
class SelectorFactory {
  static create(type, value) {
    switch (type) {
      case 'css':
        return value;
      case 'xpath':
        return { type: 'xpath', value };
      case 'id':
        return `#${value}`;
      case 'class':
        return `.${value}`;
      case 'name':
        return `[name="${value}"]`;
      case 'text':
        return { type: 'text', value };
      case 'role':
        return { type: 'role', value };
      default:
        return value;
    }
  }

  // AI-friendly semantic selectors
  static semantic(label, options = {}) {
    return {
      type: 'semantic',
      label,
      elementType: options.elementType || 'button',
      context: options.context || null
    };
  }
}

// Semantic Element Finder (for AI agents)
class SemanticElementFinder {
  constructor(page) {
    this.page = page;
  }

  async findByLabel(label, elementType = 'button') {
    // Try multiple strategies
    const strategies = [
      // Exact text match
      `//${elementType}[text()="${label}"]`,
      // Contains text
      `//${elementType}[contains(text(), "${label}")]`,
      // Aria label
      `//${elementType}[@aria-label="${label}"]`,
      // Title attribute
      `//${elementType}[@title="${label}"]`,
      // Associated label
      `//label[text()="${label}"]//following::${elementType}[1]`,
      // Placeholder
      `//input[@placeholder="${label}"]`,
    ];

    for (const xpath of strategies) {
      try {
        const element = await this.page.$('xpath=' + xpath);
        if (element) return element;
      } catch (e) {
        continue;
      }
    }

    return null;
  }

  async findFormField(label) {
    const strategies = [
      `//label[contains(text(), "${label}")]//following::input[1]`,
      `//label[contains(text(), "${label}")]//following::textarea[1]`,
      `//label[contains(text(), "${label}")]//following::select[1]`,
      `//input[@placeholder="${label}"]`,
      `//input[@aria-label="${label}"]`,
      `//input[@name="${label.toLowerCase().replace(/\s+/g, '_')}"]`,
    ];

    for (const xpath of strategies) {
      try {
        const element = await this.page.$('xpath=' + xpath);
        if (element) return element;
      } catch (e) {
        continue;
      }
    }

    return null;
  }
}
```

---

## Screenshot & PDF Generation

### Media Capture Module

```javascript
// Unified Screenshot and PDF Generation
class MediaCapture {
  constructor(adapter) {
    this.adapter = adapter;
    this.defaultScreenshotDir = './screenshots/';
    this.defaultPDFDir = './pdfs/';
  }

  async screenshot(page, options = {}) {
    const timestamp = Date.now();
    const filename = options.filename || `screenshot_${timestamp}.png`;
    const filepath = options.path || `${this.defaultScreenshotDir}${filename}`;

    // Ensure directory exists
    const fs = require('fs');
    const path = require('path');
    const dir = path.dirname(filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    if (this.adapter.type === 'playwright') {
      await page.screenshot({
        path: filepath,
        fullPage: options.fullPage !== false,
        clip: options.clip || undefined,
        type: options.type || 'png',
        quality: options.quality || undefined,
        omitBackground: options.omitBackground || false
      });
    } else if (this.adapter.type === 'puppeteer') {
      await page.screenshot({
        path: filepath,
        fullPage: options.fullPage !== false,
        clip: options.clip || undefined,
        type: options.type || 'png',
        quality: options.quality || undefined,
        omitBackground: options.omitBackground || false,
        encoding: options.encoding || 'binary'
      });
    } else if (this.adapter.type === 'selenium') {
      const screenshot = await page.takeScreenshot();
      fs.writeFileSync(filepath, screenshot, 'base64');
    }

    return filepath;
  }

  async screenshotElement(page, selector, options = {}) {
    const timestamp = Date.now();
    const filename = options.filename || `element_${timestamp}.png`;
    const filepath = options.path || `${this.defaultScreenshotDir}${filename}`;

    if (this.adapter.type === 'playwright') {
      const element = page.locator(selector);
      await element.screenshot({ path: filepath });
    } else if (this.adapter.type === 'puppeteer') {
      const element = await page.$(selector);
      if (!element) throw new Error(`Element "${selector}" not found`);
      await element.screenshot({ path: filepath });
    } else if (this.adapter.type === 'selenium') {
      const element = await page.findElement(By.css(selector));
      const screenshot = await element.takeScreenshot();
      fs.writeFileSync(filepath, screenshot, 'base64');
    }

    return filepath;
  }

  async generatePDF(page, options = {}) {
    const timestamp = Date.now();
    const filename = options.filename || `page_${timestamp}.pdf`;
    const filepath = options.path || `${this.defaultPDFDir}${filename}`;

    // Ensure directory exists
    const fs = require('fs');
    const path = require('path');
    const dir = path.dirname(filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    if (this.adapter.type === 'playwright') {
      await page.pdf({
        path: filepath,
        scale: options.scale || 1,
        displayHeaderFooter: options.displayHeaderFooter || false,
        headerTemplate: options.headerTemplate || '',
        footerTemplate: options.footerTemplate || '',
        printBackground: options.printBackground !== false,
        landscape: options.landscape || false,
        pageRanges: options.pageRanges || '',
        format: options.format || 'A4',
        width: options.width || undefined,
        height: options.height || undefined,
        margin: options.margin || {
          top: '0.4in',
          right: '0.4in',
          bottom: '0.4in',
          left: '0.4in'
        },
        preferCSSPageSize: options.preferCSSPageSize || false
      });
    } else if (this.adapter.type === 'puppeteer') {
      await page.pdf({
        path: filepath,
        scale: options.scale || 1,
        displayHeaderFooter: options.displayHeaderFooter || false,
        headerTemplate: options.headerTemplate || '',
        footerTemplate: options.footerTemplate || '',
        printBackground: options.printBackground !== false,
        landscape: options.landscape || false,
        pageRanges: options.pageRanges || '',
        format: options.format || 'A4',
        width: options.width || undefined,
        height: options.height || undefined,
        margin: options.margin || {
          top: '0.4in',
          right: '0.4in',
          bottom: '0.4in',
          left: '0.4in'
        },
        preferCSSPageSize: options.preferCSSPageSize || false
      });
    } else if (this.adapter.type === 'selenium') {
      // Selenium doesn't natively support PDF generation
      // Use Chrome DevTools Protocol or external tool
      throw new Error('PDF generation not natively supported in Selenium. Use Playwright or Puppeteer.');
    }

    return filepath;
  }

  // Compare screenshots for visual regression
  async compareScreenshots(baselinePath, currentPath, options = {}) {
    const { pixelmatch } = require('pixelmatch');
    const PNG = require('pngjs').PNG;
    const fs = require('fs');

    const baseline = PNG.sync.read(fs.readFileSync(baselinePath));
    const current = PNG.sync.read(fs.readFileSync(currentPath));

    const { width, height } = baseline;
    const diff = new PNG({ width, height });

    const numDiffPixels = pixelmatch(
      baseline.data,
      current.data,
      diff.data,
      width,
      height,
      {
        threshold: options.threshold || 0.1,
        includeAA: options.includeAA || false
      }
    );

    const diffPath = options.diffPath || currentPath.replace('.png', '_diff.png');
    fs.writeFileSync(diffPath, PNG.sync.write(diff));

    return {
      numDiffPixels,
      diffPercentage: (numDiffPixels / (width * height)) * 100,
      diffPath,
      matches: numDiffPixels === 0
    };
  }
}
```

### Screenshot Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | string | null | File path to save screenshot |
| `type` | string | 'png' | Image format ('png', 'jpeg', 'webp') |
| `quality` | number | null | JPEG quality (0-100) |
| `fullPage` | boolean | true | Capture full scrollable page |
| `clip` | object | null | Clip region {x, y, width, height} |
| `omitBackground` | boolean | false | Transparent background |
| `encoding` | string | 'binary' | 'binary' or 'base64' |

### PDF Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | string | null | File path to save PDF |
| `scale` | number | 1 | Scale factor (0.1-2) |
| `displayHeaderFooter` | boolean | false | Show header/footer |
| `headerTemplate` | string | '' | HTML header template |
| `footerTemplate` | string | '' | HTML footer template |
| `printBackground` | boolean | true | Print background graphics |
| `landscape` | boolean | false | Landscape orientation |
| `format` | string | 'A4' | Page format |
| `width` | string/number | null | Custom width |
| `height` | string/number | null | Custom height |
| `margin` | object | {} | Page margins |

---

## Mobile Emulation & Responsive Testing

### Device Emulation Manager

```javascript
// Mobile Device Emulation Manager
class MobileEmulationManager {
  constructor(adapter) {
    this.adapter = adapter;
    this.devicePresets = this.loadDevicePresets();
  }

  loadDevicePresets() {
    return {
      // iPhones
      'iPhone 15 Pro': {
        userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        viewport: { width: 393, height: 852 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      'iPhone 15 Pro Max': {
        userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        viewport: { width: 430, height: 932 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      'iPhone 14': {
        userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      // iPads
      'iPad Pro 12.9': {
        userAgent: 'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        viewport: { width: 1024, height: 1366 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      'iPad Mini': {
        userAgent: 'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        viewport: { width: 768, height: 1024 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      // Android
      'Pixel 7': {
        userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        viewport: { width: 412, height: 915 },
        deviceScaleFactor: 2.625,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      'Samsung Galaxy S23': {
        userAgent: 'Mozilla/5.0 (Linux; Android 14; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        viewport: { width: 384, height: 854 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        isLandscape: false
      },
      // Desktop
      'Desktop HD': {
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport: { width: 1920, height: 1080 },
        deviceScaleFactor: 1,
        isMobile: false,
        hasTouch: false,
        isLandscape: true
      },
      'Desktop 4K': {
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport: { width: 3840, height: 2160 },
        deviceScaleFactor: 1,
        isMobile: false,
        hasTouch: false,
        isLandscape: true
      }
    };
  }

  async emulateDevice(page, deviceName) {
    const device = this.devicePresets[deviceName];
    if (!device) {
      throw new Error(`Device "${deviceName}" not found. Available: ${Object.keys(this.devicePresets).join(', ')}`);
    }

    if (this.adapter.type === 'playwright') {
      // Playwright has built-in device emulation
      const devices = require('playwright').devices;
      const deviceConfig = devices[deviceName.replace(/\s+/g, '').toLowerCase()];
      if (deviceConfig) {
        await page.setViewportSize(deviceConfig.viewport);
        // Note: For full emulation, create new context with device
      } else {
        await page.setViewportSize(device.viewport);
      }
    } else if (this.adapter.type === 'puppeteer') {
      await page.setUserAgent(device.userAgent);
      await page.setViewport({
        width: device.viewport.width,
        height: device.viewport.height,
        deviceScaleFactor: device.deviceScaleFactor,
        isMobile: device.isMobile,
        hasTouch: device.hasTouch,
        isLandscape: device.isLandscape
      });
    } else if (this.adapter.type === 'selenium') {
      // Selenium requires driver-level configuration
      const { Builder } = require('selenium-webdriver');
      const chrome = require('selenium-webdriver/chrome');
      
      const mobileEmulation = {
        deviceMetrics: {
          width: device.viewport.width,
          height: device.viewport.height,
          pixelRatio: device.deviceScaleFactor
        },
        userAgent: device.userAgent
      };

      const options = new chrome.Options();
      options.setMobileEmulation(mobileEmulation);
      // Note: Requires driver restart
    }

    return device;
  }

  async setViewport(page, width, height, options = {}) {
    if (this.adapter.type === 'playwright') {
      await page.setViewportSize({ width, height });
    } else if (this.adapter.type === 'puppeteer') {
      await page.setViewport({
        width,
        height,
        deviceScaleFactor: options.deviceScaleFactor || 1,
        isMobile: options.isMobile || false,
        hasTouch: options.hasTouch || false
      });
    } else if (this.adapter.type === 'selenium') {
      await page.manage().window().setRect({ width, height });
    }
  }

  async emulateOrientation(page, landscape) {
    const currentViewport = await page.viewportSize?.() || { width: 1920, height: 1080 };
    const newWidth = landscape ? Math.max(currentViewport.width, currentViewport.height) : Math.min(currentViewport.width, currentViewport.height);
    const newHeight = landscape ? Math.min(currentViewport.width, currentViewport.height) : Math.max(currentViewport.width, currentViewport.height);
    
    await this.setViewport(page, newWidth, newHeight, { isLandscape: landscape });
  }

  async emulateNetworkConditions(page, conditions) {
    const profiles = {
      'slow-3g': { download: 400 * 1024 / 8, upload: 400 * 1024 / 8, latency: 2000 },
      'fast-3g': { download: 1.6 * 1024 * 1024 / 8, upload: 750 * 1024 / 8, latency: 150 },
      '4g': { download: 4 * 1024 * 1024 / 8, upload: 3 * 1024 * 1024 / 8, latency: 50 },
      'offline': { download: 0, upload: 0, latency: 0, offline: true }
    };

    const profile = profiles[conditions] || conditions;

    if (this.adapter.type === 'playwright') {
      const context = page.context();
      await context.setOffline(profile.offline || false);
      // For throttling, use CDP session
      const session = await context.newCDPSession(page);
      await session.send('Network.emulateNetworkConditions', {
        offline: profile.offline || false,
        downloadThroughput: profile.download,
        uploadThroughput: profile.upload,
        latency: profile.latency
      });
    } else if (this.adapter.type === 'puppeteer') {
      const client = await page.target().createCDPSession();
      await client.send('Network.emulateNetworkConditions', {
        offline: profile.offline || false,
        downloadThroughput: profile.download,
        uploadThroughput: profile.upload,
        latency: profile.latency
      });
    }
  }

  getAvailableDevices() {
    return Object.keys(this.devicePresets);
  }
}
```

### Responsive Testing Suite

```javascript
// Responsive Testing Suite
class ResponsiveTestingSuite {
  constructor(mediaCapture, mobileEmulation) {
    this.mediaCapture = mediaCapture;
    this.mobileEmulation = mobileEmulation;
    this.breakpoints = [
      { name: 'mobile-small', width: 320, height: 568 },
      { name: 'mobile', width: 375, height: 667 },
      { name: 'mobile-large', width: 414, height: 896 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'tablet-large', width: 1024, height: 1366 },
      { name: 'desktop', width: 1280, height: 800 },
      { name: 'desktop-large', width: 1920, height: 1080 },
      { name: 'desktop-4k', width: 3840, height: 2160 }
    ];
  }

  async captureResponsiveScreenshots(page, url, options = {}) {
    const results = [];
    const outputDir = options.outputDir || './responsive-tests/';

    // Navigate to URL first
    await page.goto(url);

    for (const breakpoint of this.breakpoints) {
      // Set viewport
      await this.mobileEmulation.setViewport(page, breakpoint.width, breakpoint.height);
      
      // Wait for layout to settle
      await page.waitForTimeout(500);

      // Capture screenshot
      const filename = `${breakpoint.name}_${breakpoint.width}x${breakpoint.height}.png`;
      const filepath = await this.mediaCapture.screenshot(page, {
        path: `${outputDir}${filename}`,
        fullPage: options.fullPage !== false
      });

      results.push({
        breakpoint: breakpoint.name,
        dimensions: { width: breakpoint.width, height: breakpoint.height },
        filepath
      });
    }

    return results;
  }

  async testDeviceMatrix(page, url, devices, options = {}) {
    const results = [];
    const outputDir = options.outputDir || './device-tests/';

    for (const deviceName of devices) {
      try {
        // Emulate device
        await this.mobileEmulation.emulateDevice(page, deviceName);
        
        // Navigate
        await page.goto(url, { waitUntil: 'networkidle' });

        // Capture screenshot
        const filename = `${deviceName.replace(/\s+/g, '_')}.png`;
        const filepath = await this.mediaCapture.screenshot(page, {
          path: `${outputDir}${filename}`,
          fullPage: options.fullPage !== false
        });

        results.push({
          device: deviceName,
          filepath,
          success: true
        });
      } catch (error) {
        results.push({
          device: deviceName,
          error: error.message,
          success: false
        });
      }
    }

    return results;
  }
}
```

---

## Network Interception & Session Management

### Session Persistence Manager

```javascript
// Session Persistence Manager
class SessionPersistenceManager {
  constructor(adapter) {
    this.adapter = adapter;
    this.storageDir = './sessions/';
    this.fs = require('fs');
    this.path = require('path');
  }

  ensureStorageDir() {
    if (!this.fs.existsSync(this.storageDir)) {
      this.fs.mkdirSync(this.storageDir, { recursive: true });
    }
  }

  getSessionPath(sessionId) {
    return this.path.join(this.storageDir, `${sessionId}.json`);
  }

  async saveSession(context, sessionId) {
    this.ensureStorageDir();
    const sessionPath = this.getSessionPath(sessionId);

    let sessionData;

    if (this.adapter.type === 'playwright') {
      sessionData = await context.storageState();
    } else if (this.adapter.type === 'puppeteer') {
      const cookies = await context.cookies();
      // Get localStorage and sessionStorage if available
      sessionData = {
        cookies,
        origins: [] // Would need page-level access for storage
      };
    } else if (this.adapter.type === 'selenium') {
      const cookies = await context.manage().getCookies();
      sessionData = { cookies };
    }

    // Encrypt sensitive data
    const encrypted = this.encryptSessionData(sessionData);
    this.fs.writeFileSync(sessionPath, JSON.stringify(encrypted, null, 2));

    return sessionPath;
  }

  async loadSession(sessionId) {
    const sessionPath = this.getSessionPath(sessionId);
    
    if (!this.fs.existsSync(sessionPath)) {
      throw new Error(`Session "${sessionId}" not found`);
    }

    const encrypted = JSON.parse(this.fs.readFileSync(sessionPath, 'utf8'));
    return this.decryptSessionData(encrypted);
  }

  encryptSessionData(data) {
    // Simple encryption - replace with proper encryption in production
    const crypto = require('crypto');
    const key = process.env.SESSION_ENCRYPTION_KEY || 'default-key-change-in-production';
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', Buffer.from(key.padEnd(32).slice(0, 32)), iv);
    
    const jsonData = JSON.stringify(data);
    let encrypted = cipher.update(jsonData, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    return {
      iv: iv.toString('hex'),
      data: encrypted
    };
  }

  decryptSessionData(encrypted) {
    const crypto = require('crypto');
    const key = process.env.SESSION_ENCRYPTION_KEY || 'default-key-change-in-production';
    const iv = Buffer.from(encrypted.iv, 'hex');
    const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(key.padEnd(32).slice(0, 32)), iv);

    let decrypted = decipher.update(encrypted.data, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return JSON.parse(decrypted);
  }

  async applySession(context, sessionData) {
    if (this.adapter.type === 'playwright') {
      // For Playwright, session is applied at context creation
      // This method would recreate context with storageState
      return sessionData;
    } else if (this.adapter.type === 'puppeteer') {
      if (sessionData.cookies) {
        for (const cookie of sessionData.cookies) {
          await context.setCookie(cookie);
        }
      }
    } else if (this.adapter.type === 'selenium') {
      await context.manage().deleteAllCookies();
      for (const cookie of sessionData.cookies || []) {
        await context.manage().addCookie(cookie);
      }
    }
  }

  listSessions() {
    this.ensureStorageDir();
    return this.fs.readdirSync(this.storageDir)
      .filter(f => f.endsWith('.json'))
      .map(f => f.replace('.json', ''));
  }

  deleteSession(sessionId) {
    const sessionPath = this.getSessionPath(sessionId);
    if (this.fs.existsSync(sessionPath)) {
      this.fs.unlinkSync(sessionPath);
      return true;
    }
    return false;
  }
}
```

### Cookie Manager

```javascript
// Cookie Manager
class CookieManager {
  constructor(adapter) {
    this.adapter = adapter;
  }

  async getCookies(context, url = null) {
    if (this.adapter.type === 'playwright') {
      return await context.cookies(url);
    } else if (this.adapter.type === 'puppeteer') {
      return await context.cookies(url);
    } else if (this.adapter.type === 'selenium') {
      return await context.manage().getCookies();
    }
  }

  async setCookie(context, cookie) {
    if (this.adapter.type === 'playwright') {
      await context.addCookies([cookie]);
    } else if (this.adapter.type === 'puppeteer') {
      await context.setCookie(cookie);
    } else if (this.adapter.type === 'selenium') {
      await context.manage().addCookie(cookie);
    }
  }

  async deleteCookie(context, name, url = null) {
    if (this.adapter.type === 'playwright') {
      const cookies = await context.cookies(url);
      const filtered = cookies.filter(c => c.name !== name);
      await context.clearCookies();
      await context.addCookies(filtered);
    } else if (this.adapter.type === 'puppeteer') {
      await context.deleteCookie({ name });
    } else if (this.adapter.type === 'selenium') {
      await context.manage().deleteCookieNamed(name);
    }
  }

  async clearCookies(context) {
    if (this.adapter.type === 'playwright') {
      await context.clearCookies();
    } else if (this.adapter.type === 'puppeteer') {
      const cookies = await context.cookies();
      for (const cookie of cookies) {
        await context.deleteCookie(cookie);
      }
    } else if (this.adapter.type === 'selenium') {
      await context.manage().deleteAllCookies();
    }
  }

  async getCookie(context, name, url = null) {
    const cookies = await this.getCookies(context, url);
    return cookies.find(c => c.name === name);
  }
}
```

---

## AI Agent Integration Patterns

### Browser State Observer

```javascript
// Browser State Observer for AI Agents
class BrowserStateObserver {
  constructor(adapter) {
    this.adapter = adapter;
    this.stateHistory = [];
    this.maxHistorySize = 100;
  }

  async captureState(page) {
    const state = {
      timestamp: Date.now(),
      url: await this.getCurrentUrl(page),
      title: await this.getPageTitle(page),
      viewport: await this.getViewport(page),
      elements: await this.getInteractiveElements(page)
    };

    this.addToHistory(state);
    return state;
  }

  async getCurrentUrl(page) {
    if (this.adapter.type === 'playwright') {
      return page.url();
    } else if (this.adapter.type === 'puppeteer') {
      return page.url();
    } else if (this.adapter.type === 'selenium') {
      return await page.getCurrentUrl();
    }
  }

  async getPageTitle(page) {
    if (this.adapter.type === 'playwright') {
      return await page.title();
    } else if (this.adapter.type === 'puppeteer') {
      return await page.title();
    } else if (this.adapter.type === 'selenium') {
      return await page.getTitle();
    }
  }

  async getViewport(page) {
    if (this.adapter.type === 'playwright') {
      return page.viewportSize();
    } else if (this.adapter.type === 'puppeteer') {
      return await page.viewport();
    } else if (this.adapter.type === 'selenium') {
      const size = await page.manage().window().getRect();
      return { width: size.width, height: size.height };
    }
  }

  async getInteractiveElements(page) {
    const selectors = [
      'button',
      'a[href]',
      'input:not([type="hidden"])',
      'select',
      'textarea',
      '[role="button"]',
      '[role="link"]',
      '[onclick]'
    ];

    const script = `
      (${selectors}) => {
        const elements = [];
        selectors.forEach(selector => {
          document.querySelectorAll(selector).forEach((el, index) => {
            const rect = el.getBoundingClientRect();
            elements.push({
              tag: el.tagName.toLowerCase(),
              type: el.type || null,
              text: el.textContent?.trim().substring(0, 100) || null,
              id: el.id || null,
              class: el.className || null,
              name: el.name || null,
              placeholder: el.placeholder || null,
              href: el.href || null,
              selector: selector,
              index: index,
              visible: rect.width > 0 && rect.height > 0 &&
                       rect.top >= 0 && rect.left >= 0 &&
                       rect.bottom <= window.innerHeight &&
                       rect.right <= window.innerWidth,
              position: {
                x: rect.left,
                y: rect.top,
                width: rect.width,
                height: rect.height
              }
            });
          });
        });
        return elements;
      }
    `;

    if (this.adapter.type === 'playwright') {
      return await page.evaluate(script, selectors);
    } else if (this.adapter.type === 'puppeteer') {
      return await page.evaluate(script, selectors);
    } else if (this.adapter.type === 'selenium') {
      return await page.executeScript(script, selectors);
    }
  }

  addToHistory(state) {
    this.stateHistory.push(state);
    if (this.stateHistory.length > this.maxHistorySize) {
      this.stateHistory.shift();
    }
  }

  getStateHistory() {
    return this.stateHistory;
  }

  getLastState() {
    return this.stateHistory[this.stateHistory.length - 1];
  }
}
```

### Action Executor for AI

```javascript
// AI Action Executor
class AIActionExecutor {
  constructor(adapter, elementController) {
    this.adapter = adapter;
    this.elementController = elementController;
    this.actionHistory = [];
  }

  async executeAction(action) {
    const startTime = Date.now();
    let result;

    try {
      switch (action.type) {
        case 'navigate':
          result = await this.navigate(action.url, action.options);
          break;
        case 'click':
          result = await this.click(action.selector, action.options);
          break;
        case 'fill':
          result = await this.fill(action.selector, action.text, action.options);
          break;
        case 'select':
          result = await this.select(action.selector, action.value);
          break;
        case 'scroll':
          result = await this.scroll(action.direction, action.amount);
          break;
        case 'screenshot':
          result = await this.screenshot(action.options);
          break;
        case 'wait':
          result = await this.wait(action.duration || action.condition);
          break;
        case 'extract':
          result = await this.extract(action.selector, action.attribute);
          break;
        default:
          throw new Error(`Unknown action type: ${action.type}`);
      }

      this.logAction(action, result, Date.now() - startTime, true);
      return { success: true, result };
    } catch (error) {
      this.logAction(action, error.message, Date.now() - startTime, false);
      return { success: false, error: error.message };
    }
  }

  async executeActionSequence(actions) {
    const results = [];
    for (const action of actions) {
      const result = await this.executeAction(action);
      results.push(result);
      if (!result.success && action.stopOnError !== false) {
        break;
      }
    }
    return results;
  }

  logAction(action, result, duration, success) {
    this.actionHistory.push({
      timestamp: Date.now(),
      action,
      result: success ? 'success' : 'error',
      duration,
      details: result
    });
  }

  getActionHistory() {
    return this.actionHistory;
  }
}
```

---

## Security Considerations

### Security Best Practices

```javascript
// Security Manager
class BrowserSecurityManager {
  constructor() {
    this.allowedDomains = [];
    this.blockedDomains = [];
    this.sandboxEnabled = true;
  }

  // Launch options for secure browsing
  getSecureLaunchOptions() {
    return {
      // Sandbox settings
      args: [
        '--sandbox',
        '--disable-setuid-sandbox',
        
        // Disable potentially dangerous features
        '--disable-plugins',
        '--disable-plugins-discovery',
        '--disable-background-networking',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-breakpad',
        '--disable-client-side-phishing-detection',
        '--disable-component-update',
        '--disable-default-apps',
        '--disable-dev-shm-usage',
        '--disable-extensions',
        '--disable-features=TranslateUI',
        '--disable-hang-monitor',
        '--disable-ipc-flooding-protection',
        '--disable-popup-blocking',
        '--disable-prompt-on-repost',
        '--disable-renderer-backgrounding',
        '--disable-sync',
        '--force-color-profile=srgb',
        '--metrics-recording-only',
        '--no-first-run',
        '--safebrowsing-disable-auto-update',
        '--enable-automation',
        '--password-store=basic',
        '--use-mock-keychain'
      ]
    };
  }

  // Content Security Policy helper
  validateUrl(url) {
    try {
      const parsed = new URL(url);
      
      // Check blocked domains
      if (this.blockedDomains.some(domain => parsed.hostname.includes(domain))) {
        throw new Error(`Domain ${parsed.hostname} is blocked`);
      }

      // Check allowed domains (if whitelist is configured)
      if (this.allowedDomains.length > 0) {
        if (!this.allowedDomains.some(domain => parsed.hostname.includes(domain))) {
          throw new Error(`Domain ${parsed.hostname} is not in allowed list`);
        }
      }

      // Only allow http and https
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        throw new Error(`Protocol ${parsed.protocol} is not allowed`);
      }

      return true;
    } catch (error) {
      throw new Error(`Invalid or blocked URL: ${error.message}`);
    }
  }

  // Sanitize user input for browser automation
  sanitizeInput(input) {
    if (typeof input !== 'string') return input;
    
    // Basic XSS prevention
    return input
      .replace(/[<>]/g, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+=/gi, '');
  }

  // Secure credential handling
  async handleCredentials(page, username, password, usernameSelector, passwordSelector) {
    // Use environment variables or secure vault in production
    const secureUsername = process.env[username] || username;
    const securePassword = process.env[password] || password;

    // Clear fields before entering
    await page.evaluate((sel) => {
      document.querySelector(sel).value = '';
    }, usernameSelector);
    
    await page.evaluate((sel) => {
      document.querySelector(sel).value = '';
    }, passwordSelector);

    // Type credentials with minimal exposure
    await page.type(usernameSelector, secureUsername);
    await page.type(passwordSelector, securePassword);
  }
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure
- [ ] Implement Puppeteer adapter
- [ ] Implement basic page navigation
- [ ] Implement element selection
- [ ] Add screenshot functionality

### Phase 2: Core Features (Weeks 3-4)
- [ ] Implement Playwright adapter
- [ ] Implement Selenium adapter
- [ ] Add unified abstraction layer
- [ ] Implement form filling
- [ ] Add PDF generation

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Implement network interception
- [ ] Add session persistence
- [ ] Implement mobile emulation
- [ ] Add responsive testing suite
- [ ] Implement tracing/recording

### Phase 4: AI Integration (Weeks 7-8)
- [ ] Implement state observer
- [ ] Add action executor
- [ ] Create semantic element finder
- [ ] Integrate with GPT-5.2
- [ ] Add error recovery

### Phase 5: Production (Weeks 9-10)
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Add comprehensive tests
- [ ] Documentation
- [ ] Deployment preparation

---

## Appendix A: Package Dependencies

```json
{
  "dependencies": {
    "puppeteer": "^21.0.0",
    "puppeteer-extra": "^3.3.6",
    "puppeteer-extra-plugin-stealth": "^2.11.2",
    "playwright": "^1.40.0",
    "selenium-webdriver": "^4.15.0",
    "chromedriver": "^119.0.0",
    "geckodriver": "^4.2.0",
    "pixelmatch": "^5.3.0",
    "pngjs": "^7.0.0",
    "crypto": "^1.0.1"
  },
  "devDependencies": {
    "jest": "^29.7.0",
    "@types/node": "^20.8.0"
  }
}
```

## Appendix B: Environment Variables

```bash
# Browser Configuration
BROWSER_DEFAULT=puppeteer
BROWSER_HEADLESS=true
BROWSER_SLOW_MO=0

# Session Security
SESSION_ENCRYPTION_KEY=your-secure-key-here
SESSION_STORAGE_DIR=./sessions/

# Screenshot/PDF Output
SCREENSHOT_DIR=./screenshots/
PDF_DIR=./pdfs/

# Security
ALLOWED_DOMAINS=example.com,trusted-site.com
BLOCKED_DOMAINS=malicious.com,phishing.net

# AI Integration
AI_MODEL=gpt-5.2
AI_MAX_TOKENS=4000
AI_TEMPERATURE=0.7
```

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Author:** OpenClaw Architecture Team
