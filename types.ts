// Type Definitions for Web Scraping System
// OpenClaw Windows 10 AI Agent Framework

// Parser Types
export interface ParserInterface {
  load(html: string, url?: string): Promise<void>;
  extract(config: ExtractConfig): ParseResult;
}

export interface ExtractConfig {
  selectors: Record<string, string>;
  attributes?: Record<string, string[]>;
  multiple?: string[];
}

export interface ParseResult {
  data: Record<string, any>;
  metadata: {
    parser: string;
    timestamp: string;
    selectorsUsed: Array<{ key: string; selector: string; count: number }>;
  };
}

// Renderer Types
export interface RenderOptions {
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle';
  timeout?: number;
  waitForSelector?: string;
  selectorTimeout?: number;
  waitForFunction?: string;
  functionTimeout?: number;
  executeScript?: string;
  scrollToBottom?: boolean;
  scrollDelay?: number;
  blockResources?: string[];
  headers?: Record<string, string>;
  screenshot?: {
    fullPage?: boolean;
    type?: 'png' | 'jpeg';
  };
  cleanContent?: boolean;
}

export interface RenderResult {
  url: string;
  status: number;
  headers: Record<string, string>;
  content: string;
  text: string;
  title: string;
  screenshot: Buffer | null;
  scriptResult: any;
  metrics: {
    loadTime: number;
    domContentLoaded?: number;
    loadComplete?: number;
  };
}

// Extractor Types
export interface SelectorEngine {
  extract(document: any, domain: string): ExtractionResult;
}

export interface ExtractionRule {
  name: string;
  type: string;
}

export interface ExtractionResult {
  data: Record<string, any>;
  confidence: number;
  errors?: string[];
}

// Structured Data Types
export interface StructuredDataExtractor {
  extract(document: Document | string): StructuredDataResult;
}

export interface StructuredDataResult {
  format: string;
  data: any[];
  count: number;
  types: string[];
  errors?: string[];
}

// Table Types
export interface TableStructure {
  headers: string[];
  rows: (string | number | null)[][];
  caption?: string;
  metadata: {
    rowCount: number;
    colCount: number;
    hasHeader: boolean;
    confidence: number;
  };
}

export interface TableExtractionOptions {
  detectHeaders?: boolean;
  headerHeuristics?: 'first-row' | 'th-tags' | 'bold' | 'auto';
  cleanWhitespace?: boolean;
  normalizeNumbers?: boolean;
  extractLinks?: boolean;
  minRows?: number;
  minCols?: number;
}

// Pipeline Types
export interface PipelineConfig {
  name: string;
  urlPattern: RegExp;
  parser: 'static' | 'dynamic';
  extractors: string[];
  outputFormat: 'json' | 'csv' | 'xml';
  postProcessors?: string[];
}

export interface PipelineContext {
  url: string;
  domain: string;
  html?: string;
  document?: Document;
  extractedData?: any;
  metadata: {
    startTime: number;
    endTime?: number;
    parserUsed?: string;
    errors: string[];
  };
}

// Rate Limiter Types
export interface RateLimitConfig {
  requestsPerSecond: number;
  requestsPerMinute: number;
  requestsPerHour: number;
  requestsPerDay: number;
  concurrentRequests: number;
  domainSpecific?: Record<string, Partial<RateLimitConfig>>;
}

// Deduplication Types
export interface DedupConfig {
  method: 'hash' | 'simhash' | 'minhash' | 'hybrid';
  threshold: number;
  storage: 'memory' | 'redis' | 'sqlite';
  ttl?: number;
}

export interface ContentFingerprint {
  hash: string;
  simhash?: string;
  size: number;
  timestamp: number;
  url: string;
}

// Evasion Types
export interface Fingerprint {
  userAgent: string;
  viewport: { width: number; height: number };
  deviceScaleFactor: number;
  locale: string;
  timezoneId: string;
  colorDepth: number;
  hardwareConcurrency: number;
  deviceMemory: number;
  platform: string;
  vendor: string;
}

export interface CaptchaDetection {
  detected: boolean;
  type: 'recaptcha' | 'hcaptcha' | 'image' | 'text' | 'none';
  confidence: number;
  selectors: string[];
}

// Browser Pool Types
export interface PoolConfig {
  maxBrowsers: number;
  maxPagesPerBrowser: number;
  headless: boolean;
  stealth: boolean;
  proxy?: string;
  userAgent?: string;
}

export interface PooledPage {
  page: import('playwright').Page;
  context: import('playwright').BrowserContext;
  browser: import('playwright').Browser;
  inUse: boolean;
  lastUsed: Date;
}

// Rate Limiter Request Record
export interface RequestRecord {
  timestamp: number;
  domain: string;
  weight: number;
}

// Robots.txt Types
export interface RobotsRule {
  userAgent: string;
  allow: string[];
  disallow: string[];
  crawlDelay?: number;
  sitemap?: string[];
}

export interface ParsedRobots {
  rules: RobotsRule[];
  sitemaps: string[];
  host?: string;
}
