// src/pipelines/PipelineOrchestrator.ts
// Main pipeline orchestrator for web scraping

import { EventEmitter } from 'events';
import { BrowserPool } from './BrowserPool';
import { RateLimiter } from './RateLimiter';
import { DeduplicationEngine } from './DeduplicationEngine';
import { CheerioParser } from './CheerioParser';
import { JSONLDExtractor } from './JSONLDExtractor';
import { PipelineConfig, PipelineContext } from './types';

export class PipelineOrchestrator extends EventEmitter {
  private browserPool: BrowserPool;
  private rateLimiter: RateLimiter;
  private dedupEngine: DeduplicationEngine;
  private pipelines: Map<string, PipelineConfig> = new Map();

  constructor(browserPool: BrowserPool, rateLimiter: RateLimiter, dedupEngine: DeduplicationEngine) {
    super();
    this.browserPool = browserPool;
    this.rateLimiter = rateLimiter;
    this.dedupEngine = dedupEngine;
  }

  registerPipeline(config: PipelineConfig): void {
    this.pipelines.set(config.name, config);
    this.emit('pipelineRegistered', config);
  }

  async execute(url: string, pipelineName?: string): Promise<any> {
    const context: PipelineContext = {
      url,
      domain: new URL(url).hostname,
      metadata: { startTime: Date.now(), errors: [] }
    };

    try {
      const pipeline = pipelineName ? this.pipelines.get(pipelineName) : this.selectPipeline(url);
      if (!pipeline) throw new Error(`No pipeline found for URL: ${url}`);

      this.emit('pipelineStarted', { url, pipeline: pipeline.name });
      await this.rateLimiter.acquire(context.domain);

      const isDuplicate = await this.dedupEngine.isDuplicate(url, url);
      if (isDuplicate) {
        this.emit('duplicateDetected', { url });
        return { skipped: true, reason: 'duplicate' };
      }

      context.html = await this.fetchContent(url, pipeline.parser);
      context.metadata.parserUsed = pipeline.parser === 'dynamic' ? 'playwright' : 'cheerio';

      const parser = new CheerioParser();
      await parser.load(context.html);

      context.extractedData = await this.runExtractors(pipeline.extractors, context.html);

      if (pipeline.postProcessors) {
        context.extractedData = await this.runPostProcessors(pipeline.postProcessors, context.extractedData);
      }

      await this.dedupEngine.add(context.html, url);

      const output = this.formatOutput(context.extractedData, pipeline.outputFormat);
      context.metadata.endTime = Date.now();

      this.emit('pipelineCompleted', {
        url, pipeline: pipeline.name,
        duration: context.metadata.endTime - context.metadata.startTime
      });

      return { success: true, data: output, metadata: context.metadata };

    } catch (error) {
      context.metadata.errors.push(String(error));
      context.metadata.endTime = Date.now();
      this.emit('pipelineFailed', {
        url, error: String(error),
        duration: context.metadata.endTime - context.metadata.startTime
      });
      return { success: false, error: String(error), metadata: context.metadata };
    } finally {
      this.rateLimiter.release(context.domain);
    }
  }

  private selectPipeline(url: string): PipelineConfig | null {
    for (const pipeline of this.pipelines.values()) {
      if (pipeline.urlPattern.test(url)) return pipeline;
    }
    return null;
  }

  private async fetchContent(url: string, parser: 'static' | 'dynamic'): Promise<string> {
    if (parser === 'static') {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' }
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      return response.text();
    }
    // Dynamic rendering would use DynamicRenderer here
    throw new Error('Dynamic rendering not implemented in this example');
  }

  private async runExtractors(extractorNames: string[], html: string): Promise<any> {
    const results: any = {};
    for (const name of extractorNames) {
      switch (name) {
        case 'jsonld':
          results.jsonld = new JSONLDExtractor().extract(html);
          break;
        case 'meta':
          results.meta = this.extractMetaTags(html);
          break;
        case 'links':
          results.links = this.extractLinks(html);
          break;
        default:
          console.warn(`Unknown extractor: ${name}`);
      }
    }
    return results;
  }

  private extractMetaTags(html: string): Record<string, string> {
    const meta: Record<string, string> = {};
    const regex = /<meta[^>]*(?:name|property)=["']([^"']+)["'][^>]*content=["']([^"']*)["'][^>]*>/gi;
    let match;
    while ((match = regex.exec(html)) !== null) meta[match[1]] = match[2];
    return meta;
  }

  private extractLinks(html: string): string[] {
    const links: string[] = [];
    const regex = /<a[^>]*href=["']([^"']+)["'][^>]*>/gi;
    let match;
    while ((match = regex.exec(html)) !== null) links.push(match[1]);
    return [...new Set(links)];
  }

  private async runPostProcessors(processors: string[], data: any): Promise<any> {
    let result = data;
    for (const processor of processors) {
      switch (processor) {
        case 'clean': result = this.cleanData(result); break;
        case 'validate': result = this.validateData(result); break;
        case 'enrich': result = await this.enrichData(result); break;
        default: console.warn(`Unknown processor: ${processor}`);
      }
    }
    return result;
  }

  private cleanData(data: any): any {
    if (Array.isArray(data)) {
      return data.map(item => this.cleanData(item)).filter(item => item !== null && item !== '');
    }
    if (typeof data === 'object' && data !== null) {
      const cleaned: any = {};
      for (const [key, value] of Object.entries(data)) {
        const cleanedValue = this.cleanData(value);
        if (cleanedValue !== null && cleanedValue !== '') cleaned[key] = cleanedValue;
      }
      return cleaned;
    }
    return data;
  }

  private validateData(data: any): any {
    return data;
  }

  private async enrichData(data: any): Promise<any> {
    return data;
  }

  private formatOutput(data: any, format: string): any {
    switch (format) {
      case 'json': return data;
      case 'csv': return Array.isArray(data) ? this.toCSV(data) : data;
      case 'xml': return this.toXML(data);
      default: return data;
    }
  }

  private toCSV(data: any[]): string {
    if (data.length === 0) return '';
    const headers = Object.keys(data[0]);
    const rows = data.map(obj =>
      headers.map(h => {
        const val = obj[h];
        const str = String(val ?? '');
        return str.includes(',') || str.includes('"') ? `"${str.replace(/"/g, '""')}"` : str;
      }).join(',')
    );
    return [headers.join(','), ...rows].join('\n');
  }

  private toXML(data: any): string {
    const convert = (obj: any, name: string): string => {
      if (Array.isArray(obj)) return obj.map(item => convert(item, name)).join('');
      if (typeof obj === 'object' && obj !== null) {
        const children = Object.entries(obj).map(([key, value]) => convert(value, key)).join('');
        return `<${name}>${children}</${name}>`;
      }
      return `<${name}>${String(obj).replace(/[<>&]/g, c => c === '<' ? '&lt;' : c === '>' ? '&gt;' : '&amp;')}</${name}>`;
    };
    return convert(data, 'root');
  }
}
