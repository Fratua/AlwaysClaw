// src/extractors/JSONLDExtractor.ts
// Extract JSON-LD structured data from HTML

import { StructuredDataExtractor, StructuredDataResult } from './types';

interface JSONLDGraph {
  '@context'?: string | Record<string, any>;
  '@graph'?: any[];
  [key: string]: any;
}

export class JSONLDExtractor implements StructuredDataExtractor {
  private supportedTypes = new Set([
    'Article', 'NewsArticle', 'BlogPosting',
    'Product', 'Offer', 'AggregateRating', 'Review',
    'Organization', 'Person', 'LocalBusiness',
    'Event', 'JobPosting', 'Recipe',
    'VideoObject', 'ImageObject', 'AudioObject',
    'WebPage', 'WebSite', 'BreadcrumbList',
    'FAQPage', 'HowTo', 'Course'
  ]);

  extract(document: Document | string): StructuredDataResult {
    const html = typeof document === 'string' 
      ? document 
      : document.documentElement?.outerHTML || '';

    const scripts = this.extractJSONLDScripts(html);
    const results: any[] = [];
    const errors: string[] = [];

    for (const script of scripts) {
      try {
        const data = this.parseJSONLD(script);
        if (data) results.push(...this.flattenGraph(data));
      } catch (error) {
        errors.push(`JSON-LD parse error: ${error}`);
      }
    }

    return {
      format: 'json-ld',
      data: results,
      count: results.length,
      types: this.extractTypes(results),
      errors: errors.length > 0 ? errors : undefined
    };
  }

  private extractJSONLDScripts(html: string): string[] {
    const scripts: string[] = [];
    const regex = /<script[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
    let match;

    while ((match = regex.exec(html)) !== null) {
      scripts.push(match[1].trim());
    }

    return scripts;
  }

  private parseJSONLD(jsonString: string): JSONLDGraph | null {
    const cleaned = jsonString
      .replace(/^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/,(\s*[}\]])/g, '$1');

    try {
      return JSON.parse(cleaned);
    } catch (error) {
      const aggressiveClean = cleaned.replace(/\n/g, ' ').replace(/\s+/g, ' ');
      return JSON.parse(aggressiveClean);
    }
  }

  private flattenGraph(data: JSONLDGraph): any[] {
    if (data['@graph'] && Array.isArray(data['@graph'])) {
      return data['@graph'].map(item => ({
        ...item,
        '@context': data['@context']
      }));
    }
    if (Array.isArray(data)) return data;
    return [data];
  }

  private extractTypes(data: any[]): string[] {
    const types = new Set<string>();
    for (const item of data) {
      if (item['@type']) {
        if (Array.isArray(item['@type'])) {
          item['@type'].forEach((t: string) => types.add(t));
        } else {
          types.add(item['@type']);
        }
      }
    }
    return Array.from(types);
  }

  extractByType(document: Document | string, schemaType: string): any[] {
    const result = this.extract(document);
    return result.data.filter(item => {
      const types = Array.isArray(item['@type']) ? item['@type'] : [item['@type']];
      return types.some((t: string) => t.toLowerCase() === schemaType.toLowerCase());
    });
  }

  validate(data: any): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    if (!data['@type']) errors.push('Missing @type property');
    if (!data['@context'] && !this.hasValidContext(data)) errors.push('Missing or invalid @context');

    const type = Array.isArray(data['@type']) ? data['@type'][0] : data['@type'];
    switch (type) {
      case 'Product': if (!data.name) errors.push('Product missing name'); break;
      case 'Article': if (!data.headline) errors.push('Article missing headline'); break;
      case 'Organization': if (!data.name) errors.push('Organization missing name'); break;
    }

    return { valid: errors.length === 0, errors };
  }

  private hasValidContext(data: any): boolean {
    return !!(data['@id'] || data.url || (data.mainEntityOfPage && data.mainEntityOfPage['@type']));
  }
}
