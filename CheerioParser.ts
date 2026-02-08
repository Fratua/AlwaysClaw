// src/parsers/CheerioParser.ts
// Fast HTML parser using Cheerio for static content

import * as cheerio from 'cheerio';
import { ParserInterface, ParseResult, ExtractConfig } from './types';

export class CheerioParser implements ParserInterface {
  private $: cheerio.CheerioAPI | null = null;
  private html: string = '';

  async load(html: string): Promise<void> {
    this.html = html;
    this.$ = cheerio.load(html, {
      xml: {
        xmlMode: false,
        decodeEntities: true,
      }
    });
  }

  extract(config: ExtractConfig): ParseResult {
    if (!this.$) throw new Error('Parser not loaded');

    const result: ParseResult = {
      data: {},
      metadata: {
        parser: 'cheerio',
        timestamp: new Date().toISOString(),
        selectorsUsed: []
      }
    };

    for (const [key, selector] of Object.entries(config.selectors)) {
      const elements = this.$(selector);
      
      if (config.multiple?.includes(key)) {
        result.data[key] = elements.map((_: number, el: any) =>
          this.extractElement(this.$!(el), config.attributes?.[key])
        ).get();
      } else {
        result.data[key] = this.extractElement(elements.first(), config.attributes?.[key]);
      }
      
      result.metadata.selectorsUsed.push({ key, selector, count: elements.length });
    }

    return result;
  }

  private extractElement(
    element: cheerio.Cheerio<any>,
    attributes?: string[]
  ): Record<string, any> {
    const result: Record<string, any> = {
      text: element.text().trim(),
      html: element.html() || ''
    };

    if (attributes) {
      attributes.forEach(attr => {
        result[attr] = element.attr(attr);
      });
    }

    return result;
  }

  xpath(expression: string): cheerio.Cheerio<any> {
    if (!this.$) throw new Error('Parser not loaded');
    const cssSelector = this.xpathToCss(expression);
    return this.$(cssSelector);
  }

  private xpathToCss(xpath: string): string {
    return xpath
      .replace(/\/\//g, '')
      .replace(/\[@([^=]+)='([^']+)'\]/g, '[$1="$2"]')
      .replace(/\[@class='([^']+)'\]/g, '.$1')
      .replace(/\[@id='([^']+)'\]/g, '#$1');
  }
}
