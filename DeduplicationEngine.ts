// src/dedup/DeduplicationEngine.ts
// Content deduplication using hash, simhash, and minhash

import { createHash } from 'crypto';
import { DedupConfig, ContentFingerprint } from './types';

export class DeduplicationEngine {
  private config: DedupConfig;
  private memoryStore: Map<string, ContentFingerprint> = new Map();
  private simhashStore: Map<string, Set<string>> = new Map();

  constructor(config: Partial<DedupConfig> = {}) {
    this.config = {
      method: 'hybrid',
      threshold: 0.85,
      storage: 'memory',
      ttl: 86400000,
      ...config
    };
    if (this.config.ttl) setInterval(() => this.cleanExpired(), this.config.ttl);
  }

  async isDuplicate(content: string, url: string): Promise<boolean> {
    const fingerprint = this.createFingerprint(content, url);

    switch (this.config.method) {
      case 'hash': return this.checkHashDuplicate(fingerprint);
      case 'simhash': return this.checkSimhashDuplicate(fingerprint);
      case 'minhash': return this.checkMinhashDuplicate(content, fingerprint);
      case 'hybrid': default: return this.checkHybridDuplicate(content, fingerprint);
    }
  }

  async add(content: string, url: string): Promise<void> {
    const fingerprint = this.createFingerprint(content, url);
    this.memoryStore.set(fingerprint.hash, fingerprint);

    if (fingerprint.simhash) {
      const bucket = this.getSimhashBucket(fingerprint.simhash);
      if (!this.simhashStore.has(bucket)) this.simhashStore.set(bucket, new Set());
      this.simhashStore.get(bucket)!.add(fingerprint.simhash);
    }
  }

  private createFingerprint(content: string, url: string): ContentFingerprint {
    const normalized = this.normalizeContent(content);
    return {
      hash: this.computeHash(normalized),
      simhash: this.computeSimhash(normalized),
      size: normalized.length,
      timestamp: Date.now(),
      url
    };
  }

  private normalizeContent(content: string): string {
    return content.toLowerCase().replace(/\s+/g, ' ').replace(/[^\w\s]/g, '').trim();
  }

  private computeHash(content: string): string {
    return createHash('sha256').update(content).digest('hex');
  }

  private computeSimhash(content: string): string {
    const features = this.extractFeatures(content);
    const vector = new Array(64).fill(0);

    for (const feature of features) {
      const hash = parseInt(createHash('md5').update(feature).digest('hex').slice(0, 16), 16);
      for (let i = 0; i < 64; i++) {
        const bit = (hash >> i) & 1;
        vector[i] += bit === 1 ? 1 : -1;
      }
    }

    return vector.map(v => v >= 0 ? '1' : '0').join('');
  }

  private extractFeatures(content: string, n: number = 3): string[] {
    const words = content.split(' ');
    const features: string[] = [];
    for (let i = 0; i <= words.length - n; i++) {
      features.push(words.slice(i, i + n).join(' '));
    }
    return features;
  }

  private checkHashDuplicate(fingerprint: ContentFingerprint): boolean {
    return this.memoryStore.has(fingerprint.hash);
  }

  private checkSimhashDuplicate(fingerprint: ContentFingerprint): boolean {
    if (!fingerprint.simhash) return false;
    const bucket = this.getSimhashBucket(fingerprint.simhash);
    const candidates = this.simhashStore.get(bucket);
    if (!candidates) return false;

    for (const candidate of candidates) {
      if (this.simhashSimilarity(fingerprint.simhash, candidate) >= this.config.threshold) {
        return true;
      }
    }
    return false;
  }

  private checkMinhashDuplicate(content: string, fingerprint: ContentFingerprint): boolean {
    const features = this.extractFeatures(content);
    const signatures = this.computeMinhashSignatures(features);

    for (const [hash, existing] of this.memoryStore) {
      if (Math.abs(existing.size - fingerprint.size) > fingerprint.size * 0.2) continue;
      const existingFeatures = this.extractFeaturesFromHash(hash);
      const existingSignatures = this.computeMinhashSignatures(existingFeatures);
      if (this.jaccardSimilarity(signatures, existingSignatures) >= this.config.threshold) return true;
    }
    return false;
  }

  private checkHybridDuplicate(content: string, fingerprint: ContentFingerprint): boolean {
    if (this.checkHashDuplicate(fingerprint)) return true;
    if (this.checkSimhashDuplicate(fingerprint)) return true;
    return this.checkMinhashDuplicate(content, fingerprint);
  }

  private getSimhashBucket(simhash: string): string {
    return simhash.slice(0, 16);
  }

  private simhashSimilarity(hash1: string, hash2: string): number {
    let distance = 0;
    for (let i = 0; i < hash1.length; i++) if (hash1[i] !== hash2[i]) distance++;
    return 1 - (distance / hash1.length);
  }

  private computeMinhashSignatures(features: string[]): number[] {
    const numHashes = 128;
    const signatures: number[] = [];

    for (let i = 0; i < numHashes; i++) {
      let minHash = Infinity;
      for (const feature of features) {
        const hash = parseInt(createHash('md5').update(`${i}:${feature}`).digest('hex').slice(0, 8), 16);
        minHash = Math.min(minHash, hash);
      }
      signatures.push(minHash);
    }
    return signatures;
  }

  private jaccardSimilarity(a: number[], b: number[]): number {
    let intersection = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) intersection++;
    return intersection / a.length;
  }

  private extractFeaturesFromHash(hash: string): string[] {
    return [];
  }

  private cleanExpired(): void {
    if (!this.config.ttl) return;
    const cutoff = Date.now() - this.config.ttl;
    for (const [hash, fingerprint] of this.memoryStore) {
      if (fingerprint.timestamp < cutoff) this.memoryStore.delete(hash);
    }
  }

  getStats() {
    const memory = JSON.stringify([...this.memoryStore]).length;
    return {
      totalFingerprints: this.memoryStore.size,
      simhashBuckets: this.simhashStore.size,
      memoryUsage: `${(memory / 1024 / 1024).toFixed(2)} MB`
    };
  }
}
