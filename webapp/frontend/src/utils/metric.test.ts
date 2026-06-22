import { describe, it, expect } from 'vitest';
import { formatMetric, formatSignedMetric } from './metric';

describe('metric formatting utilities', () => {
  describe('formatMetric', () => {
    it('should format normal numbers to 4 decimal places by default', () => {
      expect(formatMetric(1.23456)).toBe('1.2346');
      expect(formatMetric(0)).toBe('0.0000');
    });

    it('should accept custom decimal places config', () => {
      expect(formatMetric(1.23456, 2)).toBe('1.23');
      expect(formatMetric(10.5, 0)).toBe('11');
    });

    it('should return "-" if value is null, undefined, or NaN', () => {
      expect(formatMetric(null)).toBe('-');
      expect(formatMetric(undefined)).toBe('-');
      expect(formatMetric(NaN)).toBe('-');
    });
  });

  describe('formatSignedMetric', () => {
    it('should prefix positive numbers and zero with + sign', () => {
      expect(formatSignedMetric(1.23456)).toBe('+1.2346');
      expect(formatSignedMetric(0)).toBe('+0.0000');
    });

    it('should show - sign for negative numbers', () => {
      expect(formatSignedMetric(-1.23456)).toBe('-1.2346');
    });

    it('should return "-" if value is null, undefined, or NaN', () => {
      expect(formatSignedMetric(null)).toBe('-');
      expect(formatSignedMetric(undefined)).toBe('-');
      expect(formatSignedMetric(NaN)).toBe('-');
    });
  });
});
