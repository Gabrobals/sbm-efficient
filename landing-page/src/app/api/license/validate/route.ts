/**
 * License Validation API
 * 
 * Validates license keys and tracks usage.
 * All tiers (including Community) require a valid key.
 */

import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';

// In-memory store (in production, use database)
const usageLog: Map<string, { count: number; lastUsed: string; tier: string }> = new Map();

function decodePayload(licenseKey: string): { valid: boolean; payload?: any; error?: string } {
  try {
    const [payloadB64, signature] = licenseKey.split('.');
    if (!payloadB64 || !signature) {
      return { valid: false, error: 'Invalid key format' };
    }
    
    // Decode payload
    const payloadJson = Buffer.from(payloadB64, 'base64').toString('utf-8');
    const payload = JSON.parse(payloadJson);
    
    // Verify signature
    const secret = process.env.ADAPTIVE_K_SECRET || 'VERTEX_ADAPTIVE_K_2026';
    const expectedSig = crypto
      .createHash('sha256')
      .update(payloadB64 + secret)
      .digest('hex')
      .substring(0, 16);
    
    if (signature !== expectedSig) {
      return { valid: false, error: 'Invalid signature' };
    }
    
    // Check expiration
    const expiresDate = new Date(payload.expires);
    if (new Date() > expiresDate) {
      return { valid: false, error: 'License expired', payload };
    }
    
    return { valid: true, payload };
  } catch (e) {
    return { valid: false, error: 'Decode error' };
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { license_key, client_info } = body;
    
    if (!license_key) {
      return NextResponse.json({
        valid: false,
        error: 'License key required. Register at https://adaptive-k.vertexdata.it/register',
        register_url: 'https://adaptive-k.vertexdata.it/register'
      }, { status: 401 });
    }
    
    const result = decodePayload(license_key);
    
    if (!result.valid) {
      return NextResponse.json({
        valid: false,
        error: result.error,
        expired: result.error === 'License expired',
        renew_url: 'https://adaptive-k.vertexdata.it'
      }, { status: 401 });
    }
    
    const payload = result.payload;
    
    // Log usage (for analytics)
    const email = payload.email || 'unknown';
    const existing = usageLog.get(email) || { count: 0, lastUsed: '', tier: payload.tier };
    usageLog.set(email, {
      count: existing.count + 1,
      lastUsed: new Date().toISOString(),
      tier: payload.tier
    });
    
    // Log to console for monitoring (in production, send to analytics service)
    console.log(`License validated: ${email} (${payload.tier}) - ${client_info?.hostname || 'unknown host'}`);
    
    return NextResponse.json({
      valid: true,
      tier: payload.tier,
      company: payload.company,
      expires: payload.expires,
      features: getFeatures(payload.tier),
      message: `${payload.tier.charAt(0).toUpperCase() + payload.tier.slice(1)} license active`
    });
    
  } catch (error) {
    console.error('Validation error:', error);
    return NextResponse.json({
      valid: false,
      error: 'Validation failed'
    }, { status: 500 });
  }
}

function getFeatures(tier: string): string[] {
  const features: Record<string, string[]> = {
    community: ['base_routing', 'calibration', 'cli', 'basic_stats'],
    professional: ['base_routing', 'calibration', 'cli', 'basic_stats', 'cuda_kernels', 'vllm_integration', 'tensorrt_integration', 'priority_support'],
    enterprise: ['base_routing', 'calibration', 'cli', 'basic_stats', 'cuda_kernels', 'vllm_integration', 'tensorrt_integration', 'priority_support', 'custom_optimization', 'sla_guarantee', 'redistribution_rights']
  };
  return features[tier] || features.community;
}

// GET for health check
export async function GET() {
  return NextResponse.json({
    status: 'ok',
    message: 'License validation API',
    usage_count: usageLog.size
  });
}
