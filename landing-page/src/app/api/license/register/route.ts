/**
 * Community License Registration API
 * 
 * Generates free Community license keys for registered users.
 * Tracks all registrations for analytics.
 */

import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { Resend } from 'resend';

// Generate license key (same algorithm as webhook)
function generateLicenseKey(
  company: string,
  tier: string,
  expiresDate: string,
  email: string
): string {
  const payload = {
    company,
    tier,
    expires: expiresDate,
    email,
    issued: new Date().toISOString().split('T')[0]
  };
  
  const payloadJson = JSON.stringify(payload);
  const payloadB64 = Buffer.from(payloadJson).toString('base64');
  
  const secret = process.env.ADAPTIVE_K_SECRET || 'VERTEX_ADAPTIVE_K_2026';
  const signature = crypto
    .createHash('sha256')
    .update(payloadB64 + secret)
    .digest('hex')
    .substring(0, 16);
  
  return `${payloadB64}.${signature}`;
}

// Get expiration date (1 year for community)
function getExpirationDate(): string {
  const date = new Date();
  date.setFullYear(date.getFullYear() + 1);
  return date.toISOString().split('T')[0];
}

// Send welcome email with license key
async function sendWelcomeEmail(
  email: string,
  company: string,
  licenseKey: string
): Promise<boolean> {
  const resend = new Resend(process.env.RESEND_API_KEY);
  const fromEmail = process.env.RESEND_FROM_EMAIL || 'Vertex Data <onboarding@resend.dev>';
  
  try {
    const { error } = await resend.emails.send({
      from: fromEmail,
      to: email,
      subject: 'Welcome to Adaptive-K - Your License Key',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #6366f1;">Welcome to Adaptive-K!</h2>
          
          <p>Hello <strong>${company}</strong>,</p>
          
          <p>Thank you for registering for Adaptive-K Community Edition!</p>
          
          <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <h3 style="margin-top: 0; color: #374151;">Your License Key:</h3>
            <code style="background: #1f2937; color: #10b981; padding: 12px; display: block; border-radius: 4px; word-break: break-all; font-size: 11px;">${licenseKey}</code>
          </div>
          
          <h3>Quick Start:</h3>
          <pre style="background: #1f2937; color: #d1d5db; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 12px;">pip install adaptive-k-routing

export ADAPTIVE_K_LICENSE="${licenseKey}"

python -c "from adaptive_k import AdaptiveKRouter; print('Ready!')"</pre>
          
          <h3 style="margin-top: 24px;">Community Edition Includes:</h3>
          <ul style="color: #4b5563;">
            <li>✅ Base adaptive routing</li>
            <li>✅ Calibration tools</li>
            <li>✅ CLI interface</li>
            <li>✅ Basic statistics</li>
          </ul>
          
          <div style="background: #fef3c7; border: 1px solid #f59e0b; padding: 16px; border-radius: 8px; margin: 20px 0;">
            <strong style="color: #92400e;">Need More?</strong>
            <p style="color: #78350f; margin: 8px 0 0 0;">
              Upgrade to Professional for CUDA kernels, vLLM integration, and priority support.
              <a href="https://adaptive-k.vertexdata.it/#pricing" style="color: #6366f1;">View Plans →</a>
            </p>
          </div>
          
          <hr style="margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;">
          
          <p style="color: #6b7280; font-size: 14px;">
            Questions? Contact <a href="mailto:amministrazione@vertexdata.it">amministrazione@vertexdata.it</a>
          </p>
        </div>
      `,
    });
    
    return !error;
  } catch {
    return false;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, company, useCase, acceptTerms } = body;
    
    // Validate required fields
    if (!email || !company) {
      return NextResponse.json({
        success: false,
        error: 'Email and company name are required'
      }, { status: 400 });
    }
    
    if (!acceptTerms) {
      return NextResponse.json({
        success: false,
        error: 'You must accept the terms of service'
      }, { status: 400 });
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json({
        success: false,
        error: 'Invalid email format'
      }, { status: 400 });
    }
    
    // Generate community license
    const expiresDate = getExpirationDate();
    const licenseKey = generateLicenseKey(company, 'community', expiresDate, email);
    
    // Log registration (in production, save to database)
    console.log(`New registration: ${email} (${company}) - Use case: ${useCase || 'not specified'}`);
    
    // Send welcome email
    await sendWelcomeEmail(email, company, licenseKey);
    
    // Also notify admin
    const resend = new Resend(process.env.RESEND_API_KEY);
    await resend.emails.send({
      from: process.env.RESEND_FROM_EMAIL || 'Vertex Data <onboarding@resend.dev>',
      to: 'amministrazione@vertexdata.it',
      subject: `New Adaptive-K Registration: ${company}`,
      html: `
        <h3>New Community Registration</h3>
        <ul>
          <li><strong>Email:</strong> ${email}</li>
          <li><strong>Company:</strong> ${company}</li>
          <li><strong>Use Case:</strong> ${useCase || 'Not specified'}</li>
          <li><strong>Date:</strong> ${new Date().toISOString()}</li>
        </ul>
      `
    });
    
    return NextResponse.json({
      success: true,
      licenseKey,
      tier: 'community',
      expires: expiresDate,
      message: 'Registration successful! Check your email for the license key.'
    });
    
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json({
      success: false,
      error: 'Registration failed. Please try again.'
    }, { status: 500 });
  }
}
