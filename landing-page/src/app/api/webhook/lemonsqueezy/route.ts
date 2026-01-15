/**
 * LemonSqueezy Webhook Handler
 * 
 * This API route receives webhooks from LemonSqueezy when:
 * - A purchase is completed (order_created)
 * - A subscription is renewed (subscription_payment_success)
 * - A subscription is cancelled (subscription_cancelled)
 * 
 * It generates license keys and sends them to customers.
 * 
 * Setup:
 * 1. In LemonSqueezy Dashboard → Settings → Webhooks
 * 2. Add webhook URL: https://adaptive-k.vertexdata.it/api/webhook/lemonsqueezy
 * 3. Select events: order_created, subscription_payment_success
 * 4. Copy the signing secret and add to .env.local as LEMONSQUEEZY_WEBHOOK_SECRET
 * 5. Add RESEND_API_KEY to environment variables
 */

import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { Resend } from 'resend';

// License key generation (same algorithm as Python SDK)
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
  
  // Sign with secret (must match Python SDK)
  const secret = process.env.ADAPTIVE_K_SECRET || 'VERTEX_ADAPTIVE_K_2026';
  const signature = crypto
    .createHash('sha256')
    .update(payloadB64 + secret)
    .digest('hex')
    .substring(0, 16);
  
  return `${payloadB64}.${signature}`;
}

// Calculate expiration date (1 year from now)
function getExpirationDate(): string {
  const date = new Date();
  date.setFullYear(date.getFullYear() + 1);
  return date.toISOString().split('T')[0];
}

// Verify webhook signature
function verifyWebhookSignature(
  payload: string,
  signature: string,
  secret: string
): boolean {
  const hmac = crypto.createHmac('sha256', secret);
  hmac.update(payload);
  const expectedSignature = hmac.digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature)
  );
}

// Send license key via email using Resend
async function sendLicenseEmail(
  email: string,
  company: string,
  licenseKey: string,
  tier: string
): Promise<{ success: boolean; error?: string }> {
  const resend = new Resend(process.env.RESEND_API_KEY);
  
  // Use verified domain or fallback to Resend's test domain
  const fromEmail = process.env.RESEND_FROM_EMAIL || 'Vertex Data <onboarding@resend.dev>';
  
  try {
    const { data, error } = await resend.emails.send({
      from: fromEmail,
      to: email,
      subject: `Your Adaptive-K ${tier} License Key`,
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #6366f1;">🎉 Thank you for your purchase!</h2>
          
          <p>Hello <strong>${company}</strong>,</p>
          
          <p>Thank you for purchasing <strong>Adaptive-K ${tier} License</strong>!</p>
          
          <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <h3 style="margin-top: 0; color: #374151;">Your License Key:</h3>
            <code style="background: #1f2937; color: #10b981; padding: 12px; display: block; border-radius: 4px; word-break: break-all; font-size: 12px;">${licenseKey}</code>
          </div>
          
          <h3>Quick Start:</h3>
          <ol style="line-height: 1.8;">
            <li>Set environment variable:<br/>
              <code style="background: #e5e7eb; padding: 4px 8px; border-radius: 4px;">export ADAPTIVE_K_LICENSE="${licenseKey}"</code>
            </li>
            <li>Or pass directly to router:<br/>
              <code style="background: #e5e7eb; padding: 4px 8px; border-radius: 4px;">AdaptiveKRouter(license_key="...")</code>
            </li>
          </ol>
          
          <h3>Verify Installation:</h3>
          <pre style="background: #1f2937; color: #d1d5db; padding: 12px; border-radius: 4px; overflow-x: auto;">pip install adaptive-k-routing
adaptive-k license --validate</pre>
          
          <p style="margin-top: 30px;">
            <a href="https://adaptive-k.vertexdata.it/portal" style="background: #6366f1; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">Access Customer Portal</a>
          </p>
          
          <hr style="margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;">
          
          <p style="color: #6b7280; font-size: 14px;">
            Need help? Contact us at <a href="mailto:amministrazione@vertexdata.it">amministrazione@vertexdata.it</a>
          </p>
          
          <p style="color: #6b7280; font-size: 14px;">
            Best regards,<br/>
            <strong>Vertex Data Team</strong>
          </p>
        </div>
      `,
    });
    
    if (error) {
      console.error('Resend error:', error);
      return { success: false, error: error.message };
    }
    
    console.log('License email sent successfully:', data?.id);
    return { success: true };
  } catch (err) {
    console.error('Failed to send license email:', err);
    return { success: false, error: String(err) };
  }
}

export async function POST(request: NextRequest) {
  try {
    const payload = await request.text();
    const signature = request.headers.get('x-signature') || '';
    const webhookSecret = process.env.LEMONSQUEEZY_WEBHOOK_SECRET;
    
    // Verify signature in production (skip if no signature provided - for testing)
    if (webhookSecret && signature && !verifyWebhookSignature(payload, signature, webhookSecret)) {
      console.error('Invalid webhook signature');
      return NextResponse.json({ error: 'Invalid signature' }, { status: 401 });
    }
    
    const event = JSON.parse(payload);
    const eventName = event.meta?.event_name;
    
    console.log(`Received LemonSqueezy webhook: ${eventName}`);
    
    // Handle order creation
    if (eventName === 'order_created') {
      const order = event.data?.attributes;
      const customerEmail = order?.user_email;
      const customerName = order?.user_name || order?.billing_address?.name || 'Customer';
      const productName = order?.first_order_item?.product_name || '';
      
      if (!customerEmail) {
        console.error('No customer email in webhook payload');
        return NextResponse.json({ error: 'No customer email' }, { status: 400 });
      }
      
      // Determine tier from product name
      let tier = 'professional';
      if (productName.toLowerCase().includes('enterprise')) {
        tier = 'enterprise';
      }
      
      // Generate license key
      const expiresDate = getExpirationDate();
      const licenseKey = generateLicenseKey(
        customerName,
        tier,
        expiresDate,
        customerEmail
      );
      
      console.log(`Generated ${tier} license for ${customerEmail}`);
      
      // Send license email
      const emailResult = await sendLicenseEmail(customerEmail, customerName, licenseKey, tier);
      
      if (!emailResult.success) {
        console.error('Failed to send email:', emailResult.error);
        return NextResponse.json({ 
          success: false, 
          error: `Email failed: ${emailResult.error}`,
          licenseKey // Return key anyway so it's not lost
        }, { status: 500 });
      }
      
      // TODO: Store in database for portal access
      // await db.licenses.create({ email: customerEmail, key: licenseKey, tier, expires: expiresDate });
      
      return NextResponse.json({ 
        success: true, 
        message: `License generated and emailed to ${customerEmail}`,
        licenseKey
      });
    }
    
    // Handle subscription renewal
    if (eventName === 'subscription_payment_success') {
      const subscription = event.data?.attributes;
      const customerEmail = subscription?.user_email;
      const customerName = subscription?.user_name || 'Customer';
      
      // Generate new license key with extended expiration
      const expiresDate = getExpirationDate();
      const licenseKey = generateLicenseKey(
        customerName,
        'professional', // Subscriptions are typically professional tier
        expiresDate,
        customerEmail
      );
      
      console.log(`Renewed license for ${customerEmail}`);
      
      // Send renewal email
      await sendLicenseEmail(customerEmail, customerName, licenseKey, 'Professional (Renewed)');
      
      return NextResponse.json({ 
        success: true, 
        message: `License renewed for ${customerEmail}` 
      });
    }
    
    // Handle subscription cancellation
    if (eventName === 'subscription_cancelled') {
      const subscription = event.data?.attributes;
      const customerEmail = subscription?.user_email;
      
      console.log(`Subscription cancelled for ${customerEmail}`);
      
      // TODO: Mark license as cancelled in database
      // License will expire naturally at end of paid period
      
      return NextResponse.json({ 
        success: true, 
        message: `Subscription cancelled for ${customerEmail}` 
      });
    }
    
    // Unknown event
    return NextResponse.json({ success: true, message: `Ignored event: ${eventName}` });
    
  } catch (error) {
    console.error('Webhook error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Handle GET for testing
export async function GET() {
  return NextResponse.json({ 
    status: 'ok',
    message: 'LemonSqueezy webhook endpoint',
    docs: 'POST to this endpoint with LemonSqueezy webhook payload'
  });
}
