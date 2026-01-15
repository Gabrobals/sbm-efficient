# LemonSqueezy Configuration

## Setup Instructions

### 1. Create Products in LemonSqueezy Dashboard

Go to: https://app.lemonsqueezy.com/products

#### Product 1: Adaptive-K Professional
- **Name:** Adaptive-K Professional License
- **Description:** 1-year commercial license for Adaptive-K SDK
- **Price:** €2,500/year (recurring subscription)
- **Files:** None (license key sent via email webhook)
- **License Keys:** Enable built-in license keys
- **Variant ID:** Copy after creation → Update PRODUCTS.PROFESSIONAL in LemonSqueezy.tsx

#### Product 2: Adaptive-K Enterprise
- **Name:** Adaptive-K Enterprise License  
- **Description:** Custom enterprise license with dedicated support
- **Price:** €8,000/year (or "Contact Us" for custom)
- **Variant ID:** Copy after creation → Update PRODUCTS.ENTERPRISE in LemonSqueezy.tsx

### 2. Configure Webhooks

Go to: https://app.lemonsqueezy.com/settings/webhooks

- **URL:** https://adaptive-k.vertexdata.it/api/webhook/lemonsqueezy
- **Events to select:**
  - order_created
  - subscription_payment_success
  - subscription_cancelled
- **Copy Signing Secret** → Add to Vercel environment variables

### 3. Environment Variables

Add to Vercel (https://vercel.com/vertex-data/adaptive-k/settings/environment-variables):

```
LEMONSQUEEZY_WEBHOOK_SECRET=your_signing_secret_here
ADAPTIVE_K_SECRET=VERTEX_ADAPTIVE_K_2026
WEB3FORMS_KEY=709b0b2d-560d-457c-8694-d83ba2bb0905
```

### 4. Update Code

After creating products, update these files:

#### landing-page/src/components/LemonSqueezy.tsx
```typescript
export const PRODUCTS = {
  PROFESSIONAL: 'xxxxxx', // Replace with actual variant ID
  ENTERPRISE: 'xxxxxx',   // Replace with actual variant ID
} as const;
```

### 5. Test Flow

1. Use LemonSqueezy test mode
2. Make a test purchase
3. Verify webhook received (check Vercel logs)
4. Verify license key email received
5. Validate license key with SDK: `adaptive-k license --key <received_key>`

### 6. Go Live

1. Complete LemonSqueezy verification (ID, bank account)
2. Switch to live mode
3. Update webhook URL if using different domain
4. Announce on website/social media

## Pricing Structure

| Tier | Price | Fee (5%) | Net Revenue |
|------|-------|----------|-------------|
| Professional | €2,500/year | €125 | €2,375 |
| Enterprise | €8,000/year | €400 | €7,600 |

## Customer Flow

```
Customer → Pricing Page → LemonSqueezy Checkout → Payment
                                    ↓
                            Webhook Triggered
                                    ↓
                          License Key Generated
                                    ↓
                           Email Sent to Customer
                                    ↓
                       Customer activates in SDK
```

## Support

- Customer portal: https://adaptive-k.vertexdata.it/portal
- LemonSqueezy handles: Payment, refunds, invoices, VAT
- We handle: License key support, technical questions
