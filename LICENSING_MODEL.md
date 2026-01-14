# Adaptive-K Licensing Model

## Overview

Modello di licensing a 3 tier per monetizzare l'SDK Adaptive-K mantenendo una versione open-source.

---

## 🆓 Tier 1: Community (Gratuito)

**Licenza:** Apache 2.0 (attuale)

**Include:**
- SDK completo su PyPI
- Routing Adaptive-K base
- Calibrazione threshold
- CLI tools
- Documentazione

**Limitazioni:**
- Nessun supporto garantito
- Solo GitHub Issues
- Attribution richiesta

**Target:** Ricercatori, studenti, hobbisti, early-stage startups

---

## 💼 Tier 2: Professional (€2,500/anno)

**Licenza:** Commercial License Agreement

**Include tutto di Community PLUS:**
- ✅ Supporto email prioritario (48h response)
- ✅ Accesso a versioni ottimizzate (CUDA kernels)
- ✅ Integrations pack (vLLM, HuggingFace, TensorRT-LLM)
- ✅ Dashboard metriche (self-hosted)
- ✅ Rimozione obbligo attribution
- ✅ 2 ore consulenza iniziale

**Limitazioni:**
- Max 10M richieste/mese
- Uso interno solo (no redistribuzione)

**Target:** Startup AI, PMI con MoE in produzione

---

## 🏢 Tier 3: Enterprise (Custom pricing)

**Prezzo base:** €8,000/anno + volume

**Include tutto di Professional PLUS:**
- ✅ Supporto dedicato (Slack/Teams channel)
- ✅ SLA garantito (99.9% uptime per componenti hosted)
- ✅ Custom ottimizzazioni per architettura cliente
- ✅ Training on-site (2 giorni)
- ✅ Code audit e security review
- ✅ Licenza redistribuzione (OEM)
- ✅ Accesso early a nuove features
- ✅ 10 ore consulenza/anno incluse

**Limitazioni:** Nessuna

**Target:** Enterprise, cloud providers, OEM

---

## 📋 License Agreement Template

```
ADAPTIVE-K COMMERCIAL LICENSE AGREEMENT

Effective Date: [DATE]
Licensor: Vertex Data S.r.l., Via [ADDRESS], P.IVA [NUMBER]
Licensee: [COMPANY NAME], [ADDRESS]

1. GRANT OF LICENSE
   Licensor grants Licensee a non-exclusive, non-transferable license 
   to use the Adaptive-K software ("Software") for internal business 
   purposes.

2. FEES
   Licensee agrees to pay the annual license fee of €[AMOUNT] within 
   30 days of invoice. License renews automatically unless cancelled 
   30 days before renewal date.

3. RESTRICTIONS
   Licensee shall NOT:
   a) Redistribute, sublicense, or sell the Software
   b) Remove copyright notices or attribution
   c) Use for competitive product development
   d) Exceed usage limits specified in selected tier

4. SUPPORT
   Licensor provides support as specified in the selected tier.
   Response times are targets, not guarantees (except Enterprise SLA).

5. WARRANTY DISCLAIMER
   SOFTWARE PROVIDED "AS IS". NO WARRANTY OF MERCHANTABILITY OR 
   FITNESS FOR PARTICULAR PURPOSE.

6. LIMITATION OF LIABILITY
   Licensor liability limited to fees paid in prior 12 months.
   No liability for indirect, consequential, or punitive damages.

7. TERM AND TERMINATION
   License effective for 12 months from Effective Date.
   Either party may terminate with 30 days written notice.
   Upon termination, Licensee must cease use and delete Software.

8. GOVERNING LAW
   This Agreement governed by laws of Italy.
   Disputes resolved in courts of [CITY].

SIGNATURES:

Licensor: _______________________  Date: ___________
         Vertex Data S.r.l.

Licensee: _______________________  Date: ___________
         [COMPANY NAME]
```

---

## 🛠️ Implementazione Tecnica

### License Key System

```python
# sdk/adaptive_k/licensing.py

import hashlib
import json
from datetime import datetime
from typing import Optional

class LicenseValidator:
    """Validates commercial licenses for Adaptive-K."""
    
    def __init__(self, license_key: Optional[str] = None):
        self.license_key = license_key
        self.tier = "community"  # Default
        
    def validate(self) -> dict:
        """Validate license and return tier info."""
        if not self.license_key:
            return {
                "tier": "community",
                "valid": True,
                "features": ["base_routing", "calibration", "cli"],
                "limits": {"requests_per_month": None}  # Unlimited for community
            }
        
        # Decode license key
        try:
            payload = self._decode_key(self.license_key)
            
            if self._is_expired(payload):
                return {"tier": "expired", "valid": False}
            
            return {
                "tier": payload["tier"],
                "valid": True,
                "features": self._get_features(payload["tier"]),
                "limits": self._get_limits(payload["tier"]),
                "company": payload.get("company", "Unknown"),
                "expires": payload["expires"]
            }
        except Exception:
            return {"tier": "invalid", "valid": False}
    
    def _decode_key(self, key: str) -> dict:
        """Decode and verify license key."""
        # Simple base64 + signature verification
        # In production: use asymmetric crypto (RSA/ECDSA)
        import base64
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError("Invalid key format")
        
        payload = json.loads(base64.b64decode(parts[0]))
        signature = parts[1]
        
        # Verify signature (simplified - use proper crypto in production)
        expected_sig = hashlib.sha256(
            (parts[0] + "VERTEX_SECRET").encode()
        ).hexdigest()[:16]
        
        if signature != expected_sig:
            raise ValueError("Invalid signature")
        
        return payload
    
    def _is_expired(self, payload: dict) -> bool:
        """Check if license has expired."""
        expires = datetime.fromisoformat(payload["expires"])
        return datetime.now() > expires
    
    def _get_features(self, tier: str) -> list:
        """Get features for tier."""
        features = {
            "community": ["base_routing", "calibration", "cli"],
            "professional": [
                "base_routing", "calibration", "cli",
                "cuda_kernels", "integrations", "dashboard",
                "priority_support"
            ],
            "enterprise": [
                "base_routing", "calibration", "cli",
                "cuda_kernels", "integrations", "dashboard",
                "priority_support", "custom_optimization",
                "redistribution", "sla"
            ]
        }
        return features.get(tier, features["community"])
    
    def _get_limits(self, tier: str) -> dict:
        """Get usage limits for tier."""
        limits = {
            "community": {"requests_per_month": None},
            "professional": {"requests_per_month": 10_000_000},
            "enterprise": {"requests_per_month": None}
        }
        return limits.get(tier, limits["community"])


def generate_license_key(
    company: str,
    tier: str,
    expires: str,
    secret: str = "VERTEX_SECRET"
) -> str:
    """Generate a license key (admin tool)."""
    import base64
    
    payload = {
        "company": company,
        "tier": tier,
        "expires": expires,
        "issued": datetime.now().isoformat()
    }
    
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    signature = hashlib.sha256((payload_b64 + secret).encode()).hexdigest()[:16]
    
    return f"{payload_b64}.{signature}"


# Usage in router
class AdaptiveKRouter:
    def __init__(self, license_key: Optional[str] = None, **kwargs):
        self.license = LicenseValidator(license_key)
        license_info = self.license.validate()
        
        if not license_info["valid"]:
            raise ValueError(f"Invalid license: {license_info['tier']}")
        
        self.tier = license_info["tier"]
        self.features = license_info["features"]
        
        # Enable/disable features based on tier
        if "cuda_kernels" not in self.features:
            self._disable_cuda_optimization()
```

---

## 💳 Payment Integration

### Opzione 1: Stripe (Raccomandato)

```javascript
// Stripe Checkout per license purchase
const stripe = require('stripe')('sk_live_...');

const session = await stripe.checkout.sessions.create({
  payment_method_types: ['card'],
  line_items: [{
    price_data: {
      currency: 'eur',
      product_data: {
        name: 'Adaptive-K Professional License',
        description: '1 year commercial license'
      },
      unit_amount: 250000, // €2,500 in cents
      recurring: {
        interval: 'year'
      }
    },
    quantity: 1
  }],
  mode: 'subscription',
  success_url: 'https://adaptive-k.vertexdata.it/license/success?session={CHECKOUT_SESSION_ID}',
  cancel_url: 'https://adaptive-k.vertexdata.it/pricing'
});
```

### Opzione 2: Gumroad (Più semplice)

- Crea prodotto su gumroad.com
- Link diretto: `https://vertexdata.gumroad.com/l/adaptive-k-pro`
- Gestione automatica fatture e tasse EU

### Opzione 3: Paddle (EU-friendly)

- Merchant of Record (gestiscono IVA EU)
- Fatturazione automatica
- Ideale per SaaS B2B

---

## 📊 Dashboard License Management

### MVP con Notion/Airtable

| Company | Email | Tier | License Key | Expires | Status |
|---------|-------|------|-------------|---------|--------|
| Acme AI | cto@acme.ai | Professional | eyJ...abc | 2027-01-14 | Active |
| BigCorp | ai@bigcorp.com | Enterprise | eyJ...xyz | 2027-06-01 | Active |

### Self-Hosted Dashboard (Fase 2)

```
/license-portal
├── /api
│   ├── generate-key.ts
│   ├── validate-key.ts
│   └── usage-stats.ts
├── /dashboard
│   ├── licenses.tsx
│   ├── analytics.tsx
│   └── billing.tsx
└── /webhooks
    └── stripe.ts
```

---

## 📈 Pricing Strategy

### Analisi Competitiva

| Competitor | Pricing | Notes |
|------------|---------|-------|
| Weights & Biases | $50/user/month | Per-seat |
| MLflow Enterprise | Custom | Usually 50-100K/year |
| Determined AI | $500/GPU/month | Infrastructure |

### Nostro Positioning

- **Più economico** di MLOps platforms
- **Più specifico** (solo routing optimization)
- **ROI chiaro** (40% FLOP savings = €X risparmio GPU)

### ROI Calculator per Sales

```
Input:
- GPU hours/month: 10,000
- GPU cost: €2/hour
- Current efficiency: 100%
- With Adaptive-K: 60% (40% savings)

Calculation:
- Current cost: 10,000 × €2 = €20,000/month
- With Adaptive-K: 6,000 × €2 = €12,000/month
- Monthly savings: €8,000
- Annual savings: €96,000
- License cost: €2,500/year
- ROI: 3,740%
```

---

## 🚀 Launch Checklist

### Immediate (This Week)
- [ ] Add licensing.py to SDK
- [ ] Create Gumroad products (Pro + Enterprise inquiry)
- [ ] Add "Pricing" section to landing page ✅ Done
- [ ] Prepare invoice template

### Short-term (1 Month)
- [ ] Stripe integration for recurring billing
- [ ] License key generation admin tool
- [ ] Usage tracking (optional, privacy-friendly)

### Medium-term (3 Months)
- [ ] Self-hosted license portal
- [ ] Automated onboarding emails
- [ ] Partner/reseller program

---

## 📞 Sales Process

### Inbound Lead (from website)

1. Lead fills contact form → Web3Forms → amministrazione@vertexdata.it
2. Reply within 24h with:
   - Availability for 30min call
   - Link to ROI calculator
   - Case study (when available)
3. Discovery call:
   - Understand their MoE setup
   - Quantify potential savings
   - Propose tier
4. Send proposal + contract
5. Close within 2 weeks

### Outbound (Cold)

1. Identify companies using MoE (LinkedIn, GitHub, papers)
2. Personalized email with specific savings estimate
3. Offer free PoC (2 weeks)
4. Convert to paid if successful

---

## 📝 Invoice Template

```
FATTURA N. [YEAR]/[NUMBER]

Vertex Data S.r.l.
Via [ADDRESS]
P.IVA: [NUMBER]
Email: amministrazione@vertexdata.it

Cliente:
[COMPANY NAME]
[ADDRESS]
P.IVA: [NUMBER]

Data: [DATE]
Scadenza: 30 giorni

─────────────────────────────────────────────────────
Descrizione                              Importo (€)
─────────────────────────────────────────────────────
Licenza Adaptive-K Professional
Periodo: [START] - [END]
(12 mesi, rinnovo automatico)                2,049.18

─────────────────────────────────────────────────────
Imponibile                                   2,049.18
IVA 22%                                        450.82
─────────────────────────────────────────────────────
TOTALE                                       2,500.00
─────────────────────────────────────────────────────

Pagamento: Bonifico bancario
IBAN: IT[XX] [XXXX] [XXXX] [XXXX] [XXXX] [XXXX] [XXX]
Causale: Fattura [NUMBER] - [COMPANY NAME]
```

---

## Next Actions

1. **Oggi:** Setup Gumroad/Stripe product
2. **Questa settimana:** Implementare licensing.py in SDK
3. **Prossima settimana:** Prima outreach a potenziali clienti
4. **Mese 1:** Primo cliente pagante = validazione

---

*Documento creato: 2026-01-14*
*Ultimo aggiornamento: 2026-01-14*
