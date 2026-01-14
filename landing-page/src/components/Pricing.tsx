'use client'

import { useLanguage } from '@/i18n'

export default function Pricing() {
  const { t } = useLanguage()

  const tiers = [
    {
      name: t.pricing.starterName,
      price: t.pricing.starterPrice,
      period: t.pricing.starterPeriod,
      description: t.pricing.starterDesc,
      features: t.pricing.starterFeatures,
      cta: t.pricing.starterCta,
      highlighted: false,
    },
    {
      name: t.pricing.proName,
      price: t.pricing.proPrice,
      period: t.pricing.proPeriod,
      description: t.pricing.proDesc,
      features: t.pricing.proFeatures,
      cta: t.pricing.proCta,
      highlighted: true,
    },
    {
      name: t.pricing.enterpriseName,
      price: t.pricing.enterprisePrice,
      period: t.pricing.enterprisePeriod,
      description: t.pricing.enterpriseDesc,
      features: t.pricing.enterpriseFeatures,
      cta: t.pricing.enterpriseCta,
      highlighted: false,
    },
  ]

  const handleGetStarted = (tier: string) => {
    localStorage.setItem('selectedService', tier)
    window.dispatchEvent(new CustomEvent('serviceSelected'))
    document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section id="pricing" className="py-20 bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            <span className="text-cyan-400">{t.pricing.transparent}</span> {t.pricing.title}
          </h2>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            {t.pricing.subtitle}
          </p>
        </div>

        {/* Pricing Tiers */}
        <div className="grid md:grid-cols-3 gap-8">
          {tiers.map((tier, index) => (
            <div
              key={index}
              className={`relative rounded-2xl p-8 ${
                tier.highlighted
                  ? 'bg-gradient-to-b from-cyan-900/50 to-gray-800 border-2 border-cyan-500'
                  : 'bg-gray-800 border border-gray-700'
              }`}
            >
              {tier.highlighted && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="bg-cyan-500 text-black text-sm font-semibold px-4 py-1 rounded-full">
                    {t.pricing.recommended}
                  </span>
                </div>
              )}

              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-white mb-2">{tier.name}</h3>
                <div className="mb-2">
                  <span className="text-4xl font-bold text-white">{tier.price}</span>
                  {tier.period && (
                    <span className="text-gray-400 ml-1">{tier.period}</span>
                  )}
                </div>
                <p className="text-gray-400 text-sm">{tier.description}</p>
              </div>

              <ul className="space-y-3 mb-8">
                {tier.features.map((feature, featureIndex) => (
                  <li key={featureIndex} className="flex items-start">
                    <svg
                      className="w-5 h-5 text-cyan-400 mr-3 mt-0.5 flex-shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-gray-300 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => handleGetStarted(tier.name)}
                className={`w-full py-3 px-6 rounded-lg font-semibold transition-all ${
                  tier.highlighted
                    ? 'bg-cyan-500 hover:bg-cyan-400 text-black'
                    : 'bg-gray-700 hover:bg-gray-600 text-white border border-gray-600'
                }`}
              >
                {tier.cta}
              </button>
            </div>
          ))}
        </div>

        {/* Enterprise callout */}
        <div className="mt-16 text-center">
          <div className="inline-flex items-center space-x-2 bg-gray-800 rounded-full px-6 py-3 border border-gray-700">
            <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
            </svg>
            <span className="text-gray-300">{t.pricing.volumeDiscount}</span>
            <a href="#contact" className="text-cyan-400 hover:text-cyan-300 font-medium">
              {t.pricing.contactSales}
            </a>
          </div>
        </div>

        {/* Trust badges */}
        <div className="mt-12 flex flex-wrap justify-center gap-8 text-gray-500 text-sm">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span>{t.pricing.securePayments}</span>
          </div>
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
            </svg>
            <span>{t.pricing.invoiceAvailable}</span>
          </div>
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>{t.pricing.moneyBack}</span>
          </div>
        </div>
      </div>
    </section>
  )
}
