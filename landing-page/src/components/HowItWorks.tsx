'use client'

import { useLanguage } from '@/i18n'

export default function HowItWorks() {
  const { t } = useLanguage()

  const steps = [
    {
      number: '01',
      titleKey: 'step1Title' as const,
      descKey: 'step1Desc' as const,
      code: 'H = -sum(p * log(p))',
    },
    {
      number: '02',
      titleKey: 'step2Title' as const,
      descKey: 'step2Desc' as const,
      code: 'K = 1 if H < 0.6 else (2 if H < 1.2 else 4)',
    },
    {
      number: '03',
      titleKey: 'step3Title' as const,
      descKey: 'step3Desc' as const,
      code: 'output = sum(expert[i](x) * w[i] for i in top_k)',
    },
  ]

  return (
    <section id="how-it-works" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-purple">{t.howItWorks.how}</span> {t.howItWorks.itWorks}
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            {t.howItWorks.subtitle}
          </p>
        </div>

        {/* Steps */}
        <div className="space-y-8">
          {steps.map((step, index) => (
            <div
              key={index}
              className="grid md:grid-cols-12 gap-6 items-center p-6 bg-vs-surface border border-vs-border rounded-lg hover:border-vs-purple/50 transition-colors"
            >
              {/* Number */}
              <div className="md:col-span-1">
                <span className="text-vs-purple font-mono text-2xl font-bold">{step.number}</span>
              </div>
              
              {/* Content */}
              <div className="md:col-span-6">
                <h3 className="text-xl font-semibold text-vs-text mb-2">{t.howItWorks[step.titleKey]}</h3>
                <p className="text-vs-muted">{t.howItWorks[step.descKey]}</p>
              </div>
              
              {/* Code */}
              <div className="md:col-span-5">
                <div className="code-block text-center">
                  <code className="text-vs-cyan font-mono">{step.code}</code>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Key insight */}
        <div className="mt-12 p-6 bg-gradient-to-r from-vs-purple/10 to-vs-blue/10 border border-vs-purple/30 rounded-lg">
          <div className="flex items-start space-x-4">
            <span className="text-2xl">💡</span>
            <div>
              <h4 className="text-lg font-semibold text-vs-text mb-2">{t.howItWorks.keyInsight}</h4>
              <p className="text-vs-muted">
                {t.howItWorks.keyInsightText}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
