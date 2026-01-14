'use client'

import { useLanguage } from '@/i18n'

export default function Results() {
  const { t } = useLanguage()

  const results = [
    {
      model: 'Mixtral 8x7B',
      savings: '52.5%',
      accuracy: '99.8%',
      descriptionKey: 'mixtralDesc' as const,
      color: 'blue',
    },
    {
      model: 'Qwen-MoE',
      savings: '32.4%',
      accuracy: '99.9%',
      descriptionKey: 'qwenDesc' as const,
      color: 'green',
    },
    {
      model: 'OLMoE-1B-7B',
      savings: '24.7%',
      accuracy: '99.7%',
      descriptionKey: 'olmDesc' as const,
      color: 'purple',
    },
  ]

  return (
    <section id="results" className="py-20 px-4 bg-vs-bg">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-green">{t.results.validated}</span> {t.results.title}
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            {t.results.subtitle}
          </p>
        </div>

        {/* Results grid */}
        <div className="grid md:grid-cols-3 gap-6">
          {results.map((result, index) => (
            <div
              key={index}
              className={`card ${result.color === 'green' ? 'card-green' : result.color === 'purple' ? 'card-purple' : ''}`}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-vs-text">{result.model}</h3>
                <span className={`text-xs px-2 py-1 rounded-full bg-vs-${result.color}/20 text-vs-${result.color}`}>
                  MoE
                </span>
              </div>
              
              <div className="space-y-4">
                <div>
                  <div className={`stat-number${result.color === 'green' ? '-green' : result.color === 'purple' ? '-purple' : ''} text-4xl font-bold`}>
                    {result.savings}
                  </div>
                  <div className="text-vs-muted text-sm">{t.results.computeReduction}</div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="text-vs-green">✓</span>
                  <span className="text-vs-text">{result.accuracy} {t.results.accuracyRetained}</span>
                </div>
                
                <p className="text-vs-muted text-sm pt-2 border-t border-vs-border">
                  {t.results[result.descriptionKey]}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom note */}
        <div className="mt-12 text-center">
          <div className="inline-flex items-center space-x-2 bg-vs-surface border border-vs-border rounded-lg px-4 py-2">
            <span className="text-vs-yellow">⚡</span>
            <span className="text-vs-muted text-sm">
              {t.results.benchmark}
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}
