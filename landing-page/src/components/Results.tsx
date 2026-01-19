'use client'

import { useLanguage } from '@/i18n'

export default function Results() {
  const { t } = useLanguage()

  const results = [
    {
      model: 'Nemotron 3 Nano',
      savings: '33.3%',
      accuracy: '99.9%',
      descriptionKey: 'nemotronDesc' as const,
      color: 'green',
      badge: 'NEW',
    },
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
      color: 'purple',
    },
    {
      model: 'OLMoE-1B-7B',
      savings: '24.7%',
      accuracy: '99.7%',
      descriptionKey: 'olmDesc' as const,
      color: 'cyan',
    },
  ]

  // NEW: Combination experiments showing multiplicative savings
  const combinations = [
    {
      name: 'Adaptive-K + Early Exit',
      savings: '68.0%',
      compute: '32.0%',
      badge: 'COMBO',
    },
    {
      name: 'Adaptive-K + ToMe',
      savings: '51.9%',
      compute: '48.1%',
      badge: 'COMBO',
    },
    {
      name: 'Triple Combo',
      savings: '96.0%',
      compute: '4.0%',
      badge: 'MAX',
    },
  ]

  return (
    <section id="results" aria-labelledby="results-title" className="py-20 px-4 bg-vs-bg">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 id="results-title" className="text-3xl md:text-4xl font-bold mb-4">
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

        {/* NEW: Combination Experiments Section */}
        <div className="mt-16">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-vs-text mb-2">
              🔬 {t.results.combinationsTitle || 'Multiplicative Savings: Technique Combinations'}
            </h3>
            <p className="text-vs-muted max-w-2xl mx-auto">
              {t.results.combinationsSubtitle || 'Adaptive-K stacks with other optimizations. Savings multiply, not just add.'}
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6">
            {combinations.map((combo, index) => (
              <div
                key={index}
                className={`card relative overflow-hidden ${combo.badge === 'MAX' ? 'border-vs-green border-2' : ''}`}
              >
                {combo.badge === 'MAX' && (
                  <div className="absolute top-0 right-0 bg-vs-green text-black text-xs font-bold px-2 py-1 rounded-bl">
                    🏆 BEST
                  </div>
                )}
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-vs-text">{combo.name}</h4>
                  <span className="text-xs px-2 py-1 rounded-full bg-vs-purple/20 text-vs-purple">
                    {combo.badge}
                  </span>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <div className={`text-4xl font-bold ${combo.badge === 'MAX' ? 'text-vs-green' : 'stat-number'}`}>
                      {combo.savings}
                    </div>
                    <div className="text-vs-muted text-sm">{t.results.computeReduction}</div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-vs-cyan">→</span>
                    <span className="text-vs-text">{t.results.onlyCompute || 'Only'} {combo.compute} {t.results.computeUsed || 'compute used'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Explanation box */}
          <div className="mt-8 bg-vs-surface border border-vs-border rounded-lg p-6">
            <p className="text-vs-muted text-sm">
              <span className="text-vs-yellow font-bold">💡 {t.results.keyInsight || 'Key Insight'}:</span>{' '}
              {t.results.multiplicativeExplanation || 'Adaptive-K reduces experts per token, Early Exit skips layers, Token Pruning (ToMe) reduces sequence length. Combined: 0.741 × 0.432 × 0.125 = 0.040 (96% savings). See Whitepaper Proposition 7.1.'}
            </p>
          </div>
        </div>

        {/* Bottom note */}
        <div className="mt-12 text-center space-y-4">
          <div className="inline-flex items-center space-x-2 bg-vs-surface border border-vs-border rounded-lg px-4 py-2">
            <span className="text-vs-yellow">⚡</span>
            <span className="text-vs-muted text-sm">
              {t.results.benchmark}
            </span>
          </div>
          
          {/* Dashboard Link */}
          <div>
            <a 
              href="/dashboard.html" 
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-gradient-to-r from-vs-blue to-vs-purple text-white px-6 py-3 rounded-lg font-semibold hover:opacity-90 transition-opacity"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              {t.results.viewDashboard || 'View Interactive Dashboard'}
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
