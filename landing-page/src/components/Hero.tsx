'use client'

import { useLanguage } from '@/i18n'

export default function Hero() {
  const { t } = useLanguage()

  return (
    <section id="hero" className="pt-32 pb-20 px-4 relative overflow-hidden" aria-labelledby="hero-title">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-vs-blue/5 to-transparent pointer-events-none" aria-hidden="true" />
      
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Text content */}
          <div className="space-y-6">
            {/* Badges row */}
            <div className="flex flex-wrap gap-2">
              <div className="inline-flex items-center space-x-2 bg-vs-surface border border-vs-border rounded-full px-4 py-1.5">
                <span className="w-2 h-2 bg-vs-green rounded-full pulse-glow" />
                <span className="text-vs-sm text-vs-muted">TensorRT-LLM PR #10672</span>
              </div>
              <a 
                href="https://doi.org/10.5281/zenodo.18282008" 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center space-x-2 bg-vs-surface border border-vs-border rounded-full px-4 py-1.5 hover:border-vs-blue transition-colors"
              >
                <span className="text-vs-sm text-vs-cyan">DOI: 10.5281/zenodo.18282008</span>
              </a>
            </div>

            {/* Headline */}
            <h1 id="hero-title" className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
              <span className="text-vs-text">{t.hero.title}</span>
              <br />
              <span className="text-vs-text">{t.hero.titleHighlight} </span>
              <span className="text-vs-green">{t.hero.savings}</span>
            </h1>

            {/* Subheadline */}
            <p className="text-lg text-vs-muted max-w-xl">
              {t.hero.subtitle}{' '}
              <span className="text-vs-cyan">Mixtral</span>,{' '}
              <span className="text-vs-cyan">Qwen-MoE</span>, {t.hero.and}{' '}
              <span className="text-vs-cyan">OLMoE</span>.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <a href="#contact" className="btn-primary text-center">
                {t.hero.cta}
              </a>
              <a href="#resources" className="btn-secondary text-center">
                {t.hero.ctaSecondary}
              </a>
            </div>

            {/* Quick stats */}
            <div className="flex flex-wrap gap-8 pt-6 border-t border-vs-border">
              <div>
                <div className="stat-number text-3xl">52.5%</div>
                <div className="text-vs-muted text-sm">{t.hero.mixtralSavings}</div>
              </div>
              <div>
                <div className="stat-number-green text-3xl font-bold">32.4%</div>
                <div className="text-vs-muted text-sm">{t.hero.qwenSavings}</div>
              </div>
              <div>
                <div className="stat-number-purple text-3xl font-bold">24.7%</div>
                <div className="text-vs-muted text-sm">{t.hero.olmSavings}</div>
              </div>
            </div>
          </div>

          {/* Right: Code block */}
          <div className="relative">
            <div className="code-block glow-blue float-animation">
              {/* Window header */}
              <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-vs-border">
                <div className="w-3 h-3 rounded-full bg-vs-red" />
                <div className="w-3 h-3 rounded-full bg-vs-yellow" />
                <div className="w-3 h-3 rounded-full bg-vs-green" />
                <span className="text-vs-muted text-xs ml-4">adaptive_k_routing.py</span>
              </div>
              
              {/* Code content */}
              <pre className="text-sm leading-relaxed overflow-x-auto">
                <code>
                  <span className="code-keyword">def</span>{' '}
                  <span className="code-function">select_experts</span>
                  <span className="code-operator">(</span>
                  <span className="code-variable">router_logits</span>
                  <span className="code-operator">):</span>
                  {'\n'}
                  <span className="code-comment">    # Compute routing entropy</span>
                  {'\n'}
                  {'    '}
                  <span className="code-variable">probs</span>
                  <span className="code-operator"> = </span>
                  <span className="code-function">softmax</span>
                  <span className="code-operator">(</span>
                  <span className="code-variable">router_logits</span>
                  <span className="code-operator">)</span>
                  {'\n'}
                  {'    '}
                  <span className="code-variable">H</span>
                  <span className="code-operator"> = -</span>
                  <span className="code-function">sum</span>
                  <span className="code-operator">(</span>
                  <span className="code-variable">p</span>
                  <span className="code-operator"> * </span>
                  <span className="code-function">log</span>
                  <span className="code-operator">(</span>
                  <span className="code-variable">p</span>
                  <span className="code-operator">))</span>
                  {'\n\n'}
                  <span className="code-comment">    # Low entropy = confident routing</span>
                  {'\n'}
                  <span className="code-comment">    # Use fewer experts!</span>
                  {'\n'}
                  {'    '}
                  <span className="code-keyword">if</span>
                  <span className="code-variable"> H</span>
                  <span className="code-operator"> {'<'} </span>
                  <span className="code-number">0.6</span>
                  <span className="code-operator">:</span>
                  {'\n'}
                  {'        '}
                  <span className="code-variable">K</span>
                  <span className="code-operator"> = </span>
                  <span className="code-number">1</span>
                  <span className="code-comment">  # 87.5% compute saved</span>
                  {'\n'}
                  {'    '}
                  <span className="code-keyword">elif</span>
                  <span className="code-variable"> H</span>
                  <span className="code-operator"> {'<'} </span>
                  <span className="code-number">1.2</span>
                  <span className="code-operator">:</span>
                  {'\n'}
                  {'        '}
                  <span className="code-variable">K</span>
                  <span className="code-operator"> = </span>
                  <span className="code-number">2</span>
                  <span className="code-comment">  # 75% compute saved</span>
                  {'\n'}
                  {'    '}
                  <span className="code-keyword">else</span>
                  <span className="code-operator">:</span>
                  {'\n'}
                  {'        '}
                  <span className="code-variable">K</span>
                  <span className="code-operator"> = </span>
                  <span className="code-number">4</span>
                  <span className="code-comment">  # Full routing</span>
                  {'\n\n'}
                  {'    '}
                  <span className="code-keyword">return</span>
                  <span className="code-function"> top_k</span>
                  <span className="code-operator">(</span>
                  <span className="code-variable">probs</span>
                  <span className="code-operator">, </span>
                  <span className="code-variable">K</span>
                  <span className="code-operator">)</span>
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
