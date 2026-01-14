export default function Results() {
  const results = [
    {
      model: 'Mixtral 8x7B',
      savings: '52.5%',
      accuracy: '99.8%',
      description: 'K=1 used 78% of the time with minimal quality loss',
      color: 'blue',
    },
    {
      model: 'Qwen-MoE',
      savings: '32.4%',
      accuracy: '99.9%',
      description: 'Effective across all entropy thresholds',
      color: 'green',
    },
    {
      model: 'OLMoE-1B-7B',
      savings: '24.7%',
      accuracy: '99.7%',
      description: 'Consistent savings on smaller MoE architecture',
      color: 'purple',
    },
  ]

  return (
    <section id="results" className="py-20 px-4 bg-vs-bg">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-green">Validated</span> Results
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            Real compute savings on production MoE models. 
            Accuracy measured relative to full Top-K routing baseline.
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
                  <div className="text-vs-muted text-sm">Compute Reduction</div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="text-vs-green">✓</span>
                  <span className="text-vs-text">{result.accuracy} accuracy retained</span>
                </div>
                
                <p className="text-vs-muted text-sm pt-2 border-t border-vs-border">
                  {result.description}
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
              Results validated via WikiText-2 perplexity benchmarks
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}
