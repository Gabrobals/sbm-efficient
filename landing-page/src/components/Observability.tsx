'use client'

import { useLanguage } from '@/i18n'

export default function Observability() {
  const { t } = useLanguage()

  const features = [
    {
      icon: '📊',
      title: 'Prometheus Metrics',
      description: 'Production-ready metrics: latency, throughput, K distribution, compute savings',
      code: `metrics.start_http_server(9090)
# adaptive_k_latency_seconds
# adaptive_k_avg_k
# adaptive_k_compute_saved_ratio`,
    },
    {
      icon: '📝',
      title: 'Structured Logging',
      description: 'JSON-formatted logs for ELK, Datadog, or any log aggregator',
      code: `logger = get_logger("inference")
logger.log_inference(trace)
# {"ts":"...", "avg_k":1.5, "latency_ms":45}`,
    },
    {
      icon: '🔍',
      title: 'Tracing & Debug',
      description: 'Per-layer entropy analysis and K selection visualization',
      code: `debugger.trace_k_selection(entropies)
# Layer 0 | H=0.42 ████░░░░ | K=1
# Layer 1 | H=1.23 ████████ | K=4`,
    },
    {
      icon: '⚡',
      title: 'A/B Testing',
      description: 'Built-in framework to compare Adaptive-K vs Full-K in production',
      code: `ab_test.assign_variant(request_id)
ab_test.compute_results()
# Latency: -32%, Quality: -0.1%`,
    },
  ]

  return (
    <section id="observability" className="py-20 px-4 bg-vs-surface/50">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="inline-block px-4 py-1 bg-vs-purple/20 text-vs-purple rounded-full text-sm font-medium mb-4">
            NEW IN v0.1.4
          </span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-purple">Production</span> Observability
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            Monitor, debug, and optimize your Adaptive-K deployment with built-in observability tools.
          </p>
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="p-6 bg-vs-bg border border-vs-border rounded-lg hover:border-vs-purple/50 transition-colors"
            >
              <div className="flex items-start gap-4 mb-4">
                <span className="text-3xl">{feature.icon}</span>
                <div>
                  <h3 className="text-xl font-semibold text-vs-text">{feature.title}</h3>
                  <p className="text-vs-muted text-sm">{feature.description}</p>
                </div>
              </div>
              <pre className="bg-vs-surface p-4 rounded-md overflow-x-auto">
                <code className="text-sm text-vs-purple">{feature.code}</code>
              </pre>
            </div>
          ))}
        </div>

        {/* Install command */}
        <div className="mt-12 text-center">
          <p className="text-vs-muted mb-4">Install with observability support:</p>
          <div className="inline-block bg-vs-surface border border-vs-border rounded-lg p-4">
            <code className="text-vs-purple text-lg">
              pip install adaptive-k-routing[observability]
            </code>
          </div>
        </div>
      </div>
    </section>
  )
}
