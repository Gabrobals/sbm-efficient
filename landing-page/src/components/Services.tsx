export default function Services() {
  const services = [
    {
      icon: '📊',
      title: 'Feasibility Assessment',
      price: 'From €2,500',
      duration: '1-2 weeks',
      description: 'Analyze your MoE deployment to estimate potential savings',
      features: [
        'Router entropy analysis',
        'Savings projection report',
        'Implementation roadmap',
        'Risk assessment',
      ],
      color: 'blue',
    },
    {
      icon: '🔧',
      title: 'Implementation Package',
      price: 'From €8,000',
      duration: '4-6 weeks',
      description: 'Full Adaptive-K integration into your inference pipeline',
      features: [
        'Custom threshold calibration',
        'Production-ready code',
        'Performance benchmarks',
        'Integration support',
        '30-day warranty',
      ],
      color: 'green',
      featured: true,
    },
    {
      icon: '🎯',
      title: 'Expert Consulting',
      price: '€1,000/day',
      duration: 'Flexible',
      description: 'On-demand expertise for your AI optimization needs',
      features: [
        'Architecture review',
        'Performance tuning',
        'Team training',
        'Code review',
      ],
      color: 'purple',
    },
  ]

  return (
    <section id="services" className="py-20 px-4 bg-vs-bg">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-yellow">Professional</span> Services
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            Bring Adaptive-K savings to your production MoE deployments.
            All services include documentation and knowledge transfer.
          </p>
        </div>

        {/* Services grid */}
        <div className="grid md:grid-cols-3 gap-6">
          {services.map((service, index) => (
            <div
              key={index}
              className={`relative card ${
                service.color === 'green' ? 'card-green' : service.color === 'purple' ? 'card-purple' : ''
              } ${service.featured ? 'ring-2 ring-vs-green' : ''}`}
            >
              {service.featured && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <span className="bg-vs-green text-vs-black text-xs font-semibold px-3 py-1 rounded-full">
                    Most Popular
                  </span>
                </div>
              )}
              
              <div className="text-3xl mb-4">{service.icon}</div>
              
              <h3 className="text-xl font-semibold text-vs-text mb-2">{service.title}</h3>
              
              <div className="flex items-baseline space-x-2 mb-2">
                <span className={`text-2xl font-bold text-vs-${service.color}`}>{service.price}</span>
              </div>
              
              <div className="text-vs-muted text-sm mb-4">
                Typical duration: {service.duration}
              </div>
              
              <p className="text-vs-muted mb-6">{service.description}</p>
              
              <ul className="space-y-2 mb-6">
                {service.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center space-x-2 text-sm">
                    <span className="text-vs-green">✓</span>
                    <span className="text-vs-text">{feature}</span>
                  </li>
                ))}
              </ul>
              
              <a
                href="#contact"
                className={`block text-center py-2 rounded-lg transition-colors ${
                  service.featured
                    ? 'bg-vs-green text-vs-black font-semibold hover:bg-vs-green/90'
                    : 'border border-vs-border text-vs-text hover:border-vs-blue hover:text-vs-blue'
                }`}
              >
                Get Started
              </a>
            </div>
          ))}
        </div>

        {/* Enterprise note */}
        <div className="mt-12 text-center">
          <p className="text-vs-muted">
            Need a custom solution?{' '}
            <a href="#contact" className="text-vs-blue hover:underline">
              Contact us for enterprise pricing
            </a>
          </p>
        </div>
      </div>
    </section>
  )
}
