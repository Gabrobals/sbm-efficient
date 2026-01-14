export default function Resources() {
  const resources = [
    {
      type: 'Paper',
      title: 'Entropy-Guided Dynamic Expert Selection in MoE Models',
      description: 'Full research paper with methodology, experiments, and results.',
      link: 'https://github.com/Gabrobals/sbm-efficient/blob/master/Entropy_Guided_Dynamic_Expert_Selection_in_Mixture_of_Experts_Models.pdf',
      icon: '📄',
    },
    {
      type: 'Code',
      title: 'Open Source Implementation',
      description: 'Reference implementation with examples for Mixtral, Qwen-MoE, and OLMoE.',
      link: 'https://github.com/Gabrobals/sbm-efficient',
      icon: '💻',
    },
    {
      type: 'PR',
      title: 'TensorRT-LLM Integration',
      description: 'Pull request adding AdaptiveKMoeRoutingMethod to NVIDIA TensorRT-LLM.',
      link: 'https://github.com/NVIDIA/TensorRT-LLM/pull/10672',
      icon: '🔀',
    },
  ]

  return (
    <section id="resources" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-cyan">Open</span> Resources
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            The research is open. The code is open. Start exploring today.
          </p>
        </div>

        {/* Resources grid */}
        <div className="grid md:grid-cols-3 gap-6">
          {resources.map((resource, index) => (
            <a
              key={index}
              href={resource.link}
              target="_blank"
              rel="noopener noreferrer"
              className="card group"
            >
              <div className="flex items-center space-x-3 mb-4">
                <span className="text-2xl">{resource.icon}</span>
                <span className="text-vs-muted text-sm uppercase tracking-wider">
                  {resource.type}
                </span>
              </div>
              
              <h3 className="text-lg font-semibold text-vs-text mb-2 group-hover:text-vs-blue transition-colors">
                {resource.title}
              </h3>
              
              <p className="text-vs-muted text-sm mb-4">{resource.description}</p>
              
              <div className="flex items-center space-x-2 text-vs-blue text-sm">
                <span>View resource</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </a>
          ))}
        </div>

        {/* Citation */}
        <div className="mt-12 p-6 bg-vs-surface border border-vs-border rounded-lg">
          <h4 className="text-sm text-vs-muted uppercase tracking-wider mb-3">Citation</h4>
          <div className="code-block text-xs overflow-x-auto">
            <pre className="text-vs-text">
{`@article{ballerini2025adaptivek,
  title={Entropy-Guided Dynamic Expert Selection in Mixture-of-Experts Models},
  author={Ballerini, Gabriel},
  year={2025},
  url={https://github.com/Gabrobals/sbm-efficient}
}`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
