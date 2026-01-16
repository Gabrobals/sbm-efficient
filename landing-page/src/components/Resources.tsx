'use client'

import { useLanguage } from '@/i18n'

// Professional SVG icons
const icons = {
  paper: (
    <svg className="w-8 h-8 text-vs-blue" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  ),
  whitepaper: (
    <svg className="w-8 h-8 text-vs-cyan" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
    </svg>
  ),
  code: (
    <svg className="w-8 h-8 text-vs-green" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
    </svg>
  ),
  pr: (
    <svg className="w-8 h-8 text-vs-purple" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
    </svg>
  ),
  dashboard: (
    <svg className="w-8 h-8 text-vs-yellow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
    </svg>
  ),
}

export default function Resources() {
  const { t } = useLanguage()

  const resources = [
    {
      typeKey: 'paperType' as const,
      titleKey: 'paperTitle' as const,
      descKey: 'paperDesc' as const,
      link: 'https://github.com/Gabrobals/sbm-efficient/blob/master/Entropy_Guided_Dynamic_Expert_Selection_in_Mixture_of_Experts_Models.pdf',
      icon: icons.paper,
    },
    {
      typeKey: 'whitepaperType' as const,
      titleKey: 'whitepaperTitle' as const,
      descKey: 'whitepaperDesc' as const,
      link: '/whitepaper.html',
      icon: icons.whitepaper,
    },
    {
      typeKey: 'dashboardType' as const,
      titleKey: 'dashboardTitle' as const,
      descKey: 'dashboardDesc' as const,
      link: '/dashboard.html',
      icon: icons.dashboard,
    },
    {
      typeKey: 'codeType' as const,
      titleKey: 'codeTitle' as const,
      descKey: 'codeDesc' as const,
      link: 'https://github.com/Gabrobals/sbm-efficient',
      icon: icons.code,
    },
    {
      typeKey: 'prType' as const,
      titleKey: 'prTitle' as const,
      descKey: 'prDesc' as const,
      link: 'https://github.com/NVIDIA/TensorRT-LLM/pull/10672',
      icon: icons.pr,
    },
  ]

  return (
    <section id="resources" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-cyan">{t.resources.open}</span> {t.resources.title}
          </h2>
          <p className="text-vs-muted max-w-2xl mx-auto">
            {t.resources.subtitle}
          </p>
        </div>

        {/* Resources grid */}
        <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-6">
          {resources.map((resource, index) => (
            <a
              key={index}
              href={resource.link}
              target="_blank"
              rel="noopener noreferrer"
              className="card group"
            >
              <div className="flex items-center space-x-3 mb-4">
                {resource.icon}
                <span className="text-vs-muted text-sm uppercase tracking-wider">
                  {t.resources[resource.typeKey]}
                </span>
              </div>
              
              <h3 className="text-lg font-semibold text-vs-text mb-2 group-hover:text-vs-blue transition-colors">
                {t.resources[resource.titleKey]}
              </h3>
              
              <p className="text-vs-muted text-sm mb-4">{t.resources[resource.descKey]}</p>
              
              <div className="flex items-center space-x-2 text-vs-blue text-sm">
                <span>{t.resources.viewResource}</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </a>
          ))}
        </div>

        {/* Citation */}
        <div className="mt-12 p-6 bg-vs-surface border border-vs-border rounded-lg">
          <h4 className="text-sm text-vs-muted uppercase tracking-wider mb-3">{t.resources.citation}</h4>
          <div className="code-block text-xs overflow-x-auto">
            <pre className="text-vs-text">
{`@article{balsamo2025adaptivek,
  title={Entropy-Guided Dynamic Expert Selection in Mixture-of-Experts Models},
  author={Balsamo, Gabriele},
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
