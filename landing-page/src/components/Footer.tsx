'use client'

import { useLanguage } from '@/i18n'

export default function Footer() {
  const { t } = useLanguage()

  return (
    <footer className="py-12 px-4 border-t border-vs-border">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-vs-blue flex items-center justify-center">
                <span className="text-white font-bold text-sm">AK</span>
              </div>
              <span className="font-semibold text-lg">
                <span className="text-vs-blue">Adaptive</span>
                <span className="text-vs-text">-K</span>
              </span>
            </div>
            <p className="text-vs-muted text-sm max-w-md">
              {t.footer.description}
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-vs-text font-semibold mb-4">{t.footer.resources}</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="https://github.com/Gabrobals/sbm-efficient" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  {t.footer.github}
                </a>
              </li>
              <li>
                <a href="https://github.com/Gabrobals/sbm-efficient/blob/master/Entropy_Guided_Dynamic_Expert_Selection_in_Mixture_of_Experts_Models.pdf" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  {t.footer.paper}
                </a>
              </li>
              <li>
                <a href="https://github.com/NVIDIA/TensorRT-LLM/pull/10672" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  TensorRT-LLM PR
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="text-vs-text font-semibold mb-4">{t.footer.contact}</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="mailto:amministrazione@vertexdata.it" className="text-vs-muted hover:text-vs-blue transition-colors">
                  amministrazione@vertexdata.it
                </a>
              </li>
              <li>
                <a href="https://www.linkedin.com/in/gabriele-balsamo-629975123/" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  LinkedIn
                </a>
              </li>
              <li>
                <a href="https://github.com/Gabrobals" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  GitHub
                </a>
              </li>
              <li>
                <a href="https://www.vertexdata.it" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  VertexData.it
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-vs-border flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <p className="text-vs-muted text-sm">
            © {new Date().getFullYear()} Gabriele Balsamo. P.IVA IT18354371009
          </p>
          <p className="text-vs-muted text-sm">
            ATECO 62.01.00 - {t.footer.ateco}
          </p>
        </div>
      </div>
    </footer>
  )
}
