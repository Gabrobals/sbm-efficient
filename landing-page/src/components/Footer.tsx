export default function Footer() {
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
              Entropy-guided dynamic expert selection for Mixture-of-Experts models.
              Reduce inference costs by 30-50% with proven methodology.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-vs-text font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="https://github.com/Gabrobals/sbm-efficient" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  GitHub Repository
                </a>
              </li>
              <li>
                <a href="https://github.com/Gabrobals/sbm-efficient/blob/master/Entropy_Guided_Dynamic_Expert_Selection_in_Mixture_of_Experts_Models.pdf" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  Research Paper
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
            <h4 className="text-vs-text font-semibold mb-4">Contact</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="mailto:gabriele.ballerini@gmail.com" className="text-vs-muted hover:text-vs-blue transition-colors">
                  gabriele.ballerini@gmail.com
                </a>
              </li>
              <li>
                <a href="https://linkedin.com/in/gabriele-ballerini" target="_blank" rel="noopener noreferrer" className="text-vs-muted hover:text-vs-blue transition-colors">
                  LinkedIn
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-vs-border flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <p className="text-vs-muted text-sm">
            © {new Date().getFullYear()} Gabriel Ballerini. P.IVA IT18354371009
          </p>
          <p className="text-vs-muted text-sm">
            ATECO 62.01.00 - Software Development & Consulting
          </p>
        </div>
      </div>
    </footer>
  )
}
