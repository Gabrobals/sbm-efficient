'use client'

import { useState } from 'react'

export default function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    message: '',
    service: 'assessment',
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Create mailto link with form data
    const subject = encodeURIComponent(`[Adaptive-K] Inquiry from ${formData.name} - ${formData.company}`)
    const body = encodeURIComponent(
      `Name: ${formData.name}\n` +
      `Company: ${formData.company}\n` +
      `Service Interest: ${formData.service}\n\n` +
      `Message:\n${formData.message}`
    )
    window.location.href = `mailto:gabriele.ballerini@gmail.com?subject=${subject}&body=${body}`
  }

  return (
    <section id="contact" className="py-20 px-4 bg-vs-bg">
      <div className="max-w-4xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-blue">Get</span> In Touch
          </h2>
          <p className="text-vs-muted max-w-xl mx-auto">
            Ready to reduce your MoE inference costs? Let's discuss how Adaptive-K can help.
          </p>
        </div>

        {/* Contact form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label htmlFor="name" className="block text-sm text-vs-muted mb-2">
                Name *
              </label>
              <input
                type="text"
                id="name"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
                placeholder="Your name"
              />
            </div>
            
            <div>
              <label htmlFor="email" className="block text-sm text-vs-muted mb-2">
                Email *
              </label>
              <input
                type="email"
                id="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
                placeholder="you@company.com"
              />
            </div>
          </div>

          <div>
            <label htmlFor="company" className="block text-sm text-vs-muted mb-2">
              Company
            </label>
            <input
              type="text"
              id="company"
              value={formData.company}
              onChange={(e) => setFormData({ ...formData, company: e.target.value })}
              className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
              placeholder="Your company"
            />
          </div>

          <div>
            <label htmlFor="service" className="block text-sm text-vs-muted mb-2">
              Service Interest
            </label>
            <select
              id="service"
              value={formData.service}
              onChange={(e) => setFormData({ ...formData, service: e.target.value })}
              className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
            >
              <option value="assessment">Feasibility Assessment (€2,500+)</option>
              <option value="implementation">Implementation Package (€8,000+)</option>
              <option value="consulting">Expert Consulting (€1,000/day)</option>
              <option value="enterprise">Enterprise / Custom Solution</option>
            </select>
          </div>

          <div>
            <label htmlFor="message" className="block text-sm text-vs-muted mb-2">
              Message *
            </label>
            <textarea
              id="message"
              required
              rows={5}
              value={formData.message}
              onChange={(e) => setFormData({ ...formData, message: e.target.value })}
              className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors resize-none"
              placeholder="Tell us about your MoE deployment and what you're looking to achieve..."
            />
          </div>

          <button
            type="submit"
            className="w-full btn-primary py-4 text-lg"
          >
            Send Message
          </button>
        </form>

        {/* Alternative contact */}
        <div className="mt-8 text-center">
          <p className="text-vs-muted text-sm">
            Prefer email directly?{' '}
            <a href="mailto:gabriele.ballerini@gmail.com" className="text-vs-blue hover:underline">
              gabriele.ballerini@gmail.com
            </a>
          </p>
        </div>
      </div>
    </section>
  )
}
