'use client'

import { useState, useEffect } from 'react'
import { useLanguage } from '@/i18n'

const WEB3FORMS_KEY = '709b0b2d-560d-457c-8694-d83ba2bb0905'

export default function Contact() {
  const { t } = useLanguage()
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    message: '',
    service: 'assessment',
  })
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')

  // Read service from localStorage when triggered from Services
  useEffect(() => {
    const updateServiceFromStorage = () => {
      const selected = localStorage.getItem('selectedService')
      if (selected) {
        const validServices = ['assessment', 'implementation', 'consulting', 'enterprise']
        if (validServices.includes(selected)) {
          setFormData(prev => ({ ...prev, service: selected }))
        }
        localStorage.removeItem('selectedService') // Clean up
      }
    }
    
    // Run on mount
    updateServiceFromStorage()
    
    // Listen for custom event from Services
    window.addEventListener('serviceSelected', updateServiceFromStorage)
    return () => window.removeEventListener('serviceSelected', updateServiceFromStorage)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setStatus('loading')
    
    try {
      const response = await fetch('https://api.web3forms.com/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          access_key: WEB3FORMS_KEY,
          subject: `[Adaptive-K] Inquiry from ${formData.name} - ${formData.company}`,
          from_name: formData.name,
          email: formData.email,
          company: formData.company,
          service: formData.service,
          message: formData.message,
        }),
      })
      
      if (response.ok) {
        setStatus('success')
        setFormData({ name: '', email: '', company: '', message: '', service: 'assessment' })
      } else {
        setStatus('error')
      }
    } catch {
      setStatus('error')
    }
  }

  return (
    <section id="contact" className="py-20 px-4 bg-vs-bg">
      <div className="max-w-4xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="text-vs-blue">{t.contact.get}</span> {t.contact.inTouch}
          </h2>
          <p className="text-vs-muted max-w-xl mx-auto">
            {t.contact.subtitle}
          </p>
        </div>

        {/* Success/Error messages */}
        {status === 'success' && (
          <div className="mb-6 p-4 bg-green-900/50 border border-green-500 rounded-lg text-center">
            <p className="text-green-400">{t.contact.successMessage}</p>
          </div>
        )}
        {status === 'error' && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-center">
            <p className="text-red-400">{t.contact.errorMessage}</p>
          </div>
        )}

        {/* Contact form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label htmlFor="name" className="block text-sm text-vs-muted mb-2">
                {t.contact.name} *
              </label>
              <input
                type="text"
                id="name"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
                placeholder={t.contact.namePlaceholder}
              />
            </div>
            
            <div>
              <label htmlFor="email" className="block text-sm text-vs-muted mb-2">
                {t.contact.email} *
              </label>
              <input
                type="email"
                id="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
                placeholder={t.contact.emailPlaceholder}
              />
            </div>
          </div>

          <div>
            <label htmlFor="company" className="block text-sm text-vs-muted mb-2">
              {t.contact.company}
            </label>
            <input
              type="text"
              id="company"
              value={formData.company}
              onChange={(e) => setFormData({ ...formData, company: e.target.value })}
              className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
              placeholder={t.contact.companyPlaceholder}
            />
          </div>

          <div>
            <label htmlFor="service" className="block text-sm text-vs-muted mb-2">
              {t.contact.serviceInterest}
            </label>
            <select
              id="service"
              value={formData.service}
              onChange={(e) => setFormData({ ...formData, service: e.target.value })}
              className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors"
            >
              <option value="assessment">{t.contact.serviceAssessment}</option>
              <option value="implementation">{t.contact.serviceImplementation}</option>
              <option value="consulting">{t.contact.serviceConsulting}</option>
              <option value="enterprise">{t.contact.serviceEnterprise}</option>
            </select>
          </div>

          <div>
            <label htmlFor="message" className="block text-sm text-vs-muted mb-2">
              {t.contact.message} *
            </label>
            <textarea
              id="message"
              required
              rows={5}
              value={formData.message}
              onChange={(e) => setFormData({ ...formData, message: e.target.value })}
              className="w-full px-4 py-3 bg-vs-surface border border-vs-border rounded-lg text-vs-text focus:border-vs-blue focus:outline-none transition-colors resize-none"
              placeholder={t.contact.messagePlaceholder}
            />
          </div>

          <button
            type="submit"
            disabled={status === 'loading'}
            className="w-full btn-primary py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'loading' ? t.contact.sending : t.contact.sendMessage}
          </button>
        </form>

        {/* Alternative contact */}
        <div className="mt-8 text-center">
          <p className="text-vs-muted text-sm">
            {t.contact.orEmail}{' '}
            <a href="mailto:amministrazione@vertexdata.it" className="text-vs-blue hover:underline">
              amministrazione@vertexdata.it
            </a>
          </p>
        </div>
      </div>
    </section>
  )
}
