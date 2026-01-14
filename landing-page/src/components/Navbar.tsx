'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useLanguage } from '@/i18n'

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const { lang, setLang, t } = useLanguage()

  const navLinks = [
    { href: '#results', label: t.nav.results },
    { href: '#how-it-works', label: t.nav.howItWorks },
    { href: '#services', label: t.nav.services },
    { href: '#pricing', label: t.nav.pricing },
    { href: '#resources', label: t.nav.resources },
    { href: '#contact', label: t.nav.contact },
  ]

  const toggleLanguage = () => {
    setLang(lang === 'en' ? 'it' : 'en')
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-vs-black/90 backdrop-blur-md border-b border-vs-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-vs-blue flex items-center justify-center">
              <span className="text-white font-bold text-sm">AK</span>
            </div>
            <span className="font-semibold text-lg">
              <span className="text-vs-blue">Adaptive</span>
              <span className="text-vs-text">-K</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-vs-muted hover:text-vs-blue transition-colors duration-200 text-sm"
              >
                {link.label}
              </Link>
            ))}
            
            {/* Language Switcher */}
            <button
              onClick={toggleLanguage}
              className="flex items-center space-x-1 px-3 py-1.5 rounded-md border border-vs-border hover:border-vs-blue text-vs-muted hover:text-vs-blue transition-all duration-200 text-sm"
              aria-label={lang === 'en' ? 'Passa a Italiano' : 'Switch to English'}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
              </svg>
              <span className="font-medium">{lang === 'en' ? 'IT' : 'EN'}</span>
            </button>

            <Link
              href="#contact"
              className="btn-primary text-sm px-4 py-2"
            >
              {t.nav.getStarted}
            </Link>
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 text-vs-muted hover:text-vs-text"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {mobileMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-vs-border">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="block py-2 text-vs-muted hover:text-vs-blue transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            
            {/* Mobile Language Switcher */}
            <button
              onClick={toggleLanguage}
              className="flex items-center space-x-2 py-2 text-vs-muted hover:text-vs-blue transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
              </svg>
              <span>{lang === 'en' ? 'Italiano' : 'English'}</span>
            </button>

            <Link
              href="#contact"
              className="block mt-4 btn-primary text-center"
              onClick={() => setMobileMenuOpen(false)}
            >
              {t.nav.getStarted}
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}
