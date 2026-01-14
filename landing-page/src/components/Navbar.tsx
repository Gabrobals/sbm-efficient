'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navLinks = [
    { href: '#results', label: 'Results' },
    { href: '#how-it-works', label: 'How It Works' },
    { href: '#services', label: 'Services' },
    { href: '#resources', label: 'Resources' },
    { href: '#contact', label: 'Contact' },
  ]

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
            <Link
              href="#contact"
              className="btn-primary text-sm px-4 py-2"
            >
              Get Started
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
            <Link
              href="#contact"
              className="block mt-4 btn-primary text-center"
              onClick={() => setMobileMenuOpen(false)}
            >
              Get Started
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}
