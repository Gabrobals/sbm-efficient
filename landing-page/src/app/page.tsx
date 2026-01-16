'use client'

import { LanguageProvider } from '@/i18n'
import Navbar from '@/components/Navbar'
import Hero from '@/components/Hero'
import Results from '@/components/Results'
import HowItWorks from '@/components/HowItWorks'
import Observability from '@/components/Observability'
import Services from '@/components/Services'
import Pricing from '@/components/Pricing'
import Resources from '@/components/Resources'
import Contact from '@/components/Contact'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <LanguageProvider>
      <main className="min-h-screen" role="main">
        <Navbar />
        <article itemScope itemType="https://schema.org/TechArticle">
          <Hero />
          <Results />
          <HowItWorks />
          <Observability />
        </article>
        <Services />
        <Pricing />
        <Resources />
        <Contact />
        <Footer />
      </main>
    </LanguageProvider>
  )
}
