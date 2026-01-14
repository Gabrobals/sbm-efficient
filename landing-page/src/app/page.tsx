import Navbar from '@/components/Navbar'
import Hero from '@/components/Hero'
import Results from '@/components/Results'
import HowItWorks from '@/components/HowItWorks'
import Services from '@/components/Services'
import Resources from '@/components/Resources'
import Contact from '@/components/Contact'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navbar />
      <Hero />
      <Results />
      <HowItWorks />
      <Services />
      <Resources />
      <Contact />
      <Footer />
    </main>
  )
}
