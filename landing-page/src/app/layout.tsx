import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Adaptive-K MoE Routing | AI Inference Optimization',
  description: 'Entropy-guided dynamic expert selection for Mixture-of-Experts models. Reduce compute costs by 30-50% with proven results on Mixtral, Qwen-MoE, and OLMoE.',
  keywords: ['MoE', 'Mixture of Experts', 'AI optimization', 'inference optimization', 'Adaptive-K', 'LLM efficiency'],
  authors: [{ name: 'Gabriel Ballerini' }],
  openGraph: {
    title: 'Adaptive-K MoE Routing | AI Inference Optimization',
    description: 'Reduce AI inference costs by 30-50% with entropy-guided dynamic expert selection.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-vs-black text-vs-text antialiased">
        {children}
      </body>
    </html>
  )
}
