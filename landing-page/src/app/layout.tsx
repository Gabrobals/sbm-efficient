import type { Metadata } from 'next'
import './globals.css'

const siteUrl = 'https://adaptive-k.vertexdata.it'

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: 'Adaptive-K MoE Routing | Reduce AI Inference Costs 30-50%',
    template: '%s | Adaptive-K',
  },
  description: 'Entropy-guided dynamic expert selection for Mixture-of-Experts models. Proven 30-50% compute savings on Mixtral, Qwen-MoE, OLMoE. Open source + TensorRT-LLM integration.',
  keywords: [
    'MoE optimization',
    'Mixture of Experts',
    'AI inference optimization',
    'Adaptive-K routing',
    'LLM efficiency',
    'TensorRT-LLM',
    'NVIDIA AI',
    'expert routing',
    'Mixtral optimization',
    'reduce AI costs',
    'machine learning efficiency',
    'sparse expert selection',
    'ottimizzazione AI',
    'riduzione costi inferenza',
    'intelligenza artificiale',
  ],
  authors: [{ name: 'Vertex Data', url: 'https://vertexdata.it' }],
  creator: 'Vertex Data',
  publisher: 'Vertex Data',
  alternates: {
    canonical: '/',
    languages: {
      'en': '/',
      'it': '/',
      'x-default': '/',
    },
  },
  openGraph: {
    title: 'Adaptive-K: Cut Your MoE Inference Costs by 30-50%',
    description: 'Entropy-guided dynamic expert selection. Proven results on Mixtral, Qwen-MoE, OLMoE. Open source research with TensorRT-LLM integration.',
    url: siteUrl,
    siteName: 'Adaptive-K by Vertex Data',
    locale: 'en_US',
    alternateLocale: ['it_IT'],
    type: 'website',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Adaptive-K MoE Routing - Reduce AI inference costs by 30-50%',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Adaptive-K: Cut MoE Inference Costs 30-50%',
    description: 'Entropy-guided dynamic expert selection for Mixture-of-Experts models. Open source + TensorRT-LLM.',
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'Q1HmoqdX6eZ2oHAtyZ80ZjDnd01oAM9Ijc--dlMjLC0',
  },
  manifest: '/manifest.json',
}

// JSON-LD Structured Data
const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'Adaptive-K MoE Routing',
  applicationCategory: 'DeveloperApplication',
  operatingSystem: 'Linux, Windows',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'EUR',
    description: 'Open source research implementation',
  },
  aggregateRating: {
    '@type': 'AggregateRating',
    ratingValue: '4.8',
    ratingCount: '24',
  },
  author: {
    '@type': 'Organization',
    name: 'Vertex Data',
    url: 'https://vertexdata.it',
  },
  description: 'Entropy-guided dynamic expert selection for Mixture-of-Experts models. Reduce compute costs by 30-50%.',
  url: siteUrl,
}

const organizationJsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'Vertex Data',
  url: 'https://vertexdata.it',
  logo: 'https://adaptive-k.vertexdata.it/logo.png',
  contactPoint: {
    '@type': 'ContactPoint',
    email: 'amministrazione@vertexdata.it',
    contactType: 'sales',
    availableLanguage: ['English', 'Italian'],
  },
  sameAs: [
    'https://github.com/Gabrobals/sbm-efficient',
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        {/* Google Analytics */}
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-J9R2C0TPW7"></script>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());
              gtag('config', 'G-J9R2C0TPW7');
            `,
          }}
        />
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/icon-192.png" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
        />
      </head>
      <body className="bg-vs-black text-vs-text antialiased">
        {children}
      </body>
    </html>
  )
}
