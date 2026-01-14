'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { translations, Language } from './translations'

type TranslationsType = typeof translations['en']

interface LanguageContextType {
  lang: Language
  setLang: (lang: Language) => void
  t: TranslationsType
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Language>('en')

  useEffect(() => {
    // Check localStorage first
    const saved = localStorage.getItem('lang') as Language
    if (saved && (saved === 'en' || saved === 'it')) {
      setLangState(saved)
      return
    }
    
    // Then check browser language
    const browserLang = navigator.language.toLowerCase()
    if (browserLang.startsWith('it')) {
      setLangState('it')
    }
  }, [])

  const setLang = (newLang: Language) => {
    setLangState(newLang)
    localStorage.setItem('lang', newLang)
  }

  const t = translations[lang] as TranslationsType

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider')
  }
  return context
}
