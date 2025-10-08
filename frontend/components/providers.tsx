'use client'

import * as React from 'react'
import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { ThemeProviderProps } from 'next-themes/dist/types'

import { TooltipProvider } from '@/components/ui/tooltip'

// Simple language context for EN/AR
export type SupportedLanguage = 'en' | 'ar'

type LanguageContextValue = {
  language: SupportedLanguage
  setLanguage: (lang: SupportedLanguage) => void
}

const LanguageContext = React.createContext<LanguageContextValue | undefined>(
  undefined
)

export function useLanguage(): LanguageContextValue {
  const ctx = React.useContext(LanguageContext)
  if (!ctx) throw new Error('useLanguage must be used within LanguageProvider')
  return ctx
}

function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = React.useState<SupportedLanguage>('en')

  React.useEffect(() => {
    if (typeof window === 'undefined') return
    const saved = window.localStorage.getItem('language') as
      | SupportedLanguage
      | null
    if (saved === 'en' || saved === 'ar') {
      setLanguage(saved)
    }
  }, [])

  React.useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('language', language)
    const html = document.documentElement
    html.lang = language
    html.dir = language === 'ar' ? 'rtl' : 'ltr'
  }, [language])

  const value = React.useMemo(() => ({ language, setLanguage }), [language])
  return (
    <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
  )
}

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <TooltipProvider>
        <LanguageProvider>{children}</LanguageProvider>
      </TooltipProvider>
    </NextThemesProvider>
  )
}
