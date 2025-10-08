'use client'

import { buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useLanguage } from '@/components/providers'

export function LanguageToggle() {
  const { language, setLanguage } = useLanguage()
  const isArabic = language === 'ar'

  return (
    <button
      type="button"
      onClick={() => setLanguage(isArabic ? 'en' : 'ar')}
      className={cn(buttonVariants({ variant: 'outline' }))}
      aria-label={isArabic ? 'Switch to English' : 'التبديل إلى العربية'}
      title={isArabic ? 'Switch to English' : 'التبديل إلى العربية'}
    >
      {isArabic ? 'AR' : 'EN'}
    </button>
  )
}


