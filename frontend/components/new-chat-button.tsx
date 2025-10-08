'use client'

import { cn } from '@/lib/utils'
import { useRouter } from 'next/navigation'
import { buttonVariants } from './ui/button'
import { useLanguage } from '@/components/providers'

export function NewChatButton() {
  const router = useRouter()
  const { language } = useLanguage()

  return (
    <button
      onClick={() => {
        // Just reload the page to reset the chat
        window.location.href = '/'
      }}
      className={cn(
        buttonVariants({ size: 'sm', variant: 'outline' }),
        'flex gap-2'
      )}
    >
      {language === 'ar' ? 'إعادة تعيين الدردشة' : 'Reset Chat'}
    </button>
  )
}
