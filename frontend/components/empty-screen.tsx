import { UseChatHelpers } from 'ai/react'

import { Button } from '@/components/ui/button'
import { ExternalLink } from '@/components/external-link'
import { IconArrowRight } from '@/components/ui/icons'
import { useLanguage } from '@/components/providers'

const exampleMessages = [
  {
    heading: 'Explain technical concepts',
    message: `What is a "serverless function"?`
  },
  {
    heading: 'Summarize an article',
    message: 'Summarize the following article for a 2nd grader: \n'
  },
  {
    heading: 'Draft an email',
    message: `Draft an email to my boss about the following: \n`
  }
]

export function EmptyScreen({ setInput }: Pick<UseChatHelpers, 'setInput'>) {
  const { language } = useLanguage()
  return (
    <div className="mx-auto max-w-2xl px-4">
      <div className="rounded-lg border bg-background p-8">
        <h1 className="mb-2 text-lg font-semibold">
          {language === 'ar' ? 'مرحبًا بك في نموذج الدردشة.' : 'Welcome to Chatbot Starter.'}
        </h1>
        <p className="mb-2 leading-normal text-muted-foreground">
          {language === 'ar'
            ? <>هذا قالب تطبيق دردشة مفتوح المصدر مبني باستخدام <ExternalLink href="https://nextjs.org">Next.js</ExternalLink>.</>
            : <>This is an open source AI chatbot app template built with <ExternalLink href="https://nextjs.org">Next.js</ExternalLink> and <ExternalLink href="https://humanloop.com/">Humanloop</ExternalLink>.</>}
        </p>
        <p className="leading-normal text-muted-foreground">
          {language === 'ar' ? 'يمكنك بدء محادثة هنا أو تجربة الأمثلة التالية:' : 'You can start a conversation here or try the following examples:'}
        </p>
        <div className="mt-4 flex flex-col items-start space-y-2">
          {exampleMessages.map((message, index) => (
            <Button
              key={index}
              variant="link"
              className="h-auto p-0 text-base"
              onClick={() => setInput(message.message)}
            >
              <IconArrowRight className="mr-2 text-muted-foreground" />
              {language === 'ar' ? 'مثال' : message.heading}
            </Button>
          ))}
        </div>
      </div>
    </div>
  )
}
