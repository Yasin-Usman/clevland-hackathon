'use client'

import * as React from 'react'
import { type Message } from 'ai'

import { ChatList } from '@/components/chat-list'
import { ChatPanel } from '@/components/chat-panel'
import { ChatScrollAnchor } from '@/components/chat-scroll-anchor'
import { EmptyScreen } from '@/components/empty-screen'
import { Header } from '@/components/header'
import { cn } from '@/lib/utils'

export interface ChatProps extends React.ComponentProps<'div'> {
  initialMessages?: Message[]
  id?: string
}

export function Chat({ id, initialMessages, className }: ChatProps) {
  const [messages, setMessages] = React.useState<Message[]>(
    initialMessages || []
  )
  const [input, setInput] = React.useState('')
  const [isLoading, setIsLoading] = React.useState(false)

  // Reset chat state
  const handleReset = React.useCallback(() => {
    setMessages([])
    setInput('')
    setIsLoading(false)
  }, [])

  // 🧠 Send to FastAPI backend
  async function handleSubmit(value: string) {
    if (!value.trim()) return

    setIsLoading(true)

    // Add user message
    setMessages(prev => [
      ...prev,
      { id: Date.now().toString(), role: 'user', content: value }
    ])

    // Create a message ID for the assistant's response
    const assistantMessageId = (Date.now() + 1).toString()

    try {
      // Add empty assistant message that we'll stream into
      setMessages(prev => [
        ...prev,
        { id: assistantMessageId, role: 'assistant', content: '' }
      ])

      const res = await fetch('http://localhost:8000/chatdoctor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ condition: value })
      })

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`)
      }

      // Read the response as a stream of events
      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let accumulatedContent = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(5))
              if (data.chunk) {
                accumulatedContent += data.chunk
                setMessages(prev =>
                  prev.map(msg =>
                    msg.id === assistantMessageId
                      ? { ...msg, content: accumulatedContent }
                      : msg
                  )
                )
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: '⚠️ Server error or backend not reachable.'
        }
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      {console.log('💬 Chat rendered')}
      {/* <Header onReset={handleReset} /> */}
      <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
        {messages.length ? (
          <>
            <ChatList messages={messages} />
            <ChatScrollAnchor trackVisibility={isLoading} />
          </>
        ) : (
          <EmptyScreen />
        )}
      </div>
      {/* bottom area */}
      <ChatPanel
        id={id}
        isLoading={isLoading}
        messages={messages}
        setMessages={setMessages}
        input={input}
        setInput={setInput}
        onSubmit={handleSubmit}
      />
    </>
  )
}
