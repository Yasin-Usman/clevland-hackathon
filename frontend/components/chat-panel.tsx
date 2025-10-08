'use client'

import * as React from 'react'
import { type Message } from 'ai'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { PromptForm } from '@/components/prompt-form'
import { Button } from '@/components/ui/button'
import { IconRefresh } from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useLanguage } from '@/components/providers'

export interface ChatPanelProps {
  id?: string
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  input: string
  setInput: (value: string) => void
  isLoading: boolean
  onSubmit: (value: string) => Promise<void>
}

export function ChatPanel({
  id,
  messages,
  setMessages,
  input,
  setInput,
  isLoading,
  onSubmit
}: ChatPanelProps) {
  const [specialty, setSpecialty] = React.useState<string>('')
  const [date, setDate] = React.useState<string>('')
  const [time, setTime] = React.useState<string>('')
  const { language } = useLanguage()

  const quickReplies =
    language === 'ar'
      ? ['حجز اليوم بعد الظهر', 'إعادة جدولة الموعد', 'إلغاء الموعد', 'مساعدة']
      : ['Book today afternoon', 'Reschedule appointment', 'Cancel appointment', 'Help']

  return (
    <div className="fixed inset-x-0 bottom-0 bg-gradient-to-b from-muted/10 from-10% to-muted/30 to-50%">
      <ButtonScrollToBottom />
      <div className="mx-auto sm:max-w-2xl sm:px-4">
        <div className="flex h-10 items-center justify-center">
          {isLoading ? (
            <Button disabled className="bg-background">
              {language === 'ar' ? 'جاري التوليد...' : 'Generating...'}
            </Button>
          ) : (
            messages?.length > 0 && (
              <Button
                variant="outline"
                onClick={() => window.location.reload()}
                className="bg-background"
              >
                <IconRefresh className="mr-2" />
                {language === 'ar' ? 'إعادة إنشاء الرد' : 'Regenerate response'}
              </Button>
            )
          )}
        </div>

        <div className="space-y-4 border-t bg-background px-4 py-2 shadow-lg sm:rounded-t-xl sm:border md:py-4">
          <div className="flex flex-wrap gap-2">
            {quickReplies.map(text => (
              <Button
                key={text}
                variant="secondary"
                size="sm"
                onClick={() => {
                  setInput(text)
                  setMessages(prev => [
                    ...prev,
                    { id: Date.now().toString(), content: text, role: 'user' }
                  ])
                }}
              >
                {text}
              </Button>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="space-y-1">
              <Label htmlFor="specialty">
                {language === 'ar' ? 'التخصص' : 'Specialty'}
              </Label>
              <Select
                onValueChange={value => setSpecialty(value)}
                value={specialty || undefined}
              >
                <SelectTrigger id="specialty">
                  <SelectValue
                    placeholder={language === 'ar' ? 'اختر' : 'Choose'}
                  />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="dermatology">
                    {language === 'ar' ? 'الأمراض الجلدية' : 'Dermatology'}
                  </SelectItem>
                  <SelectItem value="cardiology">
                    {language === 'ar' ? 'القلب' : 'Cardiology'}
                  </SelectItem>
                  <SelectItem value="pediatrics">
                    {language === 'ar' ? 'طب الأطفال' : 'Pediatrics'}
                  </SelectItem>
                  <SelectItem value="orthopedics">
                    {language === 'ar' ? 'العظام' : 'Orthopedics'}
                  </SelectItem>
                  <SelectItem value="gynecology">
                    {language === 'ar' ? 'أمراض النساء' : 'Gynecology'}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Label htmlFor="date">
                {language === 'ar' ? 'التاريخ' : 'Date'}
              </Label>
              <Input
                id="date"
                type="date"
                value={date}
                onChange={e => setDate(e.target.value)}
              />
            </div>

            <div className="space-y-1">
              <Label htmlFor="time">
                {language === 'ar' ? 'الوقت' : 'Time'}
              </Label>
              <Input
                id="time"
                type="time"
                value={time}
                onChange={e => setTime(e.target.value)}
              />
            </div>

            <div className="sm:col-span-3">
              <Button
                className="w-full"
                disabled={isLoading}
                onClick={() => {
                  const parts = []
                  if (date) parts.push(date)
                  if (time) parts.push(time)
                  const when = parts.join(' ')
                  const spec = specialty
                    ? language === 'ar'
                      ? ` مع ${specialty}`
                      : ` with a ${specialty}`
                    : ''
                  const message = when
                    ? language === 'ar'
                      ? `حجز موعد ${when}${spec}`
                      : `Book appointment for ${when}${spec}`
                    : language === 'ar'
                    ? `حجز موعد${spec}`
                    : `Book appointment${spec}`
                  setInput(message)
                  setMessages(prev => [
                    ...prev,
                    { id: Date.now().toString(), role: 'user', content: message }
                  ])
                }}
              >
                {language === 'ar' ? 'تحقق من التوافر' : 'Check availability'}
              </Button>
            </div>
          </div>

          {/* 🔥 Send message */}
          <PromptForm
            onSubmit={onSubmit}
            input={input}
            setInput={setInput}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  )
}
