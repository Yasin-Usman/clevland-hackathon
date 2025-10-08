import { StreamingTextResponse } from 'ai'

export const runtime = 'edge'

type ChatMessage = {
  role: 'system' | 'user' | 'assistant'
  content: string
}

function generateRuleBasedReply(messages: ChatMessage[], language: 'en' | 'ar'): string {
  const latestUser = [...messages].reverse().find(m => m.role === 'user')
  const userText = (latestUser?.content || '').toLowerCase()

  // Simple rules for a medical appointment assistant
  const contains = (list: string[]) => list.some(k => userText.includes(k))

  if (contains(['hello', 'hi', 'hey']) || contains(['مرحب', 'اهلا'])) {
    return language === 'ar'
      ? 'مرحبًا! يمكنني مساعدتك في حجز أو إعادة جدولة أو إلغاء موعد طبي. ماذا تحتاج؟'
      : 'Hello! I can help you book, reschedule, or cancel a medical appointment. What do you need?'
  }

  if (contains(['book', 'schedule', 'appointment']) || contains(['احجز', 'موعد', 'جدول'])) {
    // Extract naive date/time hints
    const hasTomorrow = userText.includes('tomorrow')
    const hasToday = userText.includes('today')
    const hasMorning = userText.includes('morning')
    const hasAfternoon = userText.includes('afternoon')
    const when = hasToday ? (language === 'ar' ? 'اليوم' : 'today') : hasTomorrow ? (language === 'ar' ? 'غدًا' : 'tomorrow') : (language === 'ar' ? 'أقرب يوم متاح' : 'next available day')
    const time = hasMorning ? (language === 'ar' ? 'الصباح' : 'morning') : hasAfternoon ? (language === 'ar' ? 'بعد الظهر' : 'afternoon') : (language === 'ar' ? 'أقرب وقت متاح' : 'the next available time')
    return language === 'ar'
      ? `بالتأكيد — يمكنني المساعدة. لـ ${when}، أستطيع تقديم مواعيد في ${time}. من فضلك شارك تاريخًا مفضلًا (YYYY-MM-DD)، والوقت (HH:MM)، والطبيب أو التخصص.`
      : `Sure — I can help with that. For ${when}, I can offer slots in the ${time}. Please share a preferred date (YYYY-MM-DD), time (HH:MM), and the doctor or specialty.`
  }

  if (contains(['reschedule', 'move']) || contains(['إعادة', 'انقل'])) {
    return language === 'ar'
      ? 'لا مشكلة. يرجى تزويدي برقم الموعد الحالي والتاريخ/الوقت الجديد المفضل.'
      : 'No problem. Please provide your current appointment ID and the new preferred date/time.'
  }

  if (contains(['cancel']) || contains(['إلغاء'])) {
    return language === 'ar'
      ? 'أستطيع إلغاء ذلك لك. يرجى تزويدي برقم الموعد لتأكيد الإلغاء.'
      : 'I can cancel that for you. Please provide your appointment ID to confirm cancellation.'
  }

  if (contains(['doctor', 'dr.', 'dermatolog', 'cardiolog', 'pediatric', 'orthopedic', 'gynecolog']) || contains(['طبيب', 'دكتور', 'جلدية', 'قلب', 'أطفال', 'عظام', 'نساء'])) {
    return language === 'ar'
      ? 'تم. يرجى تزويدي بتاريخ ووقت مفضلين، وسأتحقق من التوافر لهذا التخصص.'
      : 'Got it. Please provide a preferred date and time, and I will check availability for that specialty.'
  }

  if (contains(['help', 'how']) || contains(['مساعدة', 'كيف'])) {
    return language === 'ar'
      ? 'يمكنك القول: "حجز موعد بتاريخ 2025-10-10 الساعة 10:30 مع طبيب جلدية" أو "إعادة جدولة الموعد 12345 إلى 2025-10-12 الساعة 14:00".'
      : 'You can say: "Book appointment for 2025-10-10 at 10:30 with a dermatologist" or "Reschedule appointment 12345 to 2025-10-12 14:00".'
  }

  // Default fallback
  return language === 'ar'
    ? 'يمكنني مساعدتك في حجز أو إعادة جدولة أو إلغاء موعد طبي. من فضلك أخبرني بما ترغب في القيام به.'
    : 'I can help you book, reschedule, or cancel a medical appointment. Please tell me what you would like to do.'
}

export async function POST(req: Request) {
  const { messages = [], language = 'en' } = (await req.json()) as {
    messages: ChatMessage[]
    language?: 'en' | 'ar'
  }

  const reply = generateRuleBasedReply(messages, language === 'ar' ? 'ar' : 'en')

  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      // Stream the reply in a few chunks to mimic AI streaming
      const chunks = reply.match(/.{1,60}(\s|$)/g) || [reply]
      let index = 0
      const interval = setInterval(() => {
        if (index >= chunks.length) {
          clearInterval(interval)
          controller.close()
          return
        }
        controller.enqueue(encoder.encode(chunks[index]))
        index += 1
      }, 40)
    }
  })

  return new StreamingTextResponse(stream)
}
