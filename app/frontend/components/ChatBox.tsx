'use client';
import { useState, useRef, useEffect } from 'react';

type Msg = { role: 'user' | 'bot'; text: string };

const PRESETS = [
  'Should I harvest this week?',
  'What should I do if a bloom is coming?',
  'When was the last documented bloom nearby?',
  'How much can I trust this number?',
];

export function ChatBox({ region }: { region: string }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 999999, behavior: 'smooth' });
  }, [messages, busy]);

  async function ask(question: string) {
    if (!question || busy) return;
    setInput('');
    setMessages((m) => [...m, { role: 'user', text: question }]);
    setBusy(true);
    try {
      const res = await fetch('/api/backend/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, region }),
      });
      const j = await res.json();
      setMessages((m) => [...m, { role: 'bot', text: j.answer || j.detail || '(no answer)' }]);
    } catch (e: any) {
      setMessages((m) => [...m, { role: 'bot', text: `Error: ${e.message}` }]);
    } finally {
      setBusy(false);
    }
  }
  const send = () => ask(input.trim());

  return (
    <div className="mt-6 rounded-2xl bg-white/40 shadow-inner overflow-hidden flex flex-col h-[440px]">
      <div className="border-b border-teal-950/10 px-4 py-3 text-sm font-semibold uppercase tracking-widest">
        Ask BloomWatch
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {messages.length === 0 && (
          <div className="text-sm text-teal-950/70">
            <div className="mb-3 uppercase tracking-widest text-xs text-teal-950/50">Try one</div>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => ask(p)}
                  disabled={busy}
                  className="px-3 py-1.5 rounded-full bg-cream border border-teal-950/20 text-xs hover:bg-teal-950 hover:text-cream transition disabled:opacity-40"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed ${
              m.role === 'user' ? 'bg-teal-950 text-cream' : 'bg-cream border border-teal-950/20 text-teal-950'
            }`}>
              {m.text}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-cream border border-teal-950/20 text-teal-950/60 px-4 py-2 text-sm italic">
              thinking…
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-teal-950/10 p-3 flex gap-2 bg-white/50">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Ask a question…"
          disabled={busy}
          className="flex-1 px-3 py-2 rounded-full bg-white text-teal-950 outline-none border border-teal-950/20 focus:border-teal-950 disabled:opacity-50"
        />
        <button
          onClick={send}
          disabled={busy || !input.trim()}
          className="px-5 py-2 rounded-full bg-teal-950 text-cream text-sm font-semibold disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
