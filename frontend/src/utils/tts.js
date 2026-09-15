// frontend/src/utils/tts.js — Hindi spoken alert (Day 15).

const HINDI_PHRASE = (cameraName) => `संदिग्ध व्यक्ति मिला — ${cameraName}`;

/**
 * Prime the voice list.
 *
 * speechSynthesis.getVoices() returns [] on the first call in Chrome — the
 * list loads asynchronously and only fires 'voiceschanged' once. Calling
 * getVoices() for the first time inside speakCriticalAlert() therefore finds
 * no Hindi voice and silently falls back to the default one, every time.
 * Call this once on app mount so the list is warm before any alert arrives.
 */
export function warmVoiceList() {
  if (!('speechSynthesis' in window)) return;
  const voices = window.speechSynthesis.getVoices();
  if (voices.length === 0) {
    window.speechSynthesis.addEventListener(
      'voiceschanged',
      () => window.speechSynthesis.getVoices(),
      { once: true },
    );
  }
}

/**
 * Speak the Hindi alert line for a camera.
 *
 * iOS: speechSynthesis.speak() requires a prior user gesture in the current
 * browsing context. When the app is launched by tapping a push notification
 * there has been no gesture yet, so this no-ops. Rather than blocking the
 * card on audio, the utterance is queued and replayed on the first tap —
 * see flushPendingTTS(), wired to the card's onClick.
 *
 * Degrades quietly by design: no speechSynthesis at all, or no Hindi voice
 * installed, must never prevent the alert card from rendering. A silent
 * card still shows the alert; a card that threw would show nothing.
 */
export function speakCriticalAlert(cameraName) {
  if (!('speechSynthesis' in window)) return;

  const text = HINDI_PHRASE(cameraName || 'कैमरा');

  const speak = () => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'hi-IN';

    const voices = window.speechSynthesis.getVoices();
    const hindi = voices.find((v) => v.lang === 'hi-IN' || v.lang?.startsWith('hi'));
    // Optional assignment: with no Hindi voice the browser reads the
    // Devanagari with its default voice. Imperfect, but an officer hearing
    // *something* beats silence — so this is a fallback, not a skip.
    if (hindi) utterance.voice = hindi;

    utterance.addEventListener('error', (e) => {
      if (e.error === 'not-allowed' || e.error === 'not_allowed') {
        window._pendingTTS = speak;   // replay on first user gesture
      }
    });

    window.speechSynthesis.speak(utterance);
  };

  try {
    speak();
    // iOS often reports neither speaking nor pending when the gesture
    // requirement blocked it, without ever firing 'error'. Checking shortly
    // after gives a second chance to queue for the first tap.
    setTimeout(() => {
      const s = window.speechSynthesis;
      if (s && !s.speaking && !s.pending) window._pendingTTS = speak;
    }, 250);
  } catch {
    window._pendingTTS = speak;
  }
}

/** Replay a TTS utterance blocked by the iOS gesture requirement. */
export function flushPendingTTS() {
  if (typeof window._pendingTTS === 'function') {
    const fn = window._pendingTTS;
    window._pendingTTS = null;
    try { fn(); } catch { /* best effort */ }
  }
}
