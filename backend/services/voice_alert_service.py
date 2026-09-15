"""
backend/services/voice_alert_service.py — Multilingual Voice AI Alert Broadcast Engine
======================================================================================
Generates spoken audio alerts in Hindi, Gujarati, and English for police control room
PA broadcasts and officer radio dispatches.

Features:
  1. Spoken prompts tailored to 7 crime categories (Wanted Fugitive, Stolen Vehicle, Loitering, etc.)
  2. Multi-engine synthesis:
     - Primary Offline: Native Windows SAPI5 / Linux eSpeak (0ms latency, zero cloud dependency)
     - Secondary Online: Google Text-to-Speech (gTTS)
     - Fallback: Synthesized audio chime (.wav)
  3. Real audio persistence in data/audio_alerts/
"""

from __future__ import annotations

import os
import sys
import uuid
import wave
import math
import struct
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("sentinel.voice_ai")

_AUDIO_DIR = Path("data/audio_alerts")
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# Multilingual template dictionary
ALERT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "WATCHLIST_FACE_MATCH": {
        "hi": "चेतावनी! {camera_name} पर वांटेड अपराधी {subject} की पहचान हुई है। पीसीआर वैन तुरंत रवाना करें।",
        "gu": "સાવચેતી! {camera_name} પર વોન્ટેડ ગુનેગાર {subject} ની ઓળખ થઈ છે. તાત્કાલિક પીસીઆર વાન મોકલો.",
        "en": "CRITICAL ALERT! Wanted suspect {subject} detected at {camera_name}. Patrol units dispatched immediately.",
    },
    "STOLEN_VEHICLE_WATCHLIST_HIT": {
        "hi": "सावधान! {camera_name} पर चोरी का वाहन नंबर {subject} डिटेक्ट हुआ है। तुरंत नाकाबंदी करें।",
        "gu": "સાવધાન! {camera_name} પર ચોરાયેલ વાહન નંબર {subject} દેખાયું છે. તાત્કાલિક નાકાબંધી કરો.",
        "en": "HIGH ALERT! Stolen vehicle with plate {subject} intercepted at {camera_name}. Initiate perimeter blockade.",
    },
    "NIGHT_PERIMETER_INTRUSION": {
        "hi": "अलर्ट! {camera_name} पर रात में अनाधिकृत घुसपैठ दर्ज की गई है।",
        "gu": "ચેતવણી! {camera_name} પર રાત્રિ દરમિયાન પ્રતિબંધિત વિસ્તારમાં ઘૂસણખોરી થઈ છે.",
        "en": "SECURITY ALERT! Perimeter intrusion detected at {camera_name}.",
    },
    "ABANDONED_OBJECT": {
        "hi": "सुरक्षा अलर्ट! {camera_name} पर लावारिस संदिग्ध वस्तु पाई गई है। बम निरोधक दस्ता सूचित करें।",
        "gu": "સુરક્ષા ચેતવણી! {camera_name} પર બિનવારસી શંકાસ્પદ વસ્તુ મળી આવી છે.",
        "en": "BOMB SQUAD ALERT! Unattended suspicious object detected at {camera_name}.",
    },
    "TRIPLE_RIDING_NO_HELMET": {
        "hi": "ट्रैफिक अलर्ट! {camera_name} पर बिना हेलमेट ट्रिपल राइडिंग वाहन {subject} चिन्हित हुआ।",
        "gu": "ટ્રાફિક ચેતવણી! {camera_name} પર હેલ્મેટ વિના ટ્રિપલ રાઇડિંગ કરતા વાહન {subject} ની નોંધ લેવાઈ.",
        "en": "TRAFFIC VIOLATION! Triple riding without helmet detected on vehicle {subject} at {camera_name}.",
    },
    "WRONG_WAY_HAZARD": {
        "hi": "खतरा! {camera_name} पर वाहन गलत दिशा में तेज गति से चल रहा है।",
        "gu": "જોખમ! {camera_name} પર વાહન ખોટી દિશામાં પૂરપાટ ઝડપે આવી રહ્યું છે.",
        "en": "ROAD HAZARD! Wrong-way high-speed vehicle moving towards {camera_name}.",
    },
    "CROWD_SURGE_ANOMALY": {
        "hi": "चेतावनी! {camera_name} पर भीड़ अचानक बढ़ रही है। अतिरिक्त बल तैनात करें।",
        "gu": "ચેતવણી! {camera_name} પર અચાનક ભારે ભીડ એકઠી થઈ રહી છે.",
        "en": "CROWD SURGE ANOMALY! Sudden high-density crowd accumulation at {camera_name}.",
    },
}

DEFAULT_PROMPTS = {
    "hi": "चेतावनी! {camera_name} पर गंभीर सुरक्षा घटना दर्ज की गई है।",
    "gu": "ચેતવણી! {camera_name} પર ગંભીર સુરક્ષા ઘટના બની છે.",
    "en": "SECURITY ALERT! Incident detected at {camera_name}. Response units dispatched.",
}


class VoiceAlertService:
    """Production Text-To-Speech engine with multilingual voice synthesis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or _AUDIO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_prompt(
        self,
        alert_type: str,
        camera_name: str = "SG Highway Junction",
        subject: str = "Suspect",
        lang: str = "hi",
    ) -> str:
        """Constructs a localized spoken alert script safely without str.format injection risks."""
        lang = lang.lower()
        if lang not in ["hi", "gu", "en"]:
            lang = "hi"

        templates = ALERT_TEMPLATES.get(alert_type, DEFAULT_PROMPTS)
        template = templates.get(lang, DEFAULT_PROMPTS.get(lang, ""))
        
        # Clean placeholders against literal brace injection
        safe_cam = str(camera_name or "Camera Station").replace("{", "").replace("}", "")
        safe_sub = str(subject or "Target Subject").replace("{", "").replace("}", "")
        return template.replace("{camera_name}", safe_cam).replace("{subject}", safe_sub)

    def synthesize_speech(
        self,
        text: str,
        lang: str = "hi",
        file_prefix: str = "alert",
    ) -> Path:
        """Synthesizes text into audio file (.mp3 or .wav) using multi-tier offline/online engines."""
        clean_prefix = "".join(c for c in file_prefix if c.isalnum() or c in "-_")[:32]
        out_mp3 = self.output_dir / f"{clean_prefix}_{lang}.mp3"
        out_wav = self.output_dir / f"{clean_prefix}_{lang}.wav"

        # Tier 0: reuse real speech already synthesised for this alert.
        #
        # Tier 1 is gTTS, which needs the network. Without this, an alert whose
        # audio was generated perfectly well an hour ago is re-synthesised on
        # every play, and the moment the connection drops it falls all the way
        # through to the chime — replacing a spoken Hindi broadcast with a beep
        # while an operator is listening. The text for a given alert and
        # language does not change, so the file does not need to either.
        #
        # Only an mp3 is reused: the wav is what the fallback chime writes, and
        # a cached chime must never mask a network that has come back.
        if out_mp3.exists() and out_mp3.stat().st_size > 1024:
            logger.debug("Reusing synthesised speech: %s", out_mp3.name)
            return out_mp3

        # Tier 1: Try gTTS if online (bounded with 2.5s network timeout)
        try:
            import socket
            orig_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(2.5)
                from gtts import gTTS
                tts_lang = "hi" if lang == "hi" else ("gu" if lang == "gu" else "en")
                tts = gTTS(text=text, lang=tts_lang, slow=False)
                tts.save(str(out_mp3))
                if out_mp3.exists() and out_mp3.stat().st_size > 0:
                    logger.info(f"Generated gTTS audio alert: {out_mp3.name}")
                    return out_mp3
            finally:
                socket.setdefaulttimeout(orig_timeout)
        except Exception as exc:
            logger.debug(f"gTTS online engine skipped ({exc}), trying local engine...")

        # Tier 2: Try pyttsx3 (Native Windows SAPI5 / Linux eSpeak)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.save_to_file(text, str(out_wav))
            engine.runAndWait()
            if out_wav.exists() and out_wav.stat().st_size > 0:
                logger.info(f"Generated pyttsx3 native audio alert: {out_wav.name}")
                return out_wav
        except Exception as exc:
            logger.debug(f"pyttsx3 local engine skipped ({exc}), generating synthesis chime...")

        # Tier 3: Deterministic multi-tone emergency chime generator (Zero dependencies)
        self._generate_emergency_chime(out_wav)
        return out_wav

    def _generate_emergency_chime(self, out_path: Path, duration_s: float = 2.0) -> None:
        """Generates a two-tone 880Hz / 440Hz police emergency chime in PCM WAV format."""
        sample_rate = 22050
        n_samples = int(sample_rate * duration_s)
        with wave.open(str(out_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                # Dual alert siren tone
                freq = 880.0 if (int(t * 4) % 2 == 0) else 587.33
                sample_val = int(16000.0 * math.sin(2.0 * math.pi * freq * t) * (1.0 - (t / duration_s) * 0.3))
                frames.extend(struct.pack("<h", max(-32767, min(32767, sample_val))))
            wav_file.writeframes(frames)


_voice_service = VoiceAlertService()


def get_voice_service() -> VoiceAlertService:
    return _voice_service
