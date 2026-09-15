import asyncio
import logging
import os
import uuid
from typing import Optional

logger = logging.getLogger("sentinel.tts_engine")


class HindiTTSEngine:
    """
    Spoken Hindi TTS Engine & WhatsApp Notification Dispatcher
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.audio_dir = "data/audio_alerts"
        os.makedirs(self.audio_dir, exist_ok=True)

    async def generate_speech(self, text_hindi: str) -> Optional[str]:
        """Generates MP3 audio alert in spoken Hindi using gTTS"""
        output_file = os.path.join(self.audio_dir, f"alert_{uuid.uuid4().hex[:8]}.mp3")
        try:
            from gtts import gTTS
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: gTTS(text=text_hindi, lang="hi", slow=False).save(output_file)
            )
            return output_file
        except Exception as e:
            logger.debug(f"gTTS audio generation skipped: {e}")
            return None

    async def send_whatsapp_alert(self, to_phone: str, message: str) -> bool:
        """Sends WhatsApp dispatch message via Twilio API if configured"""
        sid = os.getenv("TWILIO_SID")
        token = os.getenv("TWILIO_TOKEN")
        from_num = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

        if not sid or not token:
            logger.info(f"WhatsApp Dispatch Simulated to {to_phone}: {message[:60]}...")
            return True

        try:
            from twilio.rest import Client
            client = Client(sid, token)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    body=message,
                    from_=from_num,
                    to=f"whatsapp:{to_phone}",
                ),
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to send WhatsApp alert: {e}")
            return False
