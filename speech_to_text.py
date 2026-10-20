import os
from typing import BinaryIO

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


class SpeechToText:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "La variable ELEVENLABS_API_KEY est absente du fichier .env."
            )

        self.client = ElevenLabs(api_key=api_key)

    def transcribe(self, audio_file: BinaryIO) -> str:
        audio_file.seek(0)

        transcription = self.client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
            language_code="fra",
            tag_audio_events=False,
            diarize=False,
        )

        text = transcription.text.strip()
        if not text:
            raise RuntimeError("ElevenLabs n'a détecté aucune parole dans l'audio.")

        return text
