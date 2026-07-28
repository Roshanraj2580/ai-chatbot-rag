"""
Real-time voice conversation module for continuous speech interaction.
Enables live voice conversation where user speaks and LLM responds with voice using RAG.
"""
import logging
import asyncio
import tempfile
import os
from typing import Optional
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)

# Define temporary upload directory
# Use absolute path to ensure it resolves correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")

# Ensure temp directory exists
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


class RealtimeVoiceConversation:
    """
    Real-time voice conversation handler.
    Manages continuous speech recognition and TTS response generation.
    """
    
    def __init__(self):
        """Initialize real-time voice conversation system."""
        self.recognizer = sr.Recognizer()
        # Optimized settings for real-time recognition
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause for faster response
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5
        logger.info("RealtimeVoiceConversation initialized")
    
    async def listen_stream(self, audio_stream: bytes) -> Optional[str]:
        """
        Transcribe audio stream in real-time.
        
        Args:
            audio_stream: Raw audio bytes (any format supported by pydub/ffmpeg)
        
        Returns:
            Transcribed text or None if failed
        """
        tmp_input = None
        tmp_output = None
        
        try:
            # Save input audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm', dir=TEMP_UPLOAD_DIR) as tmp_file:
                tmp_file.write(audio_stream)
                tmp_input = tmp_file.name
            
            # Convert to WAV format using pydub (supports WebM, MP3, etc.)
            logger.info(f"Converting audio from WebM/Opus to WAV for speech recognition")
            audio_segment = AudioSegment.from_file(tmp_input)
            
            # Convert to format expected by SpeechRecognition (16-bit PCM WAV, mono, 16kHz)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Export as WAV
            tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_UPLOAD_DIR).name
            audio_segment.export(tmp_output, format='wav')
            logger.info(f"Audio converted successfully, WAV size: {os.path.getsize(tmp_output)} bytes")
            
            # Now use SpeechRecognition with the WAV file
            with sr.AudioFile(tmp_output) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language='en-US')
            logger.info(f"Recognized: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Clean up temp files
            if tmp_input and os.path.exists(tmp_input):
                try:
                    os.unlink(tmp_input)
                except OSError:
                    pass
            if tmp_output and os.path.exists(tmp_output):
                try:
                    os.unlink(tmp_output)
                except OSError:
                    pass
    
    async def synthesize_stream(self, text: str) -> bytes:
        """
        Convert text to speech and return audio bytes.
        
        Args:
            text: Text to convert to speech
        
        Returns:
            Audio bytes (MP3 format)
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for TTS")
                return b''
            
            # Use gTTS for high-quality synthesis
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            audio_bytes = audio_buffer.read()
            logger.info(f"Synthesized {len(audio_bytes)} bytes of audio")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return b''
    
    async def process_conversation_turn(self, audio_stream: bytes, rag_callback) -> Optional[bytes]:
        """
        Process one turn of conversation: listen → query RAG → respond with voice.
        
        Args:
            audio_stream: Input audio bytes
            rag_callback: Async function to query RAG system
        
        Returns:
            Audio response bytes or None if failed
        """
        try:
            # Step 1: Transcribe user speech
            user_text = await self.listen_stream(audio_stream)
            if not user_text:
                return None
            
            logger.info(f"User said: {user_text}")
            
            # Step 2: Query RAG system
            answer_text = await rag_callback(user_text)
            if not answer_text:
                answer_text = "I don't know."
            
            logger.info(f"LLM response: {answer_text}")
            
            # Step 3: Synthesize response to speech
            response_audio = await self.synthesize_stream(answer_text)
            
            return response_audio
            
        except Exception as e:
            logger.error(f"Error processing conversation turn: {e}")
            return None


class ContinuousVoiceRecognizer:
    """
    Continuous voice recognition for real-time streaming.
    Detects when user starts and stops speaking.
    """
    
    def __init__(self):
        """Initialize continuous recognizer."""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        logger.info("ContinuousVoiceRecognizer initialized")
    
    async def start_listening(self, callback_on_speech):
        """
        Start continuous listening mode.
        
        Args:
            callback_on_speech: Async function called when speech is detected
        """
        try:
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("Listening for continuous speech...")
                
                # Continuous listening loop
                while True:
                    try:
                        audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=15)
                        
                        # Process in background
                        asyncio.create_task(self._process_audio(audio, callback_on_speech))
                        
                    except Exception as e:
                        logger.error(f"Error in listening loop: {e}")
                        await asyncio.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Error starting continuous listening: {e}")
    
    async def _process_audio(self, audio, callback):
        """Process recognized audio."""
        try:
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Continuous recognition: {text}")
            await callback(text)
        except sr.UnknownValueError:
            pass  # Ignore unrecognizable audio
        except Exception as e:
            logger.error(f"Error processing audio: {e}")


# Singleton instance
_conversation_instance: Optional[RealtimeVoiceConversation] = None


def get_realtime_conversation() -> RealtimeVoiceConversation:
    """Get or create singleton RealtimeVoiceConversation instance."""
    global _conversation_instance
    if _conversation_instance is None:
        _conversation_instance = RealtimeVoiceConversation()
    return _conversation_instance
