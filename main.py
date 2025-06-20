import sys, os, threading, queue, asyncio, subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import mss
from PyQt5.QtCore import QTimer, QBuffer, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel

import requests
import uuid
from google.cloud import texttospeech_v1
import google.generativeai as genai
from google.generativeai.types import Content, Part, Blob, LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig

import logging
logging.basicConfig(level=logging.INFO, # Default level
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiApp(QMainWindow):
    message_received_signal = pyqtSignal(str, str, str) # role, message_type (text/transcript), message
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Gemini Live API Integration')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.conversation_log = QTextEdit()
        self.conversation_log.setReadOnly(True)
        self.layout.addWidget(self.conversation_log)

        self.session_button = QPushButton('Start Session')
        self.session_button.clicked.connect(self.toggle_session)
        self.layout.addWidget(self.session_button)

        self.status_label = QLabel('Status: Not connected')
        self.layout.addWidget(self.status_label)

        self.last_gemini_text_response = "" # To store the latest text from Gemini

        # TTS UI
        self.tts_button = QPushButton('Synthesize Gemini Text to Audio')
        self.tts_button.clicked.connect(self.handle_tts_button)
        self.tts_button.setEnabled(False) # Disabled until Gemini provides text
        self.layout.addWidget(self.tts_button)

        # HeyGem UI
        self.heygem_avatar_video_path_label = QLabel('Path to HeyGem Avatar Video (e.g., avatar.mp4):')
        self.layout.addWidget(self.heygem_avatar_video_path_label)
        self.heygem_avatar_video_path_input = QLineEdit()
        self.heygem_avatar_video_path_input.setPlaceholderText("Enter path to silent avatar video for HeyGem")
        self.layout.addWidget(self.heygem_avatar_video_path_input)
        self.heygem_avatar_video_path_input.textChanged.connect(self._check_heygem_readiness)

        self.generate_heygem_video_button = QPushButton('Generate HeyGem Video')
        self.generate_heygem_video_button.clicked.connect(self.handle_generate_heygem_video)
        self.layout.addWidget(self.generate_heygem_video_button)
        self.generate_heygem_video_button.setEnabled(False) # Initially disabled

        self.heygem_status_label = QLabel('HeyGem Status: Idle')
        self.layout.addWidget(self.heygem_status_label)

        self.last_tts_audio_path = None
        self.is_session_active = False

        self.heygem_polling_timer = QTimer(self)
        self.heygem_polling_timer.timeout.connect(self.poll_heygem_status)
        self.current_heygem_task_code = None

        self.samplerate = 16000
        self.channels = 1
        self.dtype = 'int16'
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        self._audio_buffer = []

        self.audio_processing_timer = QTimer(self)
        self.audio_processing_timer.timeout.connect(self.process_audio_queue)

        self.screen_capture_timer = QTimer(self)
        self.screen_capture_timer.timeout.connect(self.capture_screen_frame)
        self.sct = None
        self.screen_capture_interval = 1000

        self.message_received_signal.connect(self._handle_received_message)

        self.gemini_client = None
        self.gemini_session = None
        self.google_cloud_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.google_cloud_location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        self.gemini_model_name = 'gemini-1.5-flash-preview-0514'

        self._loop_thread = None
        self._async_loop = None

        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and not self.check_gcloud_auth():
            # This append_message will be handled by _handle_received_message if it emits
            self.message_received_signal.emit("System", "system_message", "Warning: Google Cloud ADC might not be configured. See README for auth setup.")
            self.status_label.setText('Status: Warning - Check Authentication')

    def synthesize_text_to_audio(self, text_input, output_filename="gemini_tts_output.mp3"):
        try:
            # Instantiate a client
            client = texttospeech_v1.TextToSpeechClient()

            # Set the text input to be synthesized
            synthesis_input = texttospeech_v1.SynthesisInput(text=text_input)

            # Build the voice request, select the language code ("en-US") and the ssml voice gender ("neutral")
            # You can customize voice parameters here. For a standard voice:
            voice = texttospeech_v1.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Standard-C", # A standard, neutral voice
                ssml_gender=texttospeech_v1.SsmlVoiceGender.NEUTRAL,
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.MP3
            )

            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            self.message_received_signal.emit("System", "system_message", f"Synthesizing '{text_input[:30]}...' to audio.")
            response = client.synthesize_speech(
                request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
            )

            # The response's audio_content is binary.
            with open(output_filename, "wb") as out:
                out.write(response.audio_content)
                logger.info(f'Audio content written to file "{output_filename}"')
            self.message_received_signal.emit("System", "system_message", f'Audio content written to file "{output_filename}"')
            return output_filename
        except Exception as e:
            logger.error(f"TTS Synthesis failed: {e}", exc_info=True)
            self.message_received_signal.emit("Error", "error_message", f"TTS Synthesis failed: {e}")
            return None

    @pyqtSlot(str, str, str)
    def _handle_received_message(self, role, message_type, message):
        if message_type == "transcript":
            self.append_message(f"{role} (transcript)", message)
        elif message_type == "gemini_response":
            self.append_message(role, message)
            self.last_gemini_text_response = message
            self.tts_button.setEnabled(True)
        elif message_type == "system_message":
            self.append_message(role, message) # System messages also go to main log
        elif message_type == "error_message":
            self.append_message(role, message) # Errors also go to main log
            self.status_label.setText(f"Status: Error - {message[:30]}...")


    async def _receive_gemini_responses(self):
        self.message_received_signal.emit("System", "system_message", "Response listener started.")
        try:
            async for response in self.gemini_session.receive():
                if not self.is_session_active:
                    break

                if response.text:
                    self.message_received_signal.emit("Gemini", "gemini_response", response.text)

                if response.server_content:
                    if response.server_content.input_transcription and response.server_content.input_transcription.text:
                        self.message_received_signal.emit("User", "transcript", response.server_content.input_transcription.text)
                    if response.server_content.error:
                         self.message_received_signal.emit("API Error", "error_message", str(response.server_content.error))
        except Exception as e:
            if self.is_session_active:
                logger.error(f"Error receiving Gemini responses: {e}", exc_info=True)
                self.message_received_signal.emit("Error", "error_message", f"Receiving responses failed: {e}")
        finally:
            if self.is_session_active:
                 self.message_received_signal.emit("System", "system_message", "Response listener stopped.")

    def start_async_loop(self):
        if self._async_loop is None or not self._async_loop.is_running():
            self._async_loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self.run_loop, name="AsyncLoopThread", daemon=True)
            self._loop_thread.start()
            logger.info("Asyncio event loop started.")

    def run_loop(self):
        asyncio.set_event_loop(self._async_loop)
        try:
            self._async_loop.run_forever()
        except Exception as e:
            logger.error(f"Async loop error: {e}", exc_info=True)
        finally:
            if self._async_loop.is_running(): self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            # self._async_loop.close() # Close is called by run_forever completion
            logger.info("Asyncio event loop finished.")


    def stop_async_loop(self):
        if self._async_loop and self._async_loop.is_running():
            logger.info("Stopping asyncio event loop...")
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
            if self._loop_thread.is_alive():
                 logger.warning("Warning: Asyncio loop thread did not terminate cleanly.")
            self._loop_thread = None
        if self._async_loop and not self._async_loop.is_closed(): # Ensure loop is closed
            self._async_loop.close()
        self._async_loop = None
        logger.info("Asyncio event loop stopped and resources released.")


    def submit_async_task(self, coro):
        if self._async_loop and self._async_loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        else:
            logger.error("Async loop not running. Cannot submit task.")
            self.message_received_signal.emit("Error", "error_message", "Async loop not running for API call.")
            return None

    def check_gcloud_auth(self):
        try:
            process = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'], capture_output=True, text=True, check=True)
            return bool(process.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"gcloud auth check failed: {e}")
            return False

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    def start_audio_recording(self):
        try:
            self.message_received_signal.emit("System", "system_message", f'Starting audio: {self.samplerate}Hz, {self.channels}ch, {self.dtype}')
            sd.check_input_settings(samplerate=self.samplerate, channels=self.channels, dtype=self.dtype)
            self.audio_stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype=self.dtype, callback=self.audio_callback)
            self.audio_stream.start()
            self.audio_processing_timer.start(200)
            self.message_received_signal.emit("System", "system_message", 'Audio recording started.')
            return True
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}", exc_info=True)
            self.message_received_signal.emit("Error", "error_message", f'Audio start failed: {e}')
            if self.audio_stream:
                try:
                    if self.audio_stream.active: self.audio_stream.stop()
                    self.audio_stream.close()
                except Exception as close_e:
                    logger.error(f"Error closing audio stream during start_audio_recording cleanup: {close_e}", exc_info=True)
            self.audio_stream = None
            return False

    def stop_audio_recording(self):
        if self.audio_stream is not None:
            self.message_received_signal.emit("System", "system_message", 'Stopping audio recording...')
            self.audio_processing_timer.stop()
            try:
                if self.audio_stream.active: self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logger.error(f"Error stopping/closing audio stream: {e}", exc_info=True)
                self.message_received_signal.emit("Error", "error_message", f'Audio stop error: {e}')
            self.audio_stream = None
            self.process_audio_queue()
            self.message_received_signal.emit("System", "system_message", 'Audio recording stopped.')

    def process_audio_queue(self):
        if not self.is_session_active or not self.gemini_session:
            while not self.audio_queue.empty(): self.audio_queue.get()
            self._audio_buffer.clear()
            return

        if self.audio_queue.empty() and not self._audio_buffer: return
        while not self.audio_queue.empty(): self._audio_buffer.append(self.audio_queue.get())
        if not self._audio_buffer: return
        current_audio_segment = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        if current_audio_segment.size == 0: return

        pcm_bytes = current_audio_segment.tobytes()
        audio_blob = Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={self.samplerate}")

        async def send_audio_data():
            try:
                if self.gemini_session and self.is_session_active:
                    await self.gemini_session.send_realtime_input(audio=audio_blob)
                    logger.debug(f"Sent audio: {len(pcm_bytes)} bytes")
            except Exception as e:
                logger.error(f"Error sending audio data: {e}", exc_info=True)
                self.message_received_signal.emit("Error", "error_message", f"Sending audio failed: {str(e)[:100]}")

        if self.gemini_session and self.is_session_active:
            self.submit_async_task(send_audio_data())

    def start_screen_capture(self):
        try:
            self.message_received_signal.emit("System", "system_message", 'Starting screen capture...')
            self.sct = mss.mss()
            self.screen_capture_timer.start(self.screen_capture_interval)
            self.message_received_signal.emit("System", "system_message", 'Screen capture started.')
            return True
        except Exception as e:
            logger.error(f"Failed to start screen capture: {e}", exc_info=True)
            self.message_received_signal.emit("Error", "error_message", f'Screen capture failed: {e}')
            if self.sct: self.sct.close()
            self.sct = None
            return False

    def stop_screen_capture(self):
        if self.screen_capture_timer.isActive():
            self.message_received_signal.emit("System", "system_message", 'Stopping screen capture...')
            self.screen_capture_timer.stop()
        if self.sct:
            self.sct.close()
            self.sct = None

    def capture_screen_frame(self):
        if not self.sct or not self.is_session_active or not self.gemini_session: return
        try:
            sct_img = self.sct.grab(self.sct.monitors[1])
            q_image = QImage(sct_img.rgb, sct_img.width, sct_img.height, QImage.Format_RGB32)
            byte_array = QBuffer()
            byte_array.open(QBuffer.ReadWrite)
            if not q_image.save(byte_array, "JPEG", quality=75):
                logger.warning("Failed to encode screen frame to JPEG.")
                self.message_received_signal.emit("Error", "error_message", "Failed to encode screen to JPEG.")
                return
            frame_bytes = byte_array.data()
            byte_array.close()
            if not frame_bytes:
                logger.warning("JPEG encoding resulted in empty frame.")
                self.message_received_signal.emit("Error", "error_message", "JPEG encoding empty.")
                return

            video_blob = Blob(data=frame_bytes, mime_type="image/jpeg")
            async def send_video_data():
                try:
                    if self.gemini_session and self.is_session_active:
                        await self.gemini_session.send_realtime_input(video=video_blob)
                        logger.debug(f"Sent video frame: {len(frame_bytes)} bytes")
                except Exception as e:
                    logger.error(f"Error sending video data: {e}", exc_info=True)
                    self.message_received_signal.emit("Error", "error_message", f"Sending video failed: {str(e)[:100]}")
            if self.gemini_session and self.is_session_active:
                self.submit_async_task(send_video_data())
        except Exception as e:
            logger.error(f"Error capturing/processing screen frame: {e}", exc_info=True)
            # self.message_received_signal.emit("Error", "error_message", f"Screen capture processing failed: {e}") # Avoid too many errors

    def toggle_session(self):
        if not self.is_session_active: self.start_session()
        else: self.stop_session()

    def start_session(self):
        self.message_received_signal.emit("System", "system_message", 'Attempting to start session...')
        if not self.google_cloud_project:
            logger.error('GOOGLE_CLOUD_PROJECT environment variable not set.')
            self.message_received_signal.emit("Error", "error_message", 'GOOGLE_CLOUD_PROJECT env var not set.')
            self.status_label.setText('Status: Error - GCP Project Missing')
            return

        self.start_async_loop()
        if not self.gemini_client:
            try:
                self.message_received_signal.emit("System", "system_message", f'Init Gemini Client: {self.google_cloud_project}@{self.google_cloud_location}')
                self.gemini_client = genai.Client(vertexai=True, project=self.google_cloud_project, location=self.google_cloud_location)
                self.message_received_signal.emit("System", "system_message", 'Gemini Client initialized.')
            except Exception as e:
                logger.error(f'Client Init Failed: {e}', exc_info=True)
                self.message_received_signal.emit("Error", "error_message", f'Client Init Failed: {e}')
                self.status_label.setText('Status: Error - Client Init Failed')
                return

        audio_started = self.start_audio_recording()
        screen_capture_started = False
        if audio_started:
            screen_capture_started = self.start_screen_capture()
            if not screen_capture_started:
                 self.message_received_signal.emit("Error", "error_message", 'Screen capture failed. Stopping audio.')
                 self.stop_audio_recording()

        if audio_started and screen_capture_started:
            try:
                self.message_received_signal.emit("System", "system_message", f'Connecting to Gemini Live API: {self.gemini_model_name}...')
                live_config = LiveConnectConfig(response_modalities=["TEXT"], input_audio_transcription={})

                async def do_connect():
                    try:
                        if not self.gemini_client:
                             logger.error('Gemini client lost before async connection.')
                             self.message_received_signal.emit("Error", "error_message", 'Gemini client lost before async connection.')
                             self.status_label.setText('Status: Error - Client AIO Error')
                             return
                        self.gemini_session = await self.gemini_client.aio.live.connect(model=self.gemini_model_name, config=live_config)
                        self.message_received_signal.emit("System", "system_message", 'Successfully connected to Gemini Live API.')
                        self.is_session_active = True
                        self.session_button.setText('Stop Session')
                        self.status_label.setText('Status: Session Active (API Connected)')
                        self.submit_async_task(self._receive_gemini_responses())
                    except Exception as e:
                        logger.error(f'API Connect Failed: {e}', exc_info=True)
                        self.message_received_signal.emit("Error", "error_message", f'API Connect Failed: {e}')
                        self.status_label.setText('Status: Error - API Connection Failed')
                        self.stop_audio_recording()
                        self.stop_screen_capture()
                        self.is_session_active = False
                        self.session_button.setText('Start Session')

                future = self.submit_async_task(do_connect())
                if not future: # submit_async_task can return None if loop is not running
                    logger.error('Failed to schedule API connection task, async loop might not be running.')
                    self.message_received_signal.emit("Error", "error_message", 'Failed to schedule API connection task.')
                    self.status_label.setText('Status: Error - Task Schedule Failed')
                    self.stop_audio_recording()
                    self.stop_screen_capture()
                    self.is_session_active = False
                    self.session_button.setText('Start Session')
            except Exception as e: # Catch errors from immediate (non-async) setup before do_connect
                logger.error(f'API Init Failed (outer): {e}', exc_info=True)
                self.message_received_signal.emit("Error", "error_message", f'API Init Failed: {e}')
                self.status_label.setText('Status: Error - API Init Failed')
                self.stop_audio_recording()
                self.stop_screen_capture()
                self.is_session_active = False
                self.session_button.setText('Start Session')
        elif audio_started and not screen_capture_started: # Audio OK, Screen Fail
            self.is_session_active = False
            self.session_button.setText('Start Session')
            self.status_label.setText('Status: Failed - Screen Capture Error') # UI already updated by start_screen_capture
        else: # Audio Fail
            self.is_session_active = False
            self.session_button.setText('Start Session')
            self.status_label.setText('Status: Failed - Audio Error') # UI already updated by start_audio_recording

    def stop_session(self):
        self.message_received_signal.emit("System", "system_message", 'Stopping session...')
        self.is_session_active = False

        if self.gemini_session:
            logger.info('Closing Gemini Live API session...')
            # Placeholder for explicit EOS or close if needed by SDK in future.
            self.gemini_session = None
            self.message_received_signal.emit("System", "system_message", 'Gemini Live API session ended.')

        self.stop_audio_recording()
        self.stop_screen_capture()

        self.session_button.setText('Start Session')
        self.status_label.setText('Status: Session Stopped. Disconnected.')
        self.message_received_signal.emit("System", "system_message", 'Session stopped.')

    def append_message(self, sender, message):
        self.conversation_log.append(f'{sender}: {message}')
        self.conversation_log.verticalScrollBar().setValue(self.conversation_log.verticalScrollBar().maximum())

    def closeEvent(self, event):
        logger.info("Application closing...")
        self.message_received_signal.emit("System", "system_message", "Application closing...")
        if self.is_session_active:
            self.stop_session()
        self.stop_async_loop()
        super().closeEvent(event)

    def handle_tts_button(self):
        if self.last_gemini_text_response:
            self.message_received_signal.emit("System", "system_message", "TTS button clicked.")
            # Call the actual TTS function (already implemented in Step 2)
            # Example: self.synthesize_text_to_audio(self.last_gemini_text_response)
            # For now, just log and update status
            output_audio_file = self.synthesize_text_to_audio(self.last_gemini_text_response)
            if output_audio_file:
                self.heygem_status_label.setText(f"HeyGem Status: Audio ready at {output_audio_file}")
                self.last_tts_audio_path = output_audio_file
            else:
                self.heygem_status_label.setText("HeyGem Status: TTS failed.")
                self.last_tts_audio_path = None
        else:
            self.message_received_signal.emit("System", "system_message", "TTS button clicked, but no Gemini text available.")
            self.heygem_status_label.setText("HeyGem Status: No Gemini text for TTS.")
        self._check_heygem_readiness()

    def handle_generate_heygem_video(self):
        self.message_received_signal.emit("System", "system_message", "Generate HeyGem Video button clicked.")
        avatar_video_path = self.heygem_avatar_video_path_input.text()
        tts_audio_file_path = self.last_tts_audio_path
        if not tts_audio_file_path:
            self.heygem_status_label.setText("HeyGem Status: TTS audio is not available. Synthesize audio first.")
            self.message_received_signal.emit("Error", "user_error", "TTS audio is not available. Please synthesize audio from Gemini text first.")
            return

        if not avatar_video_path:
            self.heygem_status_label.setText("HeyGem Status: Avatar video path is missing.")
            self.message_received_signal.emit("Error", "user_error", "Avatar video path is missing for HeyGem.")
            return

        # Placeholder for Step 4 logic:
        self.heygem_status_label.setText(f"HeyGem Status: Preparing to generate video with {avatar_video_path} and {tts_audio_file_path}...")
        task_code = self.call_heygem_video_synthesis(tts_audio_file_path, avatar_video_path)
        if task_code:
            self.start_heygem_polling(task_code)
        else:
            # Error message already set by call_heygem_video_synthesis
            pass # Or set a generic "Submission failed, check logs."

    def call_heygem_video_synthesis(self, audio_path, avatar_video_path):
        # Note: Assumes HeyGem's /easy/submit API is running at http://127.0.0.1:8383.
        # The audio_path and avatar_video_path must be accessible by the HeyGem Docker service.
        try:
            self.heygem_status_label.setText("HeyGem Status: Submitting video synthesis request...")
            self.message_received_signal.emit("System", "heygem_status", "Submitting video synthesis request...")

            # Ensure paths are absolute or accessible by HeyGem's service
            # For local files, HeyGem docs imply D:\heygem_data as a base for some things.
            # However, its API examples for video synthesis just show "audio_url" and "video_url".
            # Assuming these can be local paths accessible by the HeyGem docker instances.
            # The user needs to ensure HeyGem services can access these paths.
            # For simplicity, we'll pass them as provided.

            task_code = str(uuid.uuid4()) # Generate a unique code for this task
            self.current_heygem_task_code = task_code # Store for polling

            payload = {
                "audio_url": audio_path,    # Path to the TTS audio file
                "video_url": avatar_video_path, # Path to the silent avatar video
                "code": task_code,          # Unique key for the task
                "chaofen": 0,               # Fixed value from HeyGem docs
                "watermark_switch": 0,      # Fixed value from HeyGem docs
                "pn": 1                     # Fixed value from HeyGem docs
            }

            # From HeyGem docs: http://127.0.0.1:8383/easy/submit
            heygem_submit_url = "http://127.0.0.1:8383/easy/submit"

            logger.info(f"Calling HeyGem submit API: {heygem_submit_url} with payload: {payload}")
            response = requests.post(heygem_submit_url, json=payload, timeout=30) # 30s timeout
            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            logger.info(f"HeyGem submit API response: {response_json}")

            # Expected response format (based on typical API design, HeyGem docs are minimal here):
            # {"code": 0, "message": "success", "data": {"taskCode": "your_task_code"}} or similar
            # Or if it directly returns the taskCode or a success status.
            # Let's assume success if no exception and check common success indicators.
            if response_json.get("code") == 0 or response_json.get("status") == "success" or response.status_code == 200:
                self.heygem_status_label.setText(f"HeyGem Status: Task submitted. Code: {task_code}. Polling...")
                self.message_received_signal.emit("System", "heygem_status", f"Task submitted. Code: {task_code}. Polling...")
                return task_code
            else:
                error_msg = response_json.get("message", "Unknown error from HeyGem submit API.")
                logger.error(f"HeyGem submit API returned an error: {error_msg} - Full response: {response_json}")
                self.heygem_status_label.setText(f"HeyGem Status: Submit failed - {error_msg}")
                self.message_received_signal.emit("Error", "heygem_error", f"Submit failed: {error_msg}")
                self.current_heygem_task_code = None
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"HeyGem submit API request failed: {e}", exc_info=True)
            self.heygem_status_label.setText(f"HeyGem Status: Submit request failed - {e}")
            self.message_received_signal.emit("Error", "heygem_error", f"Submit request error: {e}")
            self.current_heygem_task_code = None
            return None
        except Exception as e:
            logger.error(f"Error in call_heygem_video_synthesis: {e}", exc_info=True)
            self.heygem_status_label.setText(f"HeyGem Status: Error during submission - {e}")
            self.message_received_signal.emit("Error", "heygem_error", f"Submit error: {e}")
            self.current_heygem_task_code = None
            return None

    def start_heygem_polling(self, task_code):
        if task_code:
            self.current_heygem_task_code = task_code
            self.heygem_polling_timer.start(5000) # Poll every 5 seconds
            logger.info(f"Started polling for HeyGem task: {task_code}")
            self.heygem_status_label.setText(f"HeyGem Status: Polling for task {task_code}...")
        else:
            logger.warning("No task code provided to start polling.")
            self.heygem_status_label.setText(f"HeyGem Status: Invalid task code for polling.")


    def poll_heygem_status(self):
        # Note: Assumes HeyGem's /easy/query API is running at http://127.0.0.1:8383.
        # The JSON response structure (fields like 'status', 'data', 'video_url', 'progress')
        # is based on common API patterns and HeyGem's limited documentation.
        # This may need adjustment based on actual responses from a live HeyGem instance.
        if not self.current_heygem_task_code:
            logger.warning("Polling attempt without a task code.")
            self.heygem_polling_timer.stop()
            return

        try:
            task_code = self.current_heygem_task_code
            # From HeyGem docs: http://127.0.0.1:8383/easy/query?code=${taskCode}
            heygem_query_url = f"http://127.0.0.1:8383/easy/query?code={task_code}"

            logger.debug(f"Polling HeyGem status for task {task_code} at {heygem_query_url}")
            response = requests.get(heygem_query_url, timeout=10)
            response.raise_for_status()

            response_json = response.json()
            logger.debug(f"HeyGem query API response: {response_json}")

            # Based on typical API design, response might look like:
            # {"code": 0, "data": {"status": "processing/completed/failed", "video_url": "path/to/video.mp4"}}
            # The exact structure needs to be inferred or tested with a live HeyGem instance.
            # Assuming a structure like: {"status": "completed", "data": {"video_path": "..."}} or {"progress": "100%"}

            status = response_json.get("status", response_json.get("pro", "")).lower() # 'pro' for progress if 'status' not found
            data = response_json.get("data", {})

            if status == "completed" or status == "success" or str(data.get("progress")) == "100":
                self.heygem_polling_timer.stop()
                video_output_path = data.get("video_url", data.get("videoPath", data.get("outputPath", "Not provided")))
                self.heygem_status_label.setText(f"HeyGem Status: Video ready at {video_output_path}")
                self.message_received_signal.emit("System", "heygem_status", f"Video ready at {video_output_path}")
                logger.info(f"HeyGem task {task_code} completed. Video at: {video_output_path}")
                self.current_heygem_task_code = None
            elif status == "failed" or response_json.get("code") != 0 : # many APIs use code 0 for success
                self.heygem_polling_timer.stop()
                error_msg = data.get("message", "Task failed or error in response.")
                self.heygem_status_label.setText(f"HeyGem Status: Task {task_code} failed - {error_msg}")
                self.message_received_signal.emit("Error", "heygem_error", f"Task {task_code} failed: {error_msg}")
                logger.error(f"HeyGem task {task_code} failed: {error_msg} - Full response: {response_json}")
                self.current_heygem_task_code = None
            else:
                # Still processing
                progress_msg = f"Task {task_code} processing. Status: {status}, Progress: {data.get('progress', 'N/A')}"
                self.heygem_status_label.setText(f"HeyGem Status: {progress_msg}")
                logger.info(progress_msg)

        except requests.exceptions.RequestException as e:
            logger.error(f"HeyGem query API request failed for task {self.current_heygem_task_code}: {e}", exc_info=True)
            self.heygem_status_label.setText(f"HeyGem Status: Query request failed - {e}")
            # Optionally stop polling on repeated errors, or let it continue
        except Exception as e:
            logger.error(f"Error polling HeyGem status for task {self.current_heygem_task_code}: {e}", exc_info=True)
            self.heygem_status_label.setText(f"HeyGem Status: Error during polling - {e}")
            # Optionally stop polling
            self.heygem_polling_timer.stop()
            self.current_heygem_task_code = None

    def _check_heygem_readiness(self):
        tts_ready = bool(self.last_tts_audio_path)
        avatar_path_entered = bool(self.heygem_avatar_video_path_input.text().strip())

        if tts_ready and avatar_path_entered:
            self.generate_heygem_video_button.setEnabled(True)
        else:
            self.generate_heygem_video_button.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Potentially set app name for logger if it uses it by default
    # QCoreApplication.setApplicationName("GeminiLiveApp")
    # QCoreApplication.setOrganizationName("MyOrg")
    from PyQt5.QtWidgets import QLineEdit # Added QLineEdit here
    main_window = GeminiApp()
    main_window.show()
    logger.info("Application started.")
    sys.exit(app.exec_())
