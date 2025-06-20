import sys, os, threading, queue, asyncio, subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import mss
from PyQt5.QtCore import QTimer, QBuffer, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel

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

        self.is_session_active = False

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

    @pyqtSlot(str, str, str)
    def _handle_received_message(self, role, message_type, message):
        if message_type == "transcript":
            self.append_message(f"{role} (transcript)", message)
        elif message_type == "gemini_response":
            self.append_message(role, message)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Potentially set app name for logger if it uses it by default
    # QCoreApplication.setApplicationName("GeminiLiveApp")
    # QCoreApplication.setOrganizationName("MyOrg")
    main_window = GeminiApp()
    main_window.show()
    logger.info("Application started.")
    sys.exit(app.exec_())
