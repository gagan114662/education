# Gemini Live API Integration Application

This application allows you to interact with Google's Gemini model using the Vertex AI Live API, enabling voice and screen sharing capabilities.

## Prerequisites

1.  **Python Environment:** Ensure you have Python 3.7+ installed. A virtual environment is recommended.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt # We will create this file later
    ```

2.  **Google Cloud Project:**
    *   A Google Cloud Platform project with billing enabled.
    *   The Vertex AI API must be enabled for your project. You can do this from the Google Cloud Console.

3.  **Authentication (Application Default Credentials - ADC):**
    *   Install the Google Cloud CLI: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
    *   Log in with your Google Cloud account and set up ADC:
        ```bash
        gcloud auth application-default login
        ```
    *   This command will open a browser window for you to authenticate. Once completed, your local environment will be authenticated to access Google Cloud services that your account has permissions for.
    *   Ensure the authenticated user/service account has the necessary IAM permissions for Vertex AI (e.g., "Vertex AI User" role).

## Running the Application

Once the prerequisites are met and dependencies are installed:

```bash
python main.py
```

---
*This README will be updated as more features are added.*
## HeyGem Integration (Experimental)

This application includes an experimental integration with HeyGem for video synthesis. This allows you to generate a video of an avatar speaking the text synthesized from Gemini's responses.

### HeyGem Setup (Assumed)

*   **HeyGem Docker Instances:** This integration assumes you have HeyGem's services (specifically `/easy/submit` and `/easy/query` APIs) running locally, typically via Docker. The application is configured to access these services at `http://127.0.0.1:8383`.
*   **File Accessibility:** For HeyGem to process your TTS audio and avatar video, these files must be accessible from within the HeyGem Docker containers. This usually means placing them in a directory that is mounted as a volume in your HeyGem Docker setup (e.g., a shared `D:\heygem_data` or similar, as hinted in some HeyGem documentation). You will need to provide the paths to these files as they are accessible from the Docker containers' perspective, or ensure the paths used by the application are resolvable by HeyGem.

### Usage

1.  **Synthesize Audio:** After receiving a text response from Gemini, click the "Synthesize Gemini Text to Audio" button. This will use Google Cloud Text-to-Speech to generate an MP3 file (e.g., `gemini_tts_output.mp3`).
2.  **Provide Avatar Video Path:** In the "Path to HeyGem Avatar Video" input field, enter the full path to a silent video of your avatar. This path must be accessible by your HeyGem Docker service.
3.  **Generate HeyGem Video:** Once both the TTS audio is ready and the avatar video path is entered, the "Generate HeyGem Video" button will become enabled. Click it to submit the task to the HeyGem API.
4.  **Polling and Output:** The application will poll the HeyGem API for the status of the video generation. Updates will be shown in the "HeyGem Status" label. Once completed, the path to the generated video will be displayed.

**Note:** This is an experimental feature. The HeyGem API interaction is based on available documentation and typical API behaviors. Adjustments to API endpoints or payload parameters in `main.py` might be necessary depending on your specific HeyGem setup or version.
