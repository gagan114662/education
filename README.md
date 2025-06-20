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
