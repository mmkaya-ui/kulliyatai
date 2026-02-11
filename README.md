# Külliyat AI (Library Assistant)

A RAG (Retrieval-Augmented Generation) application that allows you to chat with your PDF library using AI models like Gemini and Claude.

## Features
-   **Chat with your PDFs**: Upload your documents and ask questions.
-   **Multi-Model Support**: Switch between Gemini Flash (Fast), Gemini Pro (Smart), and Claude 3.5 (Deep).
-   **Source Citations**: See exactly which page of which PDF the answer came from.
-   **Secure**: API keys are managed safely via Streamlit secrets or Environment Variables.
-   **Turkish UI**: Fully localized interface.

## Local Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mmkaya-ui/kulliyatai.git
    cd kulliyatai
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Secrets**:
    Create a `.streamlit/secrets.toml` file or a `.env` file with your API keys:
    ```toml
    OPENAI_API_KEY = "sk-..."
    GOOGLE_API_KEY = "AIza..."
    OPENROUTER_API_KEY = "sk-..."
    password = "your_secret_password"
    ```

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Deployment (Streamlit Community Cloud)

This is the recommended way to deploy the app for free.

1.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2.  Click **"New app"** and select this repository.
3.  **Advanced Settings**: Go to the "Secrets" section and paste your keys:
    ```toml
    OPENAI_API_KEY = "sk-..."
    GOOGLE_API_KEY = "AIza..."
    OPENROUTER_API_KEY = "sk-..."
    password = "your_secret_password"
    ```
4.  Click **Deploy**.
