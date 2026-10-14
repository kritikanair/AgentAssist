
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


def get_api_key():
    """Read the API key from Streamlit Secrets or local .env."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GROQ_API_KEY", "")


def chat(messages, temperature=0.2):
    """Send messages to the configured Groq model."""
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Configure it in "
            "Streamlit Secrets or your local .env file."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content or ""
