# Quandry Demo
Quandry is an open-source application for testing Large Language Model (LLM) model vulnerability against abuse attacks.

## Quick Start Guide
- We recommend using Docker to run Quandry.
- Replace ... in KEY_OPENAI=... and KEY_GEMINIAPI=... with your own API keys as necessary.
    - Example (not a valid key): KEY_OPENAI=uiwncisuibryalxu238hucwsiu97x2nu

### With Docker
- Ensure Docker Desktop is installed and open.
- Create a new file, ".env" and enter the following on lines 1 and 2 respectively
    - KEY_OPENAI=...
    - KEY_GEMINIAPI=...
- $ docker build -t quandry-demo . 
- $ docker run --env-file .env -p 8501:8501 quandry-demo
- Visit: http://localhost:8501

### Running Locally
- Ensure conda is installed and added to your PATH
- $ conda create -n quandry python=3.11
- $ conda activate quandry
- $ conda env config vars set KEY_OPENAI=...
- $ conda env config vars set KEY_GEMINIAPI=...
- $ conda deactivate
- $ conda activate quandry
- $ pip install -r requirements.txt
- $ streamlit run app/main.py
