# Setup
 
Create a `.env` file in the project folder and add your ElevenLabs API key:
 
```dotenv
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```
 
With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run these commands from the project folder to create a virtual environment, install dependencies, and start the app:
 
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```
