# How to Run

Follow these instructions to start the Maternal Health Risk Expert System Web Interface.

## Prerequisites
1. Ensure you have Python 3.9+ installed.
2. The project directory must contain the following structure:
   - `models/`: Joblib files for the ML components (`nn1.joblib`, `nn2.joblib`, etc.)
   - `LLM/`: Local snapshots of the Qwen2.5-1.5B-Instruct weights.
   - `interface/`: The Flask web application.

## 1. Install Dependencies

Open your terminal in the `FinalProject` directory and install the requirements:

```bash
pip install -r requirements.txt
```

This will install the necessary ML packages (`scikit-learn`, `pandas`, etc.), the backend server (`flask`, `flask-cors`), and LLM-related libraries (`transformers`, `torch`, `accelerate`).

## 2. Start the Server

Navigate into the `interface` directory and start the Flask application:

```bash
cd interface
python app.py
```

### Note on Startup Time and Hardware
- **Model Loading:** The local Qwen2.5-1.5B-Instruct model is loaded into memory during startup. This takes **approximately 10-20 seconds** before the server is ready. The console will print `[LLM] Ready.` once it's loaded.
- **GPU vs CPU Mode:** The system automatically detects if an NVIDIA GPU (CUDA) is available.
  - If a GPU is detected, it loads the model in `float16` precision onto the GPU, which significantly speeds up response generation in the chat interface.
  - If no GPU is available, it gracefully falls back to `float32` on the CPU. CPU mode works perfectly fine but chat generation will be slower (streaming ~2-5 tokens per second instead of ~30+ tokens/s).
- **Graceful Fallback:** If the LLM weights are missing or corrupt, the server will *still* launch successfully. The LLM feature will be disabled (yielding a polite error message in chat), but the core Expert System and Diagnosis tools will remain fully functional.

## 3. Access the Application

Once the terminal outputs `* Running on http://127.0.0.1:5000`, open your web browser and navigate to:

**http://127.0.0.1:5000**

You can now run patient diagnoses, view the Chain of Evidence, download full PDF reports, and interact with the AI assistant.
