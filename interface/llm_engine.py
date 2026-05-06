"""
Loads Qwen2.5-1.5B-Instruct from the local snapshots directory.
Exposes stream_response(system_prompt, messages) as a generator of string tokens.
"""
import threading
from queue import Queue
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import torch
import os

# Base path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.path.join(
    BASE_DIR, 
    'LLM', 'Qwen2.5-1.5B-Instruct', 'snapshots', 
    '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
)

LLM_AVAILABLE = False
llm_lock = threading.Lock()

try:
    print(f"[LLM] Loading tokenizer from {SNAPSHOT}…")
    TOKENIZER = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)

    print("[LLM] Loading model…")
    MODEL = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
        trust_remote_code=True,
    )
    MODEL.eval()
    print("[LLM] Ready.")
    LLM_AVAILABLE = True
except Exception as e:
    print(f"[LLM] Failed to load model: {e}")
    LLM_AVAILABLE = False

# ── Generation config ─────────────────────────────────────────────────
GENERATION_KWARGS = {
    'max_new_tokens':  512,
    'temperature':     0.7,
    'top_p':           0.9,
    'repetition_penalty': 1.1,
    'do_sample':       True,
}

def stream_response(system_prompt: str, messages: list):
    """
    Yields string tokens one at a time (Server-Sent Events source).

    messages: list of {role: 'user'|'assistant', content: str}
    """
    if not LLM_AVAILABLE:
        yield "System Error: LLM is currently unavailable.\n\n"
        return

    # Build the chat template
    chat = [{'role': 'system', 'content': system_prompt}] + messages

    input_ids = TOKENIZER.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
    ).to(MODEL.device)

    streamer = TextIteratorStreamer(
        TOKENIZER,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        **GENERATION_KWARGS,
    )

    def generation_task():
        try:
            MODEL.generate(**gen_kwargs)
        except Exception as e:
            print(f"[LLM] Generation error: {e}")

    # Run generation in a background thread so we can yield tokens
    thread = threading.Thread(target=generation_task)
    thread.start()

    for token in streamer:
        yield token

    thread.join()

def generate_sync(system_prompt: str, messages: list) -> str:
    """
    Synchronous generation, primarily for the PDF report.
    Returns the complete generated string.
    """
    if not LLM_AVAILABLE:
        return "System Error: LLM is currently unavailable."
    
    # We acquire lock waiting if necessary, as this is a background report generation
    with llm_lock:
        chat = [{'role': 'system', 'content': system_prompt}] + messages
        input_ids = TOKENIZER.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(MODEL.device)

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

        outputs = MODEL.generate(**gen_kwargs)
        generated_ids = outputs[0][input_ids.shape[1]:]
        text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
        return text
