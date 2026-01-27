import modal
import subprocess
import os
import time
import urllib.request
import urllib.error
from difflib import get_close_matches

app = modal.App("vazenespravy-processor")
auth_secret = modal.Secret.from_name("vazene-spravy-api-key")
ollama_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)

PROMPTS = {
    "sk": (
        "You are a highly talented Slovak journalist, and you want to classify articles as either liberal, "
        "neutral, or conservative. In your responses, you can only list a single classification, "
        "and cannot list any other words. For example, if you classify an article as politically neutral, "
        "you will only say the word neutral. If you classify an article as liberal, you will only "
        "say the word liberal. If you classify an article as conservative, you will also only say the word "
        "conservative. Under no circumstances should you list more than a single classification in your "
        "responses, or a single word other than the classification. Classify the following article:"
    ),
    "en": (
        "You are a highly talented journalist, and you want to classify articles as either liberal, "
        "neutral, or conservative. In your responses, you can only list a single classification, "
        "and cannot list any other words. For example, if you classify an article as politically neutral, "
        "you will only say the word neutral. If you classify an article as liberal, you will only "
        "say the word liberal. If you classify an article as conservative, you will also only say the word "
        "conservative. Under no circumstances should you list more than a single classification in your "
        "responses, or a single word other than the classification. Classify the following article:"
    ),
}

BIASES = ["conservative", "liberal", "neutral"]
BIAS_TO_NUM = {"liberal": -1, "neutral": 0, "conservative": 1}
NUM_TO_BIAS = {-1: "liberal", 0: "center", 1: "conservative"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("ollama>=0.4.0", "numpy>=1.26", "scikit-learn==1.3.1", "joblib>=1.3", "fastapi[standard]")
    .env({"OLLAMA_MODELS": "/vol/ollama"})
    .add_local_file("logistic_model.joblib", "/app/logistic_model.joblib")
)

ollama_process = None


def start_ollama():
    """Start Ollama server and wait for it to be ready."""
    global ollama_process
    if ollama_process is not None:
        return

    print("Starting Ollama server...")
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/vol/ollama"

    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for i in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            print(f"Ollama ready after {i+1}s")
            return
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(1)

    if ollama_process.poll() is not None:
        _, stderr = ollama_process.communicate()
        raise RuntimeError(f"Ollama crashed: {stderr.decode()}")
    raise RuntimeError("Ollama failed to start within 60s")


def classify_with_model(model: str, text: str, prompt: str) -> int:
    """Run classification with a specific model."""
    import ollama  # type: ignore

    result = ollama.generate(model=model, prompt=prompt + text, options={"temperature": 0.1})
    response = result["response"].split("</think>")[-1].strip().lower()
    matches = get_close_matches(response, BIASES, n=1)
    return BIAS_TO_NUM[matches[0]] if matches else 0


# Individual model functions - each gets its own T4 GPU
@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=300,
)
def classify_gemma(text: str, language: str) -> int:
    start_ollama()
    prompt = PROMPTS.get(language, PROMPTS["en"])
    return classify_with_model("gemma3:12b", text, prompt)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=300,
)
def classify_qwen(text: str, language: str) -> int:
    start_ollama()
    prompt = PROMPTS.get(language, PROMPTS["en"])
    return classify_with_model("qwen3:8b", text, prompt)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol/ollama": ollama_volume},
    timeout=300,
    scaledown_window=300,
)
def classify_llama(text: str, language: str) -> int:
    start_ollama()
    prompt = PROMPTS.get(language, PROMPTS["en"])
    return classify_with_model("llama3.1:8b", text, prompt)


# Main endpoint - no GPU needed, just orchestrates the 3 model calls
@app.function(
    image=image,
    secrets=[auth_secret],
    timeout=360,
)
@modal.fastapi_endpoint(method="POST")
async def analyze(data: dict):
    """
    Classify text for political bias.

    Headers: Authorization: Bearer <api-key>
    Body: {"text": "...", "language": "sk"}
    Returns: {"politicalBias": "liberal" | "center" | "conservative"}
    """
    from fastapi import HTTPException  # type: ignore
    import numpy as np
    from joblib import load

    text = data.get("text", "")
    language = data.get("language", "sk")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")

    # Call all 3 models in parallel
    gemma_result = classify_gemma.spawn(text, language)
    qwen_result = classify_qwen.spawn(text, language)
    llama_result = classify_llama.spawn(text, language)

    # Wait for all results
    results = [
        gemma_result.get(),
        qwen_result.get(),
        llama_result.get(),
    ]

    # Load logistic regression model and predict
    lr_model = load("/app/logistic_model.joblib")
    predictions = np.array([results])
    label = lr_model.predict(predictions)[0]

    return {"politicalBias": NUM_TO_BIAS[label]}


@app.function(
    image=image,
    volumes={"/vol/ollama": ollama_volume},
    timeout=1800,
)
def download_models():
    """Pre-download models. Run once: modal run modal_app.py::download_models"""
    import ollama  # type: ignore

    start_ollama()

    for model in ["gemma3:12b", "qwen3:8b", "llama3.1:8b"]:
        print(f"Pulling {model}...")
        ollama.pull(model)
        print(f"Done: {model}")

    ollama_volume.commit()
    print("All models downloaded.")


@app.local_entrypoint()
def main():
    print("Downloading models...")
    download_models.remote()
    print("Done!")
