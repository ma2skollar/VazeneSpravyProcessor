import os
from concurrent.futures import ThreadPoolExecutor
from difflib import get_close_matches
import numpy as np
from joblib import load
import ollama

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

MODELS = ["gemma3:12b", "qwen3:8b", "llama3.1:8b"]
BIASES = ["conservative", "liberal", "neutral"]
BIAS_TO_NUM = {"liberal": -1, "neutral": 0, "conservative": 1}
NUM_TO_BIAS = {-1: "liberal", 0: "center", 1: "conservative"}

# Cached model
_lr_model = None


def _get_lr_model():
    global _lr_model
    if _lr_model is None:
        path = os.path.join(os.path.dirname(__file__), "..", "logistic_model.joblib")
        if not os.path.exists(path):
            path = "/app/logistic_model.joblib"
        _lr_model = load(path)
    return _lr_model


def _get_classification(model: str, text: str, prompt: str) -> int:
    result = ollama.generate(model=model, prompt=prompt + text, options={"temperature": 0.1})
    response = result["response"].split("</think>")[-1].strip().lower()
    matches = get_close_matches(response, BIASES, n=1)
    return BIAS_TO_NUM[matches[0]] if matches else 0


def classify_text(text: str, language: str = "sk") -> dict:
    """Classify text for political bias."""
    prompt = PROMPTS.get(language, PROMPTS["en"])

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda m: _get_classification(m, text, prompt), MODELS))

    predictions = np.array([results])
    label = _get_lr_model().predict(predictions)[0]

    return {"politicalBias": NUM_TO_BIAS[label]}
