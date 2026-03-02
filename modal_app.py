import modal
import os
import sys

# Force unbuffered output for Modal logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    from fastapi import Header
except ImportError:
    # Fallback for local imports (fastapi only available in Modal container)
    def Header(default=""):
        return default


def log(msg: str):
    """Log message with flush for Modal visibility."""
    print(f"[DIAG] {msg}", flush=True)

app = modal.App("vazenespravy-processor")

# Secret for API authentication
auth_secret = modal.Secret.from_name("vazenespravy")

# Volume for storing HuggingFace model weights
models_volume = modal.Volume.from_name("hf-models", create_if_missing=True)

# Image with all dependencies for transformer inference
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "sentencepiece>=0.2",
        "protobuf>=3.20",
        "spacy>=3.7",
        "fastapi[standard]",
    )
    .run_commands("python -m spacy download xx_sent_ud_sm")
    .env({"HF_HOME": "/vol/models"})
)


@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    timeout=1800,
)
def download_models():
    """Pre-download models. Runs on every deployment to ensure fresh models."""
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    # force_download=True ensures we always get the latest version from HuggingFace
    # This is important when model weights or tokenizers are updated

    print("Downloading NLLB translation model...")
    AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", use_fast=False, force_download=True
    )
    AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", force_download=True
    )

    print("Downloading Political DEBATE politicalness classifier...")
    AutoTokenizer.from_pretrained(
        "mlburnham/Political_DEBATE_large_v1.0", force_download=True
    )
    AutoModelForSequenceClassification.from_pretrained(
        "mlburnham/Political_DEBATE_large_v1.0", force_download=True
    )

    print("Downloading DeBERTa political leaning classifier...")
    # Download tokenizer from the fine-tuned model itself (not the base model)
    AutoTokenizer.from_pretrained(
        "matous-volf/political-leaning-deberta-large", force_download=True
    )
    AutoModelForSequenceClassification.from_pretrained(
        "matous-volf/political-leaning-deberta-large", force_download=True
    )

    print("Committing models to volume...")
    models_volume.commit()
    print("All models downloaded and committed.")


class Analyzer:
    """Analyzer class for loading models and processing text."""

    def __init__(self):
        """Load all models into GPU memory."""
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )
        import spacy

        print("Loading spaCy sentence splitter...")
        self.nlp = spacy.load("xx_sent_ud_sm")

        print("Loading NLLB translation model...")
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M", local_files_only=True, use_fast=False
        )
        # Suppress tied weights warnings by loading with tie_word_embeddings config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "facebook/nllb-200-distilled-600M", local_files_only=True
        )
        config.tie_word_embeddings = False
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            local_files_only=True,
            config=config
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Political DEBATE politicalness classifier...")
        self.politicalness_pipeline = pipeline(
            "zero-shot-classification",
            model="mlburnham/Political_DEBATE_large_v1.0",
            device=0 if torch.cuda.is_available() else -1,
            local_files_only=True,
        )

        print("Loading DeBERTa political leaning classifier...")
        self.economic_pipeline = pipeline(
            "text-classification",
            model="matous-volf/political-leaning-deberta-large",
            # Let the pipeline use the model's own tokenizer (don't specify explicitly)
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=256,
            local_files_only=True,
        )

        # Log the model's actual label configuration
        log("=" * 60)
        log("ECONOMIC MODEL CONFIGURATION:")
        model_config = self.economic_pipeline.model.config
        log(f"  id2label: {getattr(model_config, 'id2label', 'NOT FOUND')}")
        log(f"  label2id: {getattr(model_config, 'label2id', 'NOT FOUND')}")
        log(f"  num_labels: {getattr(model_config, 'num_labels', 'NOT FOUND')}")
        log("=" * 60)

        print("All models loaded successfully.")

    def translate(self, text: str) -> str:
        """
        Translate text to English using NLLB.

        ALWAYS translates to English, even if input is already English.
        Source language defaults to Slovak (slk_Latn).

        Uses spaCy to split into sentences (NLLB has 512 token limit),
        translates each sentence individually, then rejoins.
        """
        # TODO: add language detection (e.g. fasttext lid) for automatic source language selection

        # Split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        log(f"Translation: Split into {len(sentences)} sentences")

        if not sentences:
            log("Translation: No sentences found, returning original text")
            return text

        translated_sentences = []
        for sentence in sentences:
            # Tokenize with source language (Slovak) and target language (English)
            inputs = self.nllb_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.nllb_model.device)

            # Generate translation
            # Slovak: slk_Latn, English: eng_Latn
            # Get the forced BOS token ID for English
            try:
                forced_bos_token_id = self.nllb_tokenizer.lang_code_to_id["eng_Latn"]
                log(f"Translation: Using lang_code_to_id, forced_bos_token_id={forced_bos_token_id}")
            except (AttributeError, KeyError) as e:
                # Fallback: convert the language code token to ID
                log(f"Translation: lang_code_to_id failed ({e}), using convert_tokens_to_ids")
                forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids("eng_Latn")
                log(f"Translation: Fallback forced_bos_token_id={forced_bos_token_id}")

            translated_tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
            )

            translated = self.nllb_tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            translated_sentences.append(translated)

        log(f"Translation: Completed {len(translated_sentences)} sentences")
        return " ".join(translated_sentences)

    def check_politicalness(self, text: str) -> tuple[bool, float]:
        """
        Check if text is political using Political DEBATE NLI classifier.

        Returns tuple of (is_political, score).
        """
        result = self.politicalness_pipeline(
            text,
            candidate_labels=["This text is about politics."],
            hypothesis_template="{}",
        )

        log(f"Politicalness raw result: {result}")

        # result["scores"][0] is the score for the hypothesis
        score = result["scores"][0] if result["scores"] else 0.0
        is_political = score > 0.5

        log(f"Politicalness score: {score}, threshold: 0.5, is_political: {is_political}")

        return is_political, score

    def classify_economic(self, text: str) -> tuple[str, dict]:
        """
        Classify political leaning on economic axis using DeBERTa.

        Returns: tuple of (mapped_label, debug_info)
        """
        result = self.economic_pipeline(text, truncation=True, max_length=256)

        log(f"Economic pipeline raw result: {result}")

        # DeBERTa model outputs: 0 = left, 1 = center, 2 = right
        # The pipeline returns label as "LABEL_0", "LABEL_1", or "LABEL_2"
        raw_label = result[0]["label"]
        raw_score = result[0]["score"]

        label_map = {
            "LABEL_0": "left",
            "LABEL_1": "center",
            "LABEL_2": "right",
        }

        mapped_label = label_map.get(raw_label, "center")

        log(f"Economic classification: raw_label='{raw_label}', score={raw_score}, mapped_to='{mapped_label}'")

        # Check if raw_label was in our map
        if raw_label not in label_map:
            log(f"WARNING: Unknown label '{raw_label}' not in label_map, defaulted to 'center'")

        debug_info = {
            "raw_label": raw_label,
            "raw_score": raw_score,
            "mapped_label": mapped_label,
            "label_was_mapped": raw_label in label_map,
        }

        return mapped_label, debug_info

    def process_text(self, text: str) -> dict:
        """
        Process text for political content and bias classification.

        Args:
            text: Input text to analyze

        Returns: {
            "politicalness": true/false,
            "axes": {"economic": "left|center|right"} or {}
        }
        """
        log("=" * 60)
        log("STARTING ANALYSIS")
        log(f"Input text length: {len(text)} chars")
        log(f"Input text preview: {text[:200]}...")

        # Step 1: Translate to English (always, even if already English)
        log("Step 1: Translating text...")
        translated_text = self.translate(text)
        log(f"Translated text length: {len(translated_text)} chars")
        log(f"Translated text preview: {translated_text[:300]}...")

        # Step 2: Check politicalness
        log("Step 2: Checking politicalness...")
        is_political, pol_score = self.check_politicalness(translated_text)

        # Step 3: If political, classify economic axis
        axes = {}
        economic_debug = None
        if is_political:
            log("Step 3: Classifying economic axis...")
            economic_label, economic_debug = self.classify_economic(translated_text)
            axes["economic"] = economic_label
        else:
            log("Step 3: SKIPPED - text not political, no economic classification")

        # Final summary
        log("-" * 40)
        log("ANALYSIS COMPLETE - SUMMARY:")
        log(f"  politicalness: {is_political} (score: {pol_score})")
        log(f"  axes: {axes}")
        if economic_debug:
            log(f"  economic_debug: {economic_debug}")
        log("=" * 60)

        return {"politicalness": is_political, "axes": axes}


# Global analyzer instance (loaded once per container)
analyzer = None

@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol/models": models_volume},
    secrets=[auth_secret],
    timeout=300,
    scaledown_window=180,
    env={"HF_HUB_OFFLINE": "1"},
    max_containers=1,
)
@modal.fastapi_endpoint(method="POST")
async def analyze(data: dict, authorization: str = Header(default="")):
    """
    Classify text for political content and bias.

    Headers: Authorization: Bearer <api-key>
    Body: {"text": "..."}
    Returns: {
        "politicalness": true/false,
        "axes": {"economic": "left|center|right"} or {}
    }
    """
    from fastapi import HTTPException

    # Validate authentication
    expected_key = os.environ.get("API_KEY", "")
    provided_key = (
        authorization.replace("Bearer ", "")
        if authorization.startswith("Bearer ")
        else authorization
    )
    if not expected_key or provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Validate input
    text = data.get("text", "")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Missing or empty 'text' field")

    # Lazy load analyzer on first request
    global analyzer
    if analyzer is None:
        analyzer = Analyzer()

    # Process text
    result = analyzer.process_text(text)
    return result


@app.local_entrypoint()
def main():
    """Local entrypoint for initial model download."""
    print("Downloading models to volume...")
    download_models.remote()
    print("Done!")
