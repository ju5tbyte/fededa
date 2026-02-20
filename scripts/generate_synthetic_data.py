"""Generate synthetic QA pairs from text chunks using LLM.

This script:
1. Loads text chunks from JSON files
2. Generates 0-N questions per chunk using DeepSeek API (N configurable via --max-questions)
3. Saves unfiltered results
4. Validates generated QA pairs using a validation prompt
5. Saves filtered results (PASS only) and validation results
6. Converts filtered results to finetune dataset format

Usage:
    # Run all phases: generation, validation, and finetune conversion (default)
    python scripts/generate_synthetic_data.py <input_json_path>
    python scripts/generate_synthetic_data.py data/processed/digital_design_knowledge_source/an_animated_introduction_to_digital_logic_design.json
    python scripts/generate_synthetic_data.py <input_json_path> --max-questions 5

    # Run only generation phase
    python scripts/generate_synthetic_data.py <input_json_path> --phase generate
    python scripts/generate_synthetic_data.py <input_json_path> --phase generate --max-questions 3

    # Run only validation phase (use unfiltered JSON as input)
    python scripts/generate_synthetic_data.py <unfiltered_json_path> --phase validate

    # Run only finetune conversion phase (use filtered JSON as input)
    python scripts/generate_synthetic_data.py <filtered_json_path> --phase finetune

    # Resume from checkpoint (enabled by default when unfiltered results exist)
    # Use --no-resume to disable resume mode
    python scripts/generate_synthetic_data.py <input_json_path> --no-resume

    # Use local GGUF model for validation (requires llama-cpp-python)
    python scripts/generate_synthetic_data.py <input_json_path> --validate-model local --model-path /path/to/model.gguf
"""

import json
import os
import sys
import argparse
import requests
import time
import re
from pathlib import Path
from typing import Optional

# Try to import dotenv for API key management
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# Try to import llama-cpp-python for local model support
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Prompt templates
QUESTION_GENERATION_PROMPT = """You are an expert professor in digital logic design,
computer architecture, VLSI design, and electrical/computer engineering.

**Task:** Read the text chunk below and generate high-quality questions that test
genuine understanding of these domains.

---

## SECTION 1 — Content Eligibility

Generate questions ONLY when the chunk contains substantive technical explanation:
definitions with reasoning, derivations, comparative analysis, worked examples,
or design trade-off discussions.

**Return empty array if the chunk is primarily:**
- Table of contents, index, references, copyright, or preface
- Headings and topic names without accompanying explanation
- Exercise/problem lists with no explanatory prose
- Fewer than ~5 sentences of genuine technical content

**Borderline chunks:** If the chunk explains at least one concept with enough depth
to form a non-trivial question, generate for that concept only.

---

## SECTION 2 — Question Types

Vary question types across the following categories. Do NOT cluster around one type.

| Type | Description | Example stem |
|------|-------------|--------------|
| **Concept** | Define or distinguish a concept | "What is the difference between…" |
| **Causal** | Explain the reason behind a behavior | "Why does… result in…" |
| **Quantitative** | Compute or derive a numerical value | "Calculate / Derive…" |
| **Transform** | Convert between representations (Boolean↔Gate↔HDL↔truth table) | "Convert the following SOP expression to a NAND-only circuit…" |
| **Trace** | Step through state/behavior over time (FSM, pipeline, waveform) | "After 3 clock edges, what is the value of Q…" |
| **Design** | Construct a circuit or system meeting given requirements | "Design a minimal… / Choose the best…" |
| **Optimize** | Minimize or maximize under explicit constraints | "Using a K-map, find the minimal SOP and minimal POS, and select whichever has fewer literals…" |
| **Trade-off** | Compare alternatives with justification | "Under what conditions would X outperform Y…" |
| **Fault** | Identify an error or predict its consequence | "A designer connects… What fault arises…" |

Each generated batch should include **at least two different types** when ≥3 questions
are generated.

---

## SECTION 3 — Difficulty Tiers

- **easy:** Recall or define a single concept with one key property.
- **medium:** Apply a concept, compare two alternatives, or explain causality.
- **hard:** Multi-step reasoning, optimization, cross-concept synthesis, or
  quantitative derivation with intermediate steps.

Aim for a roughly balanced spread across tiers when content supports it.

---

## SECTION 4 — Mandatory Quality Rules

**Standalone principle — FORBIDDEN in both Q and A:**
- "the text / author / passage / chapter / book / this example"
- "Figure / Table / Diagram / Section / Chapter [number or name]"
- "as shown / described / illustrated in"

**Answer requirements:**
- Technically correct and self-contained.
- Must add ≥1 specific technical fact not already stated in the question.
- For derivations: show key intermediate steps; do not skip to the final result.
- Length: 2–5 sentences for concept/causal; include equations/steps for
  quantitative and design questions (no artificial sentence cap).

**No semantic duplicates:** Distinct questions are preferred over overlapping ones.

---

## SECTION 5 — Self-Check (apply before finalizing output)

1. Does the chunk *explain*, or only *list/name*? Latter → empty array.
2. Is every question grounded in the chunk's explanatory content (not solely
   background knowledge)?
3. Do Q and A contain any forbidden references? → remove or discard.
4. Does the answer restate the question without adding information? → discard.
5. Are at least two question types represented (if ≥3 questions)?
6. Is the difficulty distribution reasonably balanced?

---

**Text Chunk:**
{text_chunk}

---

**Output Format (JSON only — no prose outside the JSON):**
{{
  "questions": []
}}
or
{{
  "questions": [
    {{
      "difficulty": "easy|medium|hard",
      "type": "concept|causal|quantitative|transform|trace|design|optimize|trade-off|fault",
      "topic": "brief topic tag (≤6 words)",
      "question": "...",
      "answer": "..."
    }}
  ]
}}

Generate 0–{max_questions} questions now:"""


VALIDATION_PROMPT = """You are a strict but fair auditor for a digital logic /
computer architecture / VLSI / EE-CE problem set.

Evaluate whether the QA pair below is suitable for teaching domain knowledge.

---

## Evaluation Criteria

### C1 — Content-Grounding
The original text must contain substantive technical explanation (definitions,
derivations, comparisons, worked examples, analysis) directly relevant to the
questioned concept.

PASS threshold: The text provides enough explanatory content that a student
reading only that text could meaningfully engage with the question. This does
not require exact sentence counting — a single dense equation with surrounding
explanation can suffice.

FAIL conditions:
- The text only names or lists the topic without explaining it.
- The chunk is purely an exercise directive ("Implement X", "Show that Y").
- The answer relies entirely on background knowledge with zero grounding in
  the chunk.

Note: Questions of type **transform** or **trace** often look like exercise
directives ("Convert…", "Trace…") but are valid if the chunk explains the
underlying concept or procedure being applied. Do not FAIL these on
Content-Grounding solely because of their imperative phrasing.

### C2 — Standalone
Q and A must be fully understandable with domain knowledge alone.

**Automatic FAIL if Q or A contains:**
- "the text / author / passage / chapter / book"
- "Figure / Table / Diagram / Section / Chapter [number]"
- "as shown / described / illustrated / presented in"
- "the example above", "the previous example", "the circuit in Figure"

### C3 — Technical Correctness
The answer must be factually accurate.

- Verify any equation or Boolean identity symbolically before accepting.
  Example check: to implement AND using NAND gates,
  A AND B = (A NAND B) NAND (A NAND B). Verify: NAND(x,x) = NOT x, so
  NAND(NAND(A,B), NAND(A,B)) = NOT(NAND(A,B)) = NOT(NOT(A AND B)) = A AND B. ✓
- Any incorrect derivation, wrong Boolean identity, or erroneous numerical
  result → FAIL.
- Misleading simplifications that would confuse a student → FAIL.

### C4 — Non-Triviality
The question must require genuine domain reasoning beyond common sense or
simple metadata lookup. Pure definitional questions are acceptable only if the
definition has technical nuance.

### C5 — Relevance
The question must concern digital logic, computer architecture, VLSI, or a
closely related EE/CS topic.

### C6 — Answer Information Gain
The answer must provide ≥1 specific technical fact not already present in the
question. A tautological answer (one that merely echoes the question) → FAIL.

---

**Original Text:**
{text_chunk}

**Question:** {question}
**Proposed Answer:** {answer}

---

## Output Format (JSON only)

{{
  "grounding_evidence": "Quote or paraphrase the specific explanatory content in the
    text that supports this question. Write 'NONE' if absent.",
  "correctness_check": "For any equation or derivation in the answer, show a brief
    symbolic verification. Write 'N/A' if no derivation is present.",
  "issues": "List any specific issues found, or 'None'.",
  "result": "PASS" or "FAIL"
}}

Provide your verdict:"""


def get_deepseek_api_key() -> str:
    """Get DeepSeek API key from environment or prompt user."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable not set. "
            "Please set your API key before running this script."
        )
    return api_key


# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class LocalLlamaModel:
    """Wrapper class for local GGUF models using llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: str = "llama-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize local llama.cpp model.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            chat_format: Chat format to use (llama-3, chatml, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

        print(f"Loading local model from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            verbose=False,
        )
        print(f"Local model loaded successfully")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response using local model.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (uses default if None)
            max_tokens: Max tokens to generate (uses default if None)

        Returns:
            Generated response content
        """
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        messages = [{"role": "user", "content": prompt}]

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response["choices"][0]["message"]["content"]

    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate chat completion using local model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (uses default if None)
            max_tokens: Max tokens to generate (uses default if None)

        Returns:
            Generated response content
        """
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response["choices"][0]["message"]["content"]


class ModelInterface:
    """Unified interface for both API and local models."""

    def __init__(
        self,
        model_type: str = "api",
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        local_model_path: Optional[str] = None,
        local_chat_format: str = "llama-3",
        local_n_ctx: int = 4096,
        local_n_gpu_layers: int = 0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize model interface.

        Args:
            model_type: "api" for DeepSeek API, "local" for llama.cpp
            api_key: API key for DeepSeek (required if model_type="api")
            model: Model name for API (used for API type)
            local_model_path: Path to GGUF model (required if model_type="local")
            local_chat_format: Chat format for local model
            local_n_ctx: Context size for local model
            local_n_gpu_layers: GPU layers for local model
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.model_type = model_type
        self._model = model
        self._api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        if model_type == "local":
            if local_model_path is None:
                raise ValueError(
                    "local_model_path is required when model_type is 'local'"
                )
            self.model_instance = LocalLlamaModel(
                model_path=local_model_path,
                n_ctx=local_n_ctx,
                n_gpu_layers=local_n_gpu_layers,
                chat_format=local_chat_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            self.model_instance = None
            if api_key is None:
                raise ValueError("api_key is required when model_type is 'api'")

    @property
    def api_key(self) -> str:
        """Get API key."""
        return self._api_key

    @property
    def model(self) -> str:
        """Get model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from model.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Generated response content
        """
        if self.model_type == "local":
            return self.model_instance.generate(prompt, temperature, max_tokens)
        else:
            return call_deepseek_api(
                self.api_key,
                prompt,
                model=self.model,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate chat completion from model.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Generated response content
        """
        if self.model_type == "local":
            return self.model_instance.chat(messages, temperature, max_tokens)
        else:
            # Convert to prompt for API
            prompt = self._messages_to_prompt(messages)
            return self.generate(prompt, temperature, max_tokens)

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert messages list to a single prompt string for API."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)


def call_deepseek_api(
    api_key: str,
    prompt: str,
    model: str = "deepseek-chat",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Call DeepSeek API with exponential backoff retry logic.

    Args:
        api_key: DeepSeek API key
        prompt: User prompt
        model: Model to use (e.g., deepseek-chat, deepseek-reasoner)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Maximum retry attempts

    Returns:
        Response content from the API
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_exception = None
    backoff = INITIAL_BACKOFF

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            # Check if we should retry
            if response.status_code in RETRY_STATUS_CODES:
                last_exception = Exception(
                    f"API call failed (retryable): {response.status_code} - {response.text}"
                )
            else:
                # Non-retryable error
                raise Exception(
                    f"API call failed: {response.status_code} - {response.text}"
                )

        except requests.exceptions.Timeout:
            last_exception = Exception("API call timed out")
        except requests.exceptions.RequestException as e:
            last_exception = Exception(f"API call error: {str(e)}")

        # Wait before retrying
        if attempt < max_retries:
            print(
                f"  Retry {attempt + 1}/{max_retries} after {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER

    # All retries exhausted
    raise last_exception or Exception("API call failed after retries")


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    # Try to find JSON block in response
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to parse the whole response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Return empty structure if parsing fails
    return {"questions": []}


def generate_questions_for_chunk(
    api_key: str,
    text_chunk: str,
    model: str,
    max_questions: int = 5,
) -> list:
    """Generate questions for a single text chunk."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        text_chunk=text_chunk,
        max_questions=max_questions,
    )

    try:
        response = call_deepseek_api(api_key, prompt, model=model)
        result = extract_json_from_response(response)

        questions = result.get("questions", [])
        # Validate questions have required fields
        valid_questions = []
        for q in questions:
            if all(
                k in q
                for k in ["difficulty", "type", "topic", "question", "answer"]
            ):
                valid_questions.append(q)

        return valid_questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def validate_qa_pair(
    model_interface: ModelInterface,
    text_chunk: str,
    question: str,
    answer: str,
) -> dict:
    """Validate a single QA pair using deterministic inference (temperature=0)."""
    prompt = VALIDATION_PROMPT.format(
        text_chunk=text_chunk,
        question=question,
        answer=answer,
    )

    try:
        # Use temperature=0 for deterministic validation
        response = model_interface.generate(
            prompt,
            temperature=0.0,
        )
        result = extract_json_from_response(response)

        # Ensure result has required fields
        if "grounding_evidence" not in result:
            result["grounding_evidence"] = "None"
        if "correctness_check" not in result:
            result["correctness_check"] = "N/A"
        if "issues" not in result:
            result["issues"] = "None"
        if "result" not in result:
            result["result"] = "FAIL"

        return result

    except Exception as e:
        print(f"Error validating QA pair: {e}")
        return {
            "grounding_evidence": "None",
            "correctness_check": "N/A",
            "issues": f"Validation error: {str(e)}",
            "result": "FAIL",
        }


def process_file(
    input_path: str,
    model_interface: ModelInterface,
    delay: float = 1.0,
    start_chunk: int = 0,
    end_chunk: Optional[int] = None,
    max_questions: int = 5,
    phase: str = "all",
    resume: bool = True,
) -> dict:
    """Process a single JSON file and generate QA pairs.

    Args:
        input_path: Path to input JSON file (or unfiltered JSON for validate phase)
        model_interface: Model interface for inference
        delay: Delay between API calls
        start_chunk: Starting chunk index
        end_chunk: Ending chunk index (exclusive)
        max_questions: Maximum questions per chunk
        phase: One of "all", "generate", "validate"
        resume: Whether to resume from checkpoint
    """
    input_file = Path(input_path)
    base_name = input_file.stem
    output_dir = input_file.parent

    # Store source path for validation phase
    source_path = str(input_file)

    # Paths
    unfiltered_path = output_dir / f"{base_name}_qa_unfiltered.json"
    validation_path = output_dir / f"{base_name}_qa_validation.json"
    filtered_path = output_dir / f"{base_name}_qa_filtered.json"

    # Handle validate-only phase
    if phase == "validate":
        return validate_only(input_path, model_interface, delay, resume)

    # Load input file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if end_chunk is None:
        end_chunk = len(chunks)

    # Load existing results for resume
    unfiltered_results = []
    if resume and unfiltered_path.exists():
        print(f"Loading existing unfiltered results from: {unfiltered_path}")
        with open(unfiltered_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            unfiltered_results = existing_data.get("chunks", [])
        print(f"Resumed with {len(unfiltered_results)} existing chunks")

    # Track processed chunk IDs
    processed_chunk_ids = {r["chunk_id"] for r in unfiltered_results}

    total_chunks = min(end_chunk, len(chunks)) - start_chunk
    processed = 0

    for i in range(start_chunk, min(end_chunk, len(chunks))):
        chunk = chunks[i]
        chunk_id = chunk.get("id", i)
        text_chunk = chunk.get("text", "")

        # Skip if already processed (resume)
        if chunk_id in processed_chunk_ids:
            print(f"Skipping chunk {chunk_id} (already processed, resume mode)")
            continue

        print(
            f"Processing chunk {chunk_id} ({i + 1 - start_chunk}/{total_chunks})..."
        )

        # Generate questions using API model (generation always uses DeepSeek API)
        # Note: Generation uses the model's API settings from model_interface
        questions = generate_questions_for_chunk(
            model_interface.api_key,
            text_chunk,
            model_interface.model,
            max_questions=max_questions,
        )

        if not questions:
            print(f"  -> No questions generated for chunk {chunk_id}")
            time.sleep(delay)
            continue

        # Store unfiltered results
        chunk_result = {
            "chunk_id": chunk_id,
            "text_chunk": text_chunk,
            "questions": questions,
        }
        unfiltered_results.append(chunk_result)
        processed_chunk_ids.add(chunk_id)

        # Save immediately after each chunk (prevent data loss on crash)
        with open(unfiltered_path, "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": unfiltered_results}, f, indent=2, ensure_ascii=False
            )
        print(f"  -> Saved {len(unfiltered_results)} chunks to unfiltered JSON")

        processed += 1
        time.sleep(delay)

    # Run validation if requested
    if phase in ("all", "validate"):
        validation_results, filtered_results = run_validation(
            unfiltered_results, model_interface, delay, resume, source_path
        )
    else:
        validation_results = []
        filtered_results = []

    # Run finetune conversion if requested
    finetune_result = None
    if phase in ("all", "finetune"):
        # Determine the filtered file path for finetune conversion
        if phase == "finetune":
            # In finetune phase, input_path is already the filtered file
            filtered_file_path = input_path
        else:
            # In all phase, use the filtered file from validation
            input_file = Path(input_path)
            base_name = input_file.stem
            output_dir = input_file.parent
            filtered_file_path = str(
                output_dir / f"{base_name}_qa_filtered.json"
            )

        if Path(filtered_file_path).exists():
            finetune_result = convert_to_finetune_format(filtered_file_path)
        else:
            print(
                f"Warning: Filtered file not found at {filtered_file_path}, skipping finetune conversion"
            )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total chunks processed: {processed}")
    print(
        f"Questions generated (unfiltered): {sum(len(r['questions']) for r in unfiltered_results)}"
    )
    print(f"Questions validated (total): {len(validation_results)}")
    print(f"Questions passed validation: {len(filtered_results)}")
    if validation_results:
        pass_rate = len(filtered_results) / len(validation_results) * 100
        print(f"Pass rate: {pass_rate:.1f}%")
    else:
        print("Pass rate: N/A (no validations)")
    if finetune_result:
        print(
            f"Questions converted to finetune format: {finetune_result['total_converted']}"
        )

    return {
        "total_chunks": processed,
        "total_questions_unfiltered": sum(
            len(r["questions"]) for r in unfiltered_results
        ),
        "total_validated": len(validation_results),
        "total_passed": len(filtered_results),
        "total_converted": (
            finetune_result["total_converted"] if finetune_result else 0
        ),
    }


def validate_only(
    unfiltered_path: str,
    model_interface: ModelInterface,
    delay: float,
    resume: bool,
) -> dict:
    """Run validation only on existing unfiltered results."""
    unfiltered_file = Path(unfiltered_path)
    output_dir = unfiltered_file.parent

    # Derive original input path from unfiltered path
    # e.g., "book_qa_unfiltered.json" -> "book.json"
    unfiltered_stem = unfiltered_file.stem
    if unfiltered_stem.endswith("_qa_unfiltered"):
        original_stem = unfiltered_stem.replace("_qa_unfiltered", "")
    else:
        # Fallback: try to handle other patterns
        original_stem = unfiltered_stem.replace("_qa_unfiltered", "")

    original_input_path = output_dir / f"{original_stem}.json"

    # Load unfiltered results
    with open(unfiltered_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unfiltered_results = data.get("chunks", [])
    print(f"Loaded {len(unfiltered_results)} chunks from unfiltered results")

    validation_results, filtered_results = run_validation(
        unfiltered_results,
        model_interface,
        delay,
        resume,
        str(original_input_path),
    )

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Questions validated (total): {len(validation_results)}")
    print(f"Questions passed validation: {len(filtered_results)}")
    if validation_results:
        pass_rate = len(filtered_results) / len(validation_results) * 100
        print(f"Pass rate: {pass_rate:.1f}%")
    else:
        print("Pass rate: N/A (no validations)")

    return {
        "total_questions_unfiltered": sum(
            len(r["questions"]) for r in unfiltered_results
        ),
        "total_validated": len(validation_results),
        "total_passed": len(filtered_results),
    }


def run_validation(
    unfiltered_results: list,
    model_interface: ModelInterface,
    delay: float,
    resume: bool,
    source_path: Optional[str] = None,
) -> tuple:
    """Run validation on unfiltered results.

    Args:
        unfiltered_results: List of chunk results with generated questions
        model_interface: Model interface for inference
        delay: Delay between API calls
        resume: Whether to resume from checkpoint
        source_path: Path to the original unfiltered results JSON file.
                     If None, derives path from unfiltered_results (legacy behavior).
    """
    # Use provided source_path or derive from unfiltered_results (legacy)
    if source_path:
        input_file = Path(source_path)
    else:
        # Legacy behavior: try to derive from chunk data
        # This is problematic because text_chunk contains text, not a path
        input_file = Path(".")

    base_name = input_file.stem
    output_dir = input_file.parent

    validation_path = output_dir / f"{base_name}_qa_validation.json"
    filtered_path = output_dir / f"{base_name}_qa_filtered.json"

    # Load existing validation results for resume
    validation_results = []
    filtered_results = []
    validated_questions = set()

    if resume and validation_path.exists():
        print(f"Loading existing validation results from: {validation_path}")
        with open(validation_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            validation_results = existing_data.get("validations", [])

        # Load filtered results
        if filtered_path.exists():
            with open(filtered_path, "r", encoding="utf-8") as f:
                existing_filtered = json.load(f)
                filtered_results = existing_filtered.get("questions", [])

        # Track already validated questions
        for vr in validation_results:
            key = (vr.get("chunk_id"), vr.get("question"))
            validated_questions.add(key)

        print(f"Resumed with {len(validation_results)} existing validations")

    # Validate each question
    for idx, chunk_result in enumerate(unfiltered_results):
        chunk_id = chunk_result["chunk_id"]
        text_chunk = chunk_result.get("text_chunk", "")

        for q in chunk_result.get("questions", []):
            question_text = q.get("question", "")

            # Skip if already validated (resume)
            if (chunk_id, question_text) in validated_questions:
                continue

            print(f"Validating chunk {chunk_id}: {question_text[:50]}...")

            validation = validate_qa_pair(
                model_interface,
                text_chunk,
                question_text,
                q["answer"],
            )

            validation_result = {
                "chunk_id": chunk_id,
                "question": question_text,
                "answer": q["answer"],
                "difficulty": q.get("difficulty"),
                "topic": q.get("topic"),
                "validation": validation,
            }
            validation_results.append(validation_result)
            validated_questions.add((chunk_id, question_text))

            if validation.get("result") == "PASS":
                filtered_results.append(
                    {
                        "chunk_id": chunk_id,
                        "difficulty": q.get("difficulty"),
                        "topic": q.get("topic"),
                        "question": question_text,
                        "answer": q["answer"],
                    }
                )

            time.sleep(delay)

            # Save immediately after each validation (prevent data loss on crash)
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"validations": validation_results},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(filtered_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"questions": filtered_results},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

                if len(validation_results) % 10 == 0:
                    print(
                        f"  -> Progress: {len(validation_results)} validations saved"
                    )

    return validation_results, filtered_results


def convert_to_finetune_format(filtered_json_path: str) -> dict:
    """Convert filtered QA pairs to finetune dataset format.

    Args:
        filtered_json_path: Path to filtered JSON file (e.g., *_qa_filtered.json)

    Returns:
        Dict with conversion statistics
    """
    filtered_file = Path(filtered_json_path)
    output_dir = filtered_file.parent

    # Derive finetune output path
    base_name = filtered_file.stem
    if base_name.endswith("_qa_filtered"):
        finetune_base = base_name.replace("_qa_filtered", "_finetune")
    else:
        finetune_base = base_name + "_finetune"

    finetune_path = output_dir / f"{finetune_base}.json"

    # Load filtered results
    with open(filtered_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_questions = data.get("questions", [])

    # Convert to finetune format
    finetune_data = []
    for q in filtered_questions:
        conversation = {
            "conversations": [
                {"from": "human", "value": q["question"]},
                {"from": "gpt", "value": q["answer"]},
            ]
        }
        finetune_data.append(conversation)

    # Save finetune format
    with open(finetune_path, "w", encoding="utf-8") as f:
        json.dump(finetune_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(finetune_data)} QA pairs to finetune format")
    print(f"Saved to: {finetune_path}")

    return {
        "total_converted": len(finetune_data),
        "finetune_path": str(finetune_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA pairs from text chunks"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSON file containing text chunks",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model to use for generation (default: deepseek-chat)",
    )
    parser.add_argument(
        "--validate-model",
        type=str,
        default="api",
        choices=["api", "local"],
        help="Model type for validation: 'api' uses DeepSeek API, 'local' uses GGUF model (default: api)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to GGUF model file (required if --validate-model local)",
    )
    parser.add_argument(
        "--chat-format",
        type=str,
        default="llama-3",
        help="Chat format for local model (default: llama-3). Options: llama-3, llama-2, chatml, gemma, etc.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context size for local model (default: 4096)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU for local model (default: -1, all layers)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="Starting chunk index (default: 0)",
    )
    parser.add_argument(
        "--end-chunk",
        type=int,
        default=None,
        help="Ending chunk index (exclusive, default: None = all)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum number of questions to generate per chunk (default: 5)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "generate", "validate", "finetune"],
        default="all",
        help="Phase to run: 'all' (generate + validate + finetune), 'generate' (questions only), "
        "'validate' (validation only, use unfiltered JSON as input), "
        "'finetune' (convert filtered JSON to finetune format). Default: all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if exists (default: True). Use --no-resume to disable",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode",
    )
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or get_deepseek_api_key()

    # Create model interface
    if args.validate_model == "local":
        if not args.model_path:
            raise ValueError(
                "--model-path is required when --validate-model local. "
                "Please specify the path to your GGUF model file."
            )
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        print(f"Using local GGUF model for validation: {args.model_path}")
        model_interface = ModelInterface(
            model_type="local",
            api_key=api_key,
            model=args.model,
            local_model_path=args.model_path,
            local_chat_format=args.chat_format,
            local_n_ctx=args.n_ctx,
            local_n_gpu_layers=args.n_gpu_layers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        # Use API for validation
        model_interface = ModelInterface(
            model_type="api",
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # Process file
    print(f"Processing: {args.input_file}")
    print(f"Phase: {args.phase}")
    print(f"Generation model: {args.model}")
    print(f"Validation model: {args.validate_model}")
    if args.validate_model == "local":
        print(f"Local model path: {args.model_path}")
        print(f"Chat format: {args.chat_format}")
        print(f"Context size: {args.n_ctx}")
    print(f"Chunk range: {args.start_chunk} to {args.end_chunk or 'end'}")
    print(f"Max questions per chunk: {args.max_questions}")
    print(f"Resume: {args.resume}")
    print("=" * 50)

    result = process_file(
        args.input_file,
        model_interface,
        delay=args.delay,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        max_questions=args.max_questions,
        phase=args.phase,
        resume=args.resume,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
