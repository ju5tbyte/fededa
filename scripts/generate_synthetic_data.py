"""Generate synthetic QA pairs from text chunks using LLM.

This script:
1. Loads text chunks from JSON files
2. Generates 0-N questions per chunk using DeepSeek API (N configurable via --max-questions)
3. Saves unfiltered results
4. Validates generated QA pairs using a validation prompt
5. Saves filtered results (PASS only) and validation results

Usage:
    # Run both generation and validation (default)
    python scripts/generate_synthetic_data.py <input_json_path>
    python scripts/generate_synthetic_data.py data/processed/digital_design_knowledge_source/an_animated_introduction_to_digital_logic_design.json
    python scripts/generate_synthetic_data.py <input_json_path> --max-questions 5

    # Run only generation phase
    python scripts/generate_synthetic_data.py <input_json_path> --phase generate
    python scripts/generate_synthetic_data.py <input_json_path> --phase generate --max-questions 3

    # Run only validation phase (use unfiltered JSON as input)
    python scripts/generate_synthetic_data.py <unfiltered_json_path> --phase validate

    # Resume from checkpoint (enabled by default when unfiltered results exist)
    # Use --no-resume to disable resume mode
    python scripts/generate_synthetic_data.py <input_json_path> --no-resume
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

# Prompt templates
QUESTION_GENERATION_PROMPT = """You are an expert professor in digital logic design, computer architecture, VLSI design, and electrical/computer engineering.

**Task:** Read the text chunk below and generate high-quality questions testing genuine understanding of these domains.

**Text Chunk:**
{text_chunk}

---

**RULES:**

**1. Content-Grounding**
- Every question MUST derive from substantive technical content (definitions, derivations, examples, analysis) actually present in the chunk.
- Do NOT generate questions from: tables of contents, headings, topic lists, index entries, references, or copyright notices.
- **Key test:** If the chunk only *names* a topic without *explaining* it → return empty array.
- **Exercise chunks:** Lists of task directives ("Implement X", "Show that Y", "Design Z") are NOT explanations. If the chunk is primarily exercises → return empty array. Exception: generate questions only from accompanying explanatory prose, if substantial.

**2. Standalone Principle**
Questions and answers must be fully understandable with domain knowledge alone, without the source text.

**FORBIDDEN references (in both Q and A — any occurrence → invalid):**
- "the text/author/passage/chapter/book"
- "Figure/Table/Diagram/Section/Chapter [number]"
- "as shown/described/illustrated in", "the example above", "the circuit in Figure"

The topic and depth come from the chunk; the phrasing must be self-contained.

**3. Difficulty Tiers**
- **easy:** Define a concept or recall a key property.
  *"What is the difference between a latch and a flip-flop?"*
- **medium:** Apply a concept, compare alternatives, or explain *why* something works.
  *"Why is dynamic power in CMOS proportional to switching frequency and V_DD²?"*
- **hard:** Multi-step reasoning, design/optimization, or cross-concept synthesis.
  *"Design a minimal two-level SOP for F(A,B,C) = Σm(1,2,5,6) using a K-map, and state the literal count."*

**4. Quality Requirements**
- **No duplicates:** Each question must test a distinct concept. Fewer questions > semantic duplicates.
- **Answer quality:** Technically correct, self-contained, concise but complete (2–6 sentences). Show key intermediate steps for derivations. Must provide concrete information beyond what the question states (no tautological answers).

**5. Forbidden Question Types**
- Questions about the textbook itself (chapter titles, what the author says)
- Questions dependent on figures, tables, or diagrams
- Pure glossary lookups (unless the concept is nuanced)
- Yes/No without justification
- Questions on topics merely *named* but not *explained* in the chunk

**6. Skip Noise**
Return empty array if the chunk is:
- Table of contents, index, references, copyright, or preface without technical content
- Primarily topic names/headings without substantive explanation
- Primarily an exercise/problem list without explanatory prose
- Insufficient technical depth for any question
- **When in doubt → empty array.**

---

**Self-Check Before Generating:**
1. Does this chunk *explain* something, or merely *list/name* topics? Latter → empty array.
2. Is it primarily an exercise list without explanatory prose? → empty array.
3. For each question: is the explanation in the chunk, or am I drawing on outside knowledge only? Outside only → skip.
4. Any Figure/Table/Section/Chapter reference in Q or A? → remove or discard.
5. Does any answer merely restate the question? → discard.

**Output Format (JSON):**
{{
  "questions": []
}}
OR (1–{max_questions} questions if valid content exists):
{{
  "questions": [
    {{
      "difficulty": "easy|medium|hard",
      "topic": "brief topic tag",
      "question": "...",
      "answer": "..."
    }}
  ]
}}

Generate 0–{max_questions} questions now:"""


VALIDATION_PROMPT = """You are a strict auditor for a digital logic / computer architecture / VLSI / EE-CE problem set.

Evaluate whether this QA pair is suitable for teaching domain knowledge.

**Original Text:**
{text_chunk}

**Question:** {question}
**Proposed Answer:** {answer}

---

**Evaluation Criteria:**

**1. Content-Grounding**
Does the original text contain **substantive technical explanation** (definitions, derivations, comparisons, worked examples, analysis) of the questioned concept?
- Exercise directives ("Implement X", "Show that Y") are NOT explanations.
- Headings + brief transitions without substantive content → **FAIL**.
- A topic being well-known does not exempt it; grounding must be in *this chunk's* explanatory content.
- **To PASS:** You must identify ≥2–3 sentences of genuine technical explanation (not task directives) supporting the question.

**2. Standalone**
Can Q and A be fully understood with domain knowledge alone, without the source text?

**FAIL if either Q or A contains any of:**
- "the text/author/passage/chapter/book"
- "Figure/Table/Diagram/Section/Chapter [number]"
- "as shown/described/illustrated/presented in"
- "the example above", "the previous example", "the circuit in Figure"

If understanding requires access to a specific figure, table, or textbook section → **FAIL**.

**3. Technical Correctness**
Is the answer factually accurate?
- Apply heightened scrutiny to derivations/equations. Example: A NAND B = ((A NAND B)')' is wrong; correct AND via NAND is (A NAND B) NAND (A NAND B).
- Wrong equation, incorrect minimization, erroneous derivation → **FAIL**.
- Errors, misleading simplifications, hallucinated facts → **FAIL**.

**4. Non-Triviality**
Does the question require meaningful domain knowledge or reasoning?
- Pure metadata ("What chapter covers X?") or common-sense-only questions → **FAIL**.

**5. Relevance**
Is the question about digital logic, computer architecture, VLSI, or closely related EE/CS topics?
- Off-topic → **FAIL**.

**6. Answer Completeness & Information Gain**
- Vague or incomplete answer → **FAIL**.
- Tautological answer that merely restates the question without adding specific technical information → **FAIL**.
  *Bad example: Q: "What does the 74 prefix indicate?" A: "It indicates a 7400 series chip."*
- The answer must contain ≥1 specific technical fact not already in the question.

---

**Output Format (JSON):**
{{
  "reason": "Explain which criteria passed/failed. For Content-Grounding: cite specific explanatory content found (or not). For Technical Correctness of derivations: show verification. For Standalone: list any forbidden references found.",
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
                k in q for k in ["difficulty", "topic", "question", "answer"]
            ):
                valid_questions.append(q)

        return valid_questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def validate_qa_pair(
    api_key: str,
    text_chunk: str,
    question: str,
    answer: str,
    model: str,
) -> dict:
    """Validate a single QA pair using deterministic inference (temperature=0)."""
    prompt = VALIDATION_PROMPT.format(
        text_chunk=text_chunk,
        question=question,
        answer=answer,
    )

    try:
        # Use temperature=0 for deterministic validation
        response = call_deepseek_api(
            api_key,
            prompt,
            model=model,
            temperature=0.0,
        )
        result = extract_json_from_response(response)

        # Ensure result has required fields
        if "result" not in result:
            result["result"] = "FAIL"
        if "reason" not in result:
            result["reason"] = "Failed to parse validation result"

        return result
    except Exception as e:
        return {
            "result": "FAIL",
            "reason": f"Error during validation: {str(e)}",
        }


def process_file(
    input_path: str,
    api_key: str,
    model: str = "deepseek-chat",
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
        api_key: OpenRouter API key
        model: Model to use
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
        return validate_only(input_path, api_key, model, delay, resume)

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

        # Generate questions
        questions = generate_questions_for_chunk(
            api_key, text_chunk, model, max_questions=max_questions
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
            unfiltered_results, api_key, model, delay, resume, source_path
        )
    else:
        validation_results = []
        filtered_results = []

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

    return {
        "total_chunks": processed,
        "total_questions_unfiltered": sum(
            len(r["questions"]) for r in unfiltered_results
        ),
        "total_validated": len(validation_results),
        "total_passed": len(filtered_results),
    }


def validate_only(
    unfiltered_path: str,
    api_key: str,
    model: str,
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
        api_key,
        model,
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
    api_key: str,
    model: str,
    delay: float,
    resume: bool,
    source_path: Optional[str] = None,
) -> tuple:
    """Run validation on unfiltered results.

    Args:
        unfiltered_results: List of chunk results with generated questions
        api_key: OpenRouter API key
        model: Model to use
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
                api_key,
                text_chunk,
                question_text,
                q["answer"],
                model,
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
        help="DeepSeek model to use (default: deepseek-chat)",
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
        choices=["all", "generate", "validate"],
        default="all",
        help="Phase to run: 'all' (generate + validate), 'generate' (questions only), "
        "'validate' (validation only, use unfiltered JSON as input). Default: all",
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

    # Process file
    print(f"Processing: {args.input_file}")
    print(f"Phase: {args.phase}")
    print(f"Model: {args.model}")
    print(f"Chunk range: {args.start_chunk} to {args.end_chunk or 'end'}")
    print(f"Max questions per chunk: {args.max_questions}")
    print(f"Resume: {args.resume}")
    print("=" * 50)

    result = process_file(
        args.input_file,
        api_key,
        model=args.model,
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
