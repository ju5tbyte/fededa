#!/usr/bin/env python3
"""
Digital Design Knowledge Dataset Preprocessing Script

This script processes PDF documents containing digital design knowledge
and generates QA datasets through a 3-step pipeline:

1. Parse PDFs using LlamaParse with semantic chunking
2. Generate QA pairs (0 to 5 per chunk) using vLLM with Llama 3.3 70B AWQ
   - Only substantive technical knowledge generates questions
   - Text-only context: no figure-dependent questions
3. Filter QA pairs for consistency using vLLM validation

Usage:
    python scripts/preprocess_dataset_digital_design_knowledge.py \
    --input-pdf data/raw/digital_design_knowledge/my_doc.pdf \
    --step all \
    --bounding-box "0.1,0,0.2,0" \
    --output-chunks-dir data/processed/my_chunks \
    --output-qa-dir data/processed/my_qa \
    --output-filtered-dir data/processed/my_qa_filtered \
    --model casperhansen/llama-3.3-70b-instruct-awq \
    --tensor-parallel-size 4 \
    --temperature 0.8 \
    --max-tokens 4096 \
    --skip-existing

    python scripts/preprocess_dataset_digital_design_knowledge.py \
    --step parse \
    --bounding-box "0.1,0,0.1,0" \
    --input-pdf data/raw/digital_design_knowledge/specific_document.pdf

    python scripts/preprocess_dataset_digital_design_knowledge.py \
    --step generate \
    --input-pdf data/raw/digital_design_knowledge/specific_document.pdf \
    --tensor-parallel-size 2 \
    --temperature 0.7 \
    --max-tokens 4096

    python scripts/preprocess_dataset_digital_design_knowledge.py \
    --step filter \
    --input-pdf data/raw/digital_design_knowledge/specific_document.pdf \
    --tensor-parallel-size 2


"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownNodeParser

from llama_index.core import Document
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 16384

QUESTION_GENERATION_PROMPT = """You are an expert in digital design and electrical engineering.
Based on the provided text chunk, generate 0 to 5 questions.

**Text Chunk:**
{text_chunk}

**Strict Constraints (Read Carefully):**
1. **No External Knowledge:** Generate questions ONLY if the answer is explicitly found within the text chunk. Do not define terms based on your own knowledge if the definition is not in the text.
2. **Text-only context:** This chunk is from a PDF parse. Figures are missing. DO NOT generate questions asking about visual elements (e.g., "What does Figure 3 show?", "Draw the circuit").
3. **Avoid Trivial Metadata Questions:** DO NOT ask about the document structure.
   - BAD: "What is the topic of Chapter 5?", "What is covered in section 2.3?"
   - GOOD: "How does a D-latch store state?", "What is the difference between static and dynamic memory?"
4. **Skip Noise:** Return an empty array if the chunk contains:
   - Table of Contents, Lists of Figures, or Indices (lists of keywords without definitions)
   - Copyright notices, headers, footers, or references

**Question-Answer Generation Guidelines:**
- Questions must be self-contained and answerable **solely** from the provided text.
- If the text is an Index or Glossary list (e.g., "DeMorgan's Law... 22, 32"), DO NOT generate a question like "What is DeMorgan's Law?" because the definition is missing.
- Answers should be concise, accurate, and extracted from the text.

**Number of Questions to Generate:**
- Generate 0 if the text is low quality, noise, or purely metadata.
- Otherwise, generate 1-5 diverse questions.

**Output Format (JSON):**
{{
  "questions": []  // Return empty if content is not suitable based on constraints
}}

OR (if valid content exists, depending on the chunk, you might generate 1~5 questions):

{{
  "questions": [
    {{
      "difficulty": "easy|medium|hard",
      "question": "...",
      "answer": "..."
    }},
    {{
      "difficulty": "easy|medium|hard",
      "question": "...",
      "answer": "..."
    }},
    {{
      "difficulty": "easy|medium|hard",
      "question": "...",
      "answer": "..."
    }}
  ]
}}

Generate question and answer now:"""

VALIDATION_PROMPT = """You are a strict auditor for an educational RAG dataset.
Evaluate whether the following QA pair is **grounded solely** in the provided text.

**Original Text:**
{text_chunk}

**Question:** {question}
**Proposed Answer:** {answer}

**Strict Evaluation Criteria (If any is violated, return FAIL):**
1. **Grounding Check:** Does the text explicitly contain the answer? If the answer requires external knowledge (e.g., the text is just a keyword list like an Index, but the answer defines the term), you MUST return FAIL.
2. **Visual Dependency:** Does the question ask about a Figure, Image, or Diagram that is not visible in the text? (e.g., "What is in Figure 2-1?") -> FAIL.
3. **Triviality Check:** Is the question asking about metadata rather than content? (e.g., "What is the title of this section?", "What page is this?") -> FAIL.
4. **Accuracy:** Is the answer consistent with the provided text?

**Output Format (JSON):**
First, provide your reasoning for the validation decision. Then, provide the final result.
{{
  "reason": "Detailed explanation of why this QA pair passes or fails validation. Be specific about which criteria were checked and why.",
  "result": "PASS" or "FAIL"
}}

Provide your verdict:"""


# JSON schema for guided generation - Validation
VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Detailed explanation for the validation decision",
        },
        "result": {
            "type": "string",
            "enum": ["PASS", "FAIL"],
        },
    },
    "required": ["reason", "result"],
}


# JSON schema for guided generation - Question Generation
# Updated: questions array can be empty (0 questions) or have up to 5 questions
QUESTION_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["difficulty", "question", "answer"],
            },
        }
    },
    "required": ["questions"],
}


def parse_pdf_with_llamaparse(
    pdf_path: Path, bounding_box: str | None = None
) -> List[Dict]:
    """
    Parse PDF using LlamaParse with markdown-based chunking.

    Args:
        pdf_path: Path to the PDF file
        bounding_box: Optional bounding box margins in clockwise order from top
            (top, right, bottom, left) as comma-separated fractions (0-1).
            Example: "0.1,0,0.2,0" excludes top 10% and bottom 20%.

    Returns:
        List of chunk dictionaries with metadata
    """
    logger.info(f"Parsing PDF: {pdf_path}")

    try:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "LLAMA_CLOUD_API_KEY 환경변수가 설정되지 않았습니다."
            )

        parser_kwargs = {
            "api_key": api_key,
            "result_type": "markdown",
            "verbose": True,
            "language": "en",
            "high_res_ocr": True,
            "user_prompt": "Extract the main content of the document Do not include page headers, page numbers, or footers.",
        }

        if bounding_box:
            parser_kwargs["bounding_box"] = bounding_box
            logger.info(f"Using bounding_box: {bounding_box}")

        parser = LlamaParse(**parser_kwargs)

        logger.debug(f"LlamaParse initialized with result_type=markdown")

        # Parse the PDF - get markdown for entire document
        documents = parser.load_data(str(pdf_path))
        logger.debug(
            f"Raw documents from LlamaParse: {len(documents)} documents"
        )

        # Merge all pages into a single markdown document
        full_markdown = "\n\n".join([doc.text for doc in documents])
        logger.debug(f"Merged markdown length: {len(full_markdown)} characters")

        # Create a single Document with the full markdown
        llama_doc = Document(
            text=full_markdown,
            metadata={
                "source_file": pdf_path.name,
                "page_count": len(documents),
                "char_count": len(full_markdown),
            },
        )

        # Apply markdown-based chunking using MarkdownNodeParser
        splitter = MarkdownNodeParser()

        logger.debug("Applying markdown-based chunking...")
        nodes = splitter.get_nodes_from_documents([llama_doc])
        logger.debug(f"Markdown chunking produced {len(nodes)} chunks")

        # Convert back to our chunk format
        chunks = []
        for i, node in enumerate(nodes):
            chunk = {
                "id": i,
                "text": node.get_content(),
                "metadata": {
                    "source_file": pdf_path.name,
                    "chunk_index": i,
                    "char_count": len(node.get_content()),
                    "node_type": type(node).__name__,
                },
            }
            chunks.append(chunk)

        logger.info(f"Extracted {len(chunks)} markdown chunks from {pdf_path}")

        # Merge chunks by section to fix page boundary issues
        merged_chunks = merge_chunks_by_section(chunks)
        return merged_chunks

    except Exception as e:
        logger.error(f"Failed to parse PDF {pdf_path}: {e}")
        logger.debug(f"Exception details: {type(e).__name__}")
        return []


def extract_section_header(text: str) -> str | None:
    """Extract section header from text.

    Args:
        text: Text content

    Returns:
        Section header if found, None otherwise
    """
    stripped = text.strip()

    # Remove leading # if present
    if stripped.startswith("#"):
        stripped = stripped[1:].strip()

    # Match section headers like "6.1", "6.2.1", "7.4.1", "2.", "2.3.", etc.
    # Pattern: numbers and dots at the start, with optional trailing dot
    match = re.match(r"^(\d+(?:\.\d+)*\.?)", stripped)
    if match:
        return match.group(1)

    # Match headers like "Chapter 2.3.4" or "Chapter 2" or "Chapter 2."
    match = re.match(r"^Chapter\s+(\d+(?:\.\d+)*\.?)", stripped, re.IGNORECASE)
    if match:
        return match.group(1)

    # Match headers like "2 Introduction" or "2.3.4 Section Title"
    match = re.match(r"^(\d+(?:\.\d+)*\.?)\s+", stripped)
    if match:
        return match.group(1)

    return None


def merge_chunks_by_section(chunks: List[Dict]) -> List[Dict]:
    """Merge chunks that belong to the same section.

    Chunks without section headers are merged with the previous section.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List of merged chunk dictionaries
    """
    if not chunks:
        return []

    merged_chunks = []
    current_section = None
    current_text = ""
    current_chunk_ids = []

    for chunk in chunks:
        chunk_id = chunk["id"]
        text = chunk["text"]
        section_header = extract_section_header(text)

        logger.debug(
            f"Chunk {chunk_id}: section_header={section_header}, len(text)={len(text)}"
        )

        if section_header:
            # This chunk starts a new section
            # Save the previous section if it exists
            if current_text:
                merged_chunk = {
                    "id": len(merged_chunks),
                    "text": current_text.strip(),
                    "metadata": {
                        "source_file": chunks[0]["metadata"].get(
                            "source_file", "unknown"
                        ),
                        "chunk_index": len(merged_chunks),
                        "char_count": len(current_text.strip()),
                        "node_type": "MergedNode",
                        "original_chunk_ids": current_chunk_ids,
                        "section": current_section,
                    },
                }
                merged_chunks.append(merged_chunk)
                logger.debug(
                    f"Merged section '{current_section}' with {len(current_chunk_ids)} chunks, "
                    f"{len(current_text)} chars"
                )

            # Start a new section
            current_section = section_header
            current_text = text
            current_chunk_ids = [chunk_id]
        else:
            # This chunk belongs to the previous section
            if current_text:
                # Add separator if needed
                if not current_text.endswith("\n"):
                    current_text += "\n"
                current_text += text
                current_chunk_ids.append(chunk_id)
            else:
                # First chunk without header - treat as its own section
                current_section = "unknown"
                current_text = text
                current_chunk_ids = [chunk_id]

    # Don't forget the last section
    if current_text:
        merged_chunk = {
            "id": len(merged_chunks),
            "text": current_text.strip(),
            "metadata": {
                "source_file": chunks[0]["metadata"].get(
                    "source_file", "unknown"
                ),
                "chunk_index": len(merged_chunks),
                "char_count": len(current_text.strip()),
                "node_type": "MergedNode",
                "original_chunk_ids": current_chunk_ids,
                "section": current_section,
            },
        }
        merged_chunks.append(merged_chunk)
        logger.debug(
            f"Merged final section '{current_section}' with {len(current_chunk_ids)} chunks, "
            f"{len(current_text)} chars"
        )

    logger.info(
        f"Merged {len(chunks)} chunks into {len(merged_chunks)} sections, "
        f"reduction: {len(chunks) - len(merged_chunks)}"
    )
    return merged_chunks


def save_chunks_to_json(chunks: List[Dict], output_path: Path) -> None:
    """
    Save parsed chunks to JSON file.

    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the JSON file
    """
    if not chunks:
        logger.warning(f"No chunks to save for {output_path}")
        return

    data = {
        "source_file": chunks[0]["metadata"].get(
            "source_file", str(output_path.stem)
        ),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_from_json(json_path: Path) -> List[Dict]:
    """
    Load chunks from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        List of chunk dictionaries
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {data['total_chunks']} chunks from {json_path}")
        return data["chunks"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load chunks from {json_path}: {e}")
        return []


def generate_qa_pairs_batch(
    llm: LLM, chunks: List[Dict], sampling_params: SamplingParams
) -> List[List[Dict]]:
    """
    Generate QA pairs for multiple chunks using vLLM batch inference.

    Args:
        llm: vLLM LLM instance
        chunks: List of chunk dictionaries
        sampling_params: Sampling parameters

    Returns:
        List of QA pair lists (one per chunk)
    """
    if not chunks:
        return []

    # Prepare all prompts
    prompts = [
        QUESTION_GENERATION_PROMPT.format(text_chunk=chunk["text"])
        for chunk in chunks
    ]

    # Batch generate
    outputs = llm.generate(prompts, sampling_params)

    all_questions = []
    for chunk, output in zip(chunks, outputs):
        chunk_id = chunk["id"]
        try:
            response = output.outputs[0].text.strip()
            logger.debug(
                f"Chunk {chunk_id} raw response length: {len(response)}"
            )
            logger.debug(
                f"Chunk {chunk_id} raw response preview: {response[:200]}..."
            )

            # Parse JSON directly since using guided decoding
            qa_data = json.loads(response)
            logger.debug(f"Chunk {chunk_id}: Successfully extracted JSON")
            questions = qa_data.get("questions", [])
            logger.debug(f"Chunk {chunk_id}: Parsed {len(questions)} questions")

            # Validate structure
            for q in questions:
                if not all(
                    k in q for k in ["difficulty", "question", "answer"]
                ):
                    logger.debug(
                        f"Chunk {chunk_id}: Invalid QA structure - missing keys"
                    )
                    raise ValueError("Invalid QA structure")

            all_questions.append(questions)
            logger.debug(
                f"Chunk {chunk_id}: Successfully processed {len(questions)} questions"
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                f"Failed to parse QA response for chunk {chunk_id}: {e}"
            )
            logger.debug(
                f"Chunk {chunk_id} problematic response: {response[:500]}"
            )
            all_questions.append([])

    return all_questions


def save_qa_to_json(qa_data: Dict, output_path: Path) -> None:
    """
    Save QA pairs to JSON file.

    Args:
        qa_data: QA data dictionary
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved QA data to {output_path}")


def load_qa_from_json(json_path: Path) -> Dict:
    """
    Load QA data from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        QA data dictionary
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            f"Loaded QA data with {data['total_qa_pairs']} pairs from {json_path}"
        )
        return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load QA data from {json_path}: {e}")
        return {
            "qa_pairs": [],
            "total_qa_pairs": 0,
            "source_file": str(json_path.stem),
        }


def validate_qa_pairs_batch(
    llm: LLM,
    qa_pairs: List[Dict],
    sampling_params: SamplingParams,
) -> None:
    """
    Validate QA pairs using vLLM batch inference and mark results directly.

    Args:
        llm: vLLM LLM instance
        qa_pairs: List of QA pair dictionaries
        sampling_params: Sampling parameters

    Note:
        This function modifies qa_pairs in-place by adding validation results to each question.
    """
    if not qa_pairs:
        return

    # Prepare all prompts and track indices
    prompts = []
    indices = []  # (qa_pair_idx, question_idx)

    for qa_idx, qa_pair in enumerate(qa_pairs):
        chunk = qa_pair["original_chunk"]
        for q_idx, question in enumerate(qa_pair["questions"]):
            prompt = VALIDATION_PROMPT.format(
                text_chunk=chunk,
                question=question["question"],
                answer=question["answer"],
            )
            prompts.append(prompt)
            indices.append((qa_idx, q_idx))

    # Batch validate
    logger.info(f"Validating {len(prompts)} QA pairs in batch")
    outputs = llm.generate(prompts, sampling_params)

    # Store results directly in the question objects
    pass_count = 0
    fail_count = 0
    for (qa_idx, q_idx), output in zip(indices, outputs):
        response = output.outputs[0].text.strip()

        # DEBUG: Log raw validation response
        logger.debug(
            f"Validation response for qa[{qa_idx}][{q_idx}]: '{response}'"
        )

        # Parse JSON response from structured output
        try:
            validation_data = json.loads(response)
            result = validation_data.get("result", "FAIL")
            reason = validation_data.get("reason", "")
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse validation response for qa[{qa_idx}][{q_idx}]: {e}"
            )
            # Fallback: treat as FAIL if parsing fails
            result, reason = "FAIL", f"JSON parse error: {e}"

        if result == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        # Store validation result directly in the question object
        qa_pairs[qa_idx]["questions"][q_idx]["validation"] = {
            "result": result,
            "reason": reason,
        }

    logger.info(
        f"Validation complete: {pass_count} PASS, {fail_count} FAIL out of {len(prompts)} total"
    )


def filter_qa_pairs(qa_data: Dict) -> Dict:
    """
    Filter QA pairs that have validation result stored, keeping only PASS results.

    Args:
        qa_data: Original QA data with validation results stored in questions

    Returns:
        Filtered QA data dictionary
    """
    filtered_qa_pairs = []
    filtered_count = 0

    for qa_pair in qa_data["qa_pairs"]:
        for question in qa_pair["questions"]:
            validation = question.get("validation", {})
            result = validation.get("result", "FAIL")

            if result == "PASS":
                filtered_qa_pair = {
                    "id": len(filtered_qa_pairs),
                    "chunk_id": qa_pair["chunk_id"],
                    "original_chunk": qa_pair["original_chunk"],
                    "difficulty": question["difficulty"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "validation": validation,
                }
                filtered_qa_pairs.append(filtered_qa_pair)
            else:
                filtered_count += 1

    filtered_data = {
        "source_file": qa_data["source_file"],
        "total_qa_pairs": len(filtered_qa_pairs),
        "filtered_count": filtered_count,
        "qa_pairs": filtered_qa_pairs,
    }

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Digital Design Knowledge Dataset Preprocessing"
    )
    parser.add_argument(
        "--input-pdf", type=str, help="Specific PDF file to process"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "parse", "generate", "filter"],
        default="all",
        help="Step to execute",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip existing files"
    )

    # Step 1: Parse
    parser.add_argument(
        "--output-chunks-dir",
        type=str,
        default="data/processed/digital_design_knowledge",
        help="Output directory for chunks",
    )
    parser.add_argument(
        "--bounding-box",
        type=str,
        default=None,
        help="Bounding box margins in clockwise order from top (top,right,bottom,left) "
        "as comma-separated fractions between 0 and 1. "
        "Example: '0.1,0,0.2,0' excludes top 10%% and bottom 20%%. "
        "Default: None (parse entire page)",
    )

    # Step 2: Generate QA
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="vLLM model path"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--input-chunks-dir",
        type=str,
        default="data/processed/digital_design_knowledge",
        help="Input directory for chunks",
    )
    parser.add_argument(
        "--output-qa-dir",
        type=str,
        default="data/processed/digital_design_qa",
        help="Output directory for QA data",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens to generate",
    )

    # Step 3: Filter
    parser.add_argument(
        "--input-qa-dir",
        type=str,
        default="data/processed/digital_design_qa",
        help="Input directory for QA data",
    )
    parser.add_argument(
        "--output-filtered-dir",
        type=str,
        default="data/processed/digital_design_qa_filtered",
        help="Output directory for filtered QA data",
    )

    args = parser.parse_args()

    # Determine input PDFs
    input_dir = Path("data/raw/digital_design_knowledge")
    if args.input_pdf:
        pdf_files = [Path(args.input_pdf)]
    else:
        pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found")
        return

    # Initialize vLLM once if needed for generation or filtering
    llm = None
    gen_sampling_params = None
    val_sampling_params = None

    if args.step in ["all", "generate", "filter"]:
        logger.info(f"Initializing vLLM with model: {args.model}")
        llm = LLM(
            model=args.model, tensor_parallel_size=args.tensor_parallel_size
        )

        if args.step in ["all", "generate"]:
            structured_outputs_params_gen = StructuredOutputsParams(
                json=QUESTION_GENERATION_SCHEMA
            )
            gen_sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                structured_outputs=structured_outputs_params_gen,
            )
            logger.debug(f"Gen sampling params: no stop sequences")

        if args.step in ["all", "filter"]:
            structured_outputs_params_val = StructuredOutputsParams(
                json=VALIDATION_SCHEMA
            )
            val_sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for validation
                max_tokens=512,  # Increased for reason text
                structured_outputs=structured_outputs_params_val,
            )

    # Execute steps
    if args.step in ["all", "parse"]:
        logger.info("Starting Step 1: PDF Parsing")
        output_chunks_dir = Path(args.output_chunks_dir)
        output_chunks_dir.mkdir(parents=True, exist_ok=True)

        for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
            output_file = output_chunks_dir / f"{pdf_file.stem}_chunks.json"
            if args.skip_existing and output_file.exists():
                logger.info(f"Skipping existing file: {output_file}")
                continue

            chunks = parse_pdf_with_llamaparse(
                pdf_file, bounding_box=args.bounding_box
            )
            save_chunks_to_json(chunks, output_file)

    if args.step in ["all", "generate"]:
        logger.info("Starting Step 2: QA Generation")
        input_chunks_dir = Path(args.input_chunks_dir)
        output_qa_dir = Path(args.output_qa_dir)
        output_qa_dir.mkdir(parents=True, exist_ok=True)

        if args.input_pdf:
            pdf_stem = Path(args.input_pdf).stem
            chunk_files = [input_chunks_dir / f"{pdf_stem}_chunks.json"]
        else:
            chunk_files = list(input_chunks_dir.glob("*_chunks.json"))
        for chunk_file in tqdm(chunk_files, desc="Generating QA"):
            output_file = (
                output_qa_dir
                / f"{chunk_file.stem.replace('_chunks', '')}_qa.json"
            )
            if args.skip_existing and output_file.exists():
                logger.info(f"Skipping existing file: {output_file}")
                continue

            chunks = load_chunks_from_json(chunk_file)

            # Batch generate QA pairs for all chunks
            logger.info(
                f"Generating QA for {len(chunks)} chunks in {chunk_file.stem}"
            )
            assert (
                llm is not None and gen_sampling_params is not None
            ), "LLM and sampling params must be initialized"
            all_questions = generate_qa_pairs_batch(
                llm, chunks, gen_sampling_params
            )

            qa_pairs = []
            for chunk, questions in zip(chunks, all_questions):
                if questions:  # Only add if questions were generated
                    qa_pair = {
                        "id": chunk["id"],
                        "chunk_id": chunk["id"],
                        "original_chunk": chunk["text"],
                        "questions": questions,
                    }
                    qa_pairs.append(qa_pair)

            # Calculate total question count
            total_questions = sum(len(q["questions"]) for q in qa_pairs)

            qa_data = {
                "source_file": chunk_file.stem.replace("_chunks", ".pdf"),
                "total_qa_pairs": total_questions,
                "qa_pairs": qa_pairs,
            }
            save_qa_to_json(qa_data, output_file)

    if args.step in ["all", "filter"]:
        logger.info("Starting Step 3: QA Filtering")
        input_qa_dir = Path(args.input_qa_dir)
        output_filtered_dir = Path(args.output_filtered_dir)
        output_filtered_dir.mkdir(parents=True, exist_ok=True)

        if args.input_pdf:
            pdf_stem = Path(args.input_pdf).stem
            qa_files = [input_qa_dir / f"{pdf_stem}_qa.json"]
        else:
            qa_files = list(input_qa_dir.glob("*_qa.json"))
        for qa_file in tqdm(qa_files, desc="Filtering QA files"):
            output_file = (
                output_filtered_dir
                / f"{qa_file.stem.replace('_qa', '')}_qa_filtered.json"
            )
            if args.skip_existing and output_file.exists():
                logger.info(f"Skipping existing file: {output_file}")
                continue

            qa_data = load_qa_from_json(qa_file)

            # Batch validate all QA pairs and store results directly
            assert (
                llm is not None and val_sampling_params is not None
            ), "LLM and sampling params must be initialized"
            validate_qa_pairs_batch(
                llm, qa_data["qa_pairs"], val_sampling_params
            )

            filtered_data = filter_qa_pairs(qa_data)
            save_qa_to_json(filtered_data, output_file)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
