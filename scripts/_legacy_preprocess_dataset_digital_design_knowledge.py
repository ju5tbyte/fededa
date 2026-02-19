#!/usr/bin/env python3
"""
Digital Design Knowledge Dataset Preprocessing Script

This script processes PDF documents containing digital design knowledge
and generates QA datasets through a 3-step pipeline:

1. Parse PDFs using LlamaParse with semantic chunking
2. Generate QA pairs (0 to 10 per chunk) using vLLM with Llama 3.3 70B AWQ
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
DEFAULT_MAX_TOKENS = 8192

QUESTION_GENERATION_PROMPT = """You are an expert professor in digital logic design, computer architecture, VLSI design, and general electrical and computer engineering.

Your task: Read the text chunk below and generate high-quality questions that test **genuine understanding of concepts** in digital logic design, computer architecture, VLSI design, or related electrical/computer engineering topics. 

**Text Chunk:**
{text_chunk}

**CRITICAL RULES:**

1. **Content-Grounded Principle:** Every question MUST be derived from **substantive technical content actually present in the text chunk**. 
   - The text chunk must contain enough technical explanation, definitions, examples, derivations, or analysis to form the basis of a question.
   - Do NOT generate questions from: tables of contents, chapter titles, section headings, bullet-point topic lists, index entries, references, copyright notices, or any text that merely *mentions* a topic without *explaining* it.
   - **Key test:** If the chunk only *names* a topic (e.g., "Chapter 5: Pipelining") but does not *explain* it, you MUST return an empty array. A topic name alone is NOT sufficient basis for a question.
   - If the chunk contains a mix of substantive and non-substantive content, generate questions ONLY from the substantive portions.

   **Exercise / Problem-Set Chunks — SPECIAL HANDLING:**
   - A chunk that is primarily a list of exercises or homework problems (e.g., "1. Implement the circuit...", "2. Show that...") is a set of **task directives**, NOT technical explanations.
   - Task directives like "Implement X", "Show that Y", "Design Z" merely *name* what the student should do — they do NOT *explain* the underlying concepts.
   - Therefore: if the chunk is primarily an exercise list, return an **empty array**, even if the exercises reference well-known topics like decoders, multiplexers, or Boolean algebra.
   - **Exception:** If an exercise chunk also contains substantial explanatory prose (definitions, derivations, worked examples — not just task instructions), you may generate questions from that explanatory content only.

2. **Standalone Principle:** While questions must be grounded in the text chunk's content, the final question and answer must make complete sense with domain knowledge alone, WITHOUT referencing the specific text.

   **FORBIDDEN references in BOTH question and answer — any of these → automatically invalid:**
   - "the text", "the author", "the passage", "the chapter", "the book"
   - "the example above", "as described in", "as shown in", "as illustrated in"
   - "Figure [any number]", "Table [any number]", "Diagram [any number]" (e.g., "Figure 9-17", "Table 3-2")
   - "Section [any number]", "Chapter [any number]" (e.g., "section 10.3", "chapter 5")
   - "the circuit in Figure", "the implementation in Figure", "the truth table in Figure"
   - Any other phrasing that requires access to a specific figure, table, diagram, or textbook section to understand the question or verify the answer

   - BAD: "What does the author describe as the advantage of CMOS?"  
   - BAD: "According to the text, how many inputs does the circuit have?"
   - BAD: "Compare the circuit in Figure 9-17 with Figure 9-18."
   - BAD: "List the three types of hazards mentioned in the passage."
   - GOOD: "Why do CMOS circuits consume less static power than NMOS-only circuits?"
   - GOOD: "Given F = A'B + AB', derive an equivalent expression using NAND gates only."
   - GOOD: "Explain how a 5-stage RISC pipeline handles a data hazard caused by a RAW dependency between two consecutive instructions."

   In other words: the **topic and depth** come from the text chunk, but the **phrasing** must be self-contained.

3. **Question Difficulty Tiers**
   - **easy**: Define a concept or recall a key property.  
     Example: "What is the difference between a latch and a flip-flop?"
     Example: "What is the purpose of the program counter in a CPU?"
     Example: "What is the difference between an n-well and a p-well CMOS process?"
   - **medium**: Apply a concept, compare alternatives, or explain *why* something works.  
     Example: "Why is a synchronous counter preferred over an asynchronous (ripple) counter in high-speed designs?"
     Example: "Compare write-through and write-back cache policies in terms of memory traffic and consistency."
     Example: "Explain why dynamic power dissipation in CMOS is proportional to the switching frequency and the square of the supply voltage."
   - **hard**: Multi-step reasoning, design/optimization, or synthesis across concepts.  
     Example: "Design a minimal two-level SOP circuit for F(A,B,C) = Σm(1,2,5,6) using a Karnaugh map, and state how many literals the minimized expression contains."
     Example: "A processor has a 5-stage pipeline. Given the instruction sequence ADD R1,R2,R3 followed by SUB R4,R1,R5, describe the data hazard that arises, and compare the cycle cost of resolving it via stalling versus forwarding."
     Example: "Given a CMOS inverter with (W/L)_p = 2(W/L)_n, explain how to size the transistors so that the switching threshold is at VDD/2, and describe how this sizing affects rise and fall times."

4. **Full QA Pair Example (for reference on expected quality):**
   - **difficulty:** "medium"
   - **topic:** "Cache Memory"
   - **question:** "A direct-mapped cache has 16 cache lines and uses a block size of 4 words. If the CPU issues a byte address of 0x0000_02B4, explain how the address is partitioned into tag, index, and offset fields, and identify which cache line this address maps to."
   - **answer:** "With 4 words per block (16 bytes), the byte offset field is log2(16) = 4 bits. With 16 cache lines, the index field is log2(16) = 4 bits. The remaining upper bits form the tag. For address 0x0000_02B4 = 0000...0010_1011_0100 in binary, the lowest 4 bits (0100) are the byte offset, the next 4 bits (1011 = 11 decimal) are the index, so this address maps to cache line 11. The tag is the remaining upper bits (0x0000_02)."

5. **No Duplicate Concepts:** Each question must test a **distinct concept or reasoning skill**. Do not generate multiple questions that cover the same idea from slightly different angles. If the chunk only contains one substantive concept, generate fewer questions rather than producing semantic duplicates.

6. **Forbidden Question Types:**
   - Questions about the textbook itself (chapter titles, section numbers, what the author says)
   - Questions that reference or depend on figures, tables, or diagrams (e.g., "Compare the circuit in Figure X with Figure Y")
   - Pure definition lookups that any glossary could answer (unless the concept is nuanced)
   - Yes/No questions without requiring justification
   - Questions on topics that the text chunk merely *names* but does not *explain*

7. **Encouraged Question Types:**
   - "Why" and "How" questions that require reasoning
   - "Compare and contrast" between related concepts
   - "Given [scenario/expression/circuit description/instruction sequence], determine/simplify/analyze..."
   - Troubleshooting: "If [something goes wrong], what is the likely cause?"
   - Design questions: "How would you implement X using Y?"

8. **Answer Quality:**
   - Answers should be **technically correct and self-contained** — a knowledgeable person should verify them without needing the textbook.
   - Answers must be grounded in the technical content of the text chunk. Do not fabricate details not supported by the chunk or general domain knowledge.
   - **No references to figures, tables, sections, or chapters** in answers (e.g., do NOT write "as shown in Figure 9-18").
   - **Information Gain Requirement:** The answer must provide concrete, specific information beyond what is already stated in the question itself. An answer that merely restates or rephrases the question is invalid.
     - BAD: Q: "What does the 74 prefix indicate?" → A: "It indicates a 7400 series chip." (tautological — zero information gain)
   - **Technical Accuracy for Derivations:** For answers involving derivation such as Boolean algebra, circuit equations, K-map minimizations, etc.: Show key intermediate steps, not just the final result.
   - Keep answers concise but complete (2-6 sentences typically).

9. **Skip Noise — STRICTLY enforce this rule:** Return an empty array if ANY of the following apply:
   - The chunk is a table of contents, index, list of references, copyright page, or preface without technical content.
   - The chunk consists primarily of topic names, section headings, or enumerated lists of subjects without substantive explanation.
   - **The chunk is primarily a list of exercises or homework problems** (numbered task directives like "Implement...", "Show that...", "Design...") without accompanying explanatory prose.
   - The chunk lacks sufficient technical depth to generate a question.
   - **When in doubt, return an empty array.** Generating zero questions from a non-substantive chunk is always preferable to generating ungrounded questions.

**Output Format (JSON):**
{{
  "questions": []  // Return empty if content is not suitable based on constraints
}}
OR (if valid substantive content exists, depending on the chunk, you might generate 1~10 questions):
{{
  "questions": [
    {{
      "difficulty": "easy|medium|hard",
      "topic": "brief topic tag, e.g. 'Karnaugh Maps', 'Cache Coherence', 'CMOS Inverter Sizing', 'Pipelining'",
      "question": "...",
      "answer": "..."
    }}
  ]
}}

**Before generating, perform this self-check:**
1. Does this text chunk actually *explain* something technical, or does it merely *list/name* topics? If the latter → empty array.
2. Is this chunk primarily an exercise/problem list with numbered tasks but no explanatory prose? If yes → empty array.
3. For each question I am about to generate: does the chunk contain the actual explanation needed to form this question, or am I drawing on outside knowledge? If outside knowledge → do not generate that question.
4. Does any question or answer reference a Figure, Table, Section, or Chapter? If yes → remove that reference or discard the question.
5. Does any answer merely restate the question without adding specific technical information? If yes → discard.

Generate 0-10 questions now:"""


VALIDATION_PROMPT = """You are a strict auditor for a digital logic design, computer architecture, VLSI design, and electrical/computer engineering problem set.

Evaluate whether the following QA pair is suitable for teaching students to learn the domain knowledge.

**Original Text:**
{text_chunk}

**Question:** {question}
**Proposed Answer:** {answer}

**Evaluation Criteria:**

1. **Content-Grounding Test:** Does the original text contain **substantive technical explanation** of the questioned concept — such as definitions, derivations, comparisons, step-by-step procedures, worked examples, or analysis?

   Merely mentioning a topic name in a heading, exercise instruction, or task directive does **NOT** constitute explanation. Apply these rules strictly:
   - Exercise prompts like "Implement X", "Show that Y", "Design Z using..." are **task directives**, not explanations. A chunk that is primarily a numbered list of such directives does NOT explain the underlying concepts.
   - If the chunk is primarily an exercise/problem list and the question asks about a concept that is only *referenced* (not *explained*) in those exercises → **FAIL**.
   - If the chunk only contains a section heading + brief transitional sentence (e.g., "This chapter covers X. The following exercises...") without substantive explanation → **FAIL**.
   - A topic being *well-known in the domain* does not exempt it from this test. The question must be grounded in **this specific chunk's explanatory content**.

   **To determine PASS:** Identify the specific sentences or paragraphs in the original text that explain the concept tested by the question. If you cannot point to at least 2-3 sentences of genuine technical explanation (not task directives), → **FAIL**.

2. **Standalone Test:** Can this question be understood and answered WITH the domain knowledge of digital logic design, computer architecture, VLSI design, or general electrical/computer engineering, WITHOUT reading the original text?

   Check **both the question AND the answer** for forbidden references. If ANY of the following patterns appear in EITHER the question or the answer → **FAIL**:
   - "the text", "the author", "the passage", "the chapter", "the book"
   - "Figure" followed by any number/identifier (e.g., "Figure 9-17", "Figure 3-2")
   - "Table" followed by any number/identifier (e.g., "Table 2-1")
   - "Diagram" followed by any number/identifier
   - "Section" or "section" followed by any number (e.g., "section 10.3")
   - "Chapter" or "chapter" followed by any number
   - "as shown in", "as described in", "as illustrated in", "as presented in"
   - "the circuit in Figure", "the implementation in Figure"
   - "the example above", "the previous example", "the following figure"
   
   If the question or answer cannot be fully understood without access to a specific figure, table, or textbook section → **FAIL**.

3. **Technical Correctness:** Is the answer factually and technically accurate for its respective domain (digital logic, computer architecture, VLSI, or EE/CE)?

   **Apply heightened scrutiny to derivations and equations:** For example, verify the correct gate-level construction for NAND/NOR gate implementations. If there is a phrase "A AND B = (A' NAND B')'" it is incorrect since the correct NAND implementation of AND is ((A NAND B) NAND (A NAND B)). 
   
   A incorrect equation, wrong minimization result, or erroneous derivation step → **FAIL**.
   If the answer contains errors, misleading simplifications, or hallucinated facts → **FAIL**.

4. **Non-Triviality:** Does the question require meaningful knowledge or reasoning? Pure metadata questions ("What chapter covers X?") or questions answerable by common sense alone → **FAIL**.

5. **Relevance:** Is the question about digital logic design, computer architecture, VLSI design, or closely related EE/CS topics? Off-topic → **FAIL**.

6. **Answer Completeness & Information Gain:** Does the answer adequately address the question AND provide genuine information beyond what the question itself already states?
   - A vague or incomplete answer → **FAIL**.
   - A **tautological answer** that merely restates the question in different words without adding specific, concrete information → **FAIL**.
     - Example of tautological FAIL: Q: "What does the 74 prefix indicate?" A: "The 74 prefix indicates that the chip is a 7400 series chip." (This restates the question — it does not explain what the 7400 series is, what technology it uses, or why the prefix matters.)
   - The answer must contain at least one piece of specific technical information not already present in the question.

**Output Format (JSON):**
{{
  "reason": "Explain which criteria passed/failed and why. For Content-Grounding, cite the specific explanatory content found (or not found) in the original text. For Technical Correctness of derivations, show your verification. For Standalone, list any forbidden references found.",
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
# Updated: questions array can be empty (0 questions) or have up to 10 questions
QUESTION_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 10,
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
    Collect all QA pairs with validation results for debugging.
    Includes both PASS and FAIL results.

    Args:
        qa_data: Original QA data with validation results stored in questions

    Returns:
        QA data dictionary with all validation results
    """
    all_qa_pairs = []
    pass_count = 0
    fail_count = 0

    for qa_pair in qa_data["qa_pairs"]:
        for question in qa_pair["questions"]:
            validation = question.get("validation", {})
            result = validation.get("result", "FAIL")

            if result == "PASS":
                pass_count += 1
            else:
                fail_count += 1

            qa_item = {
                "id": len(all_qa_pairs),
                "chunk_id": qa_pair["chunk_id"],
                "original_chunk": qa_pair["original_chunk"],
                "difficulty": question["difficulty"],
                "question": question["question"],
                "answer": question["answer"],
                "validation": validation,
            }
            all_qa_pairs.append(qa_item)

    result_data = {
        "source_file": qa_data["source_file"],
        "total_qa_pairs": len(all_qa_pairs),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "qa_pairs": all_qa_pairs,
    }

    return result_data


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
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_tokens
            + 2048,  # Add buffer for prompt length
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
