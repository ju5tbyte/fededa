#!/usr/bin/env python3
"""
Textbook Preprocessing Script

This script processes a PDF textbook file, extracts text using LlamaParse
or DeepSeek-OCR (via vllm), converts to markdown, merges all pages into
a single text, and performs markdown header-based chunking.

Usage:
    # Full pipeline with DeepSeek-OCR (vllm)
    python scripts/preprocess_textbook.py \
        --file data/raw/digital_design_knowledge/vlsi_design.pdf \
        --output-dir data/processed/digital_design_knowledge_source_deepseek_ocr \
        --step all \
        --parser vllm

    # Full pipeline with LlamaParse
    python scripts/preprocess_textbook.py \
        --file data/raw/digital_design_knowledge/vlsi_design.pdf \
        --output-dir data/processed/digital_design_knowledge_source \
        --bbox "0.125,0,0,0" \
        --step all \
        --parser llamaparse

    # Only parse PDF to markdown (DeepSeek-OCR)
    python scripts/preprocess_textbook.py \
        --file data/raw/digital_design_knowledge/vlsi_design.pdf \
        --output-md data/processed/digital_design_knowledge_source/vlsi_design.md \
        --step parse \
        --parser vllm

    # Only parse PDF to markdown (LlamaParse with bounding box)
    python scripts/preprocess_textbook.py \
        --file data/raw/digital_design_knowledge/vlsi_design.pdf \
        --output-md data/processed/digital_design_knowledge_source/vlsi_design.md \
        --step parse \
        --bbox "0.1,0,0.1,0"

    # Only chunk existing markdown file
    python scripts/preprocess_textbook.py \
        --file data/processed/digital_design_knowledge_source/vlsi_design.md \
        --output-json data/processed/digital_design_knowledge_source/vlsi_design.json \
        --step chunking
"""

import argparse
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter

# For vllm OCR
from PIL import Image
from pdf2image import convert_from_path

# vllm imports
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Markdown headers to split on
MARKDOWN_HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

# OCR prompt for vllm
OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF pages to PIL Images.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (default: 300)

    Returns:
        List of PIL Image objects
    """
    logger.info(f"Converting PDF to images: {pdf_path} (dpi={dpi})")
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
        logger.info(f"Converted {len(images)} pages to images")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise


# DeepSeek-OCR model name (hardcoded)
DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR"


def parse_pdf_to_markdown_vllm(
    pdf_path: Path,
    tensor_parallel_size: int = 1,
) -> str:
    """Parse PDF using DeepSeek-OCR (via vllm) and convert to markdown.

    Args:
        pdf_path: Path to the PDF file
        tensor_parallel_size: Tensor parallel size for vllm (default: 1)

    Returns:
        Merged markdown text from all pages
    """
    if not VLLM_AVAILABLE:
        raise ImportError(
            "vllm is not installed. Install with: pip install vllm"
        )

    logger.info(
        f"Parsing PDF with vllm: {pdf_path} (model={DEEPSEEK_OCR_MODEL})"
    )

    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path)

    # Initialize vllm LLM
    logger.info(f"Initializing vllm LLM: {DEEPSEEK_OCR_MODEL}")

    # Use DeepSeek-OCR specific implementation
    try:
        from vllm.model_executor.models.deepseek_ocr import (
            NGramPerReqLogitsProcessor,
        )

        llm = LLM(
            model=DEEPSEEK_OCR_MODEL,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
    except ImportError as e:
        logger.warning(
            f"Failed to import NGramPerReqLogitsProcessor: {e}. "
            "Falling back to default LLM initialization."
        )
        llm = LLM(
            model=DEEPSEEK_OCR_MODEL,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
        )

    # DeepSeek-OCR format: batch processing
    # Use user's prompt for OCR
    prompt = OCR_PROMPT

    model_inputs = []
    for image in images:
        image_rgb = image.convert("RGB")
        model_inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image_rgb},
            }
        )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )

    # Generate all pages in batch
    logger.info(f"Processing {len(images)} pages in batch")
    outputs = llm.generate(model_inputs, sampling_params)

    # Extract text from outputs
    all_markdown = []
    for i, output in enumerate(outputs):
        markdown_text = output.outputs[0].text.strip()
        all_markdown.append(markdown_text)
        logger.debug(
            f"Page {i + 1} markdown length: {len(markdown_text)} chars"
        )

    # Merge all pages into a single markdown document
    full_markdown = "\n\n".join(all_markdown)
    logger.info(f"Merged markdown length: {len(full_markdown)} characters")

    # Post-process for DeepSeek-OCR specific artifacts
    full_markdown = postprocess_deepseek_markdown(full_markdown)
    logger.info(
        f"Post-processed markdown length: {len(full_markdown)} characters"
    )

    return full_markdown


def postprocess_deepseek_markdown(markdown_text: str) -> str:
    """Post-process markdown generated by DeepSeek-OCR.

    Performs the following transformations:
    1. Remove reference/detection lines starting with <|ref|>...<\det|>
    2. Convert HTML tables to markdown tables
    3. Convert inline math \( ... \) to $ ... $
    4. Convert display math \[ ... \] to $$ ... $$

    Args:
        markdown_text: Raw markdown text from DeepSeek-OCR

    Returns:
        Processed markdown text
    """
    logger.info("Post-processing DeepSeek-OCR markdown")

    # 1. Remove reference/detection lines
    # Pattern: <|ref|>text<|/ref|><|det|>[[numbers]]<|/det|>\n
    ref_det_pattern = r"<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>\n"
    markdown_text = re.sub(
        ref_det_pattern, "", markdown_text, flags=re.MULTILINE
    )

    # 2. Convert HTML tables to markdown tables
    def html_table_to_markdown(match):
        table_html = match.group(0)
        # Parse HTML table
        rows = []
        # Extract table rows
        tr_pattern = r"<tr>(.*?)</tr>"
        td_pattern = r"<td[^>]*>(.*?)</td>"

        for tr_match in re.finditer(
            tr_pattern, table_html, re.IGNORECASE | re.DOTALL
        ):
            row_content = tr_match.group(1)
            cells = []
            for td_match in re.finditer(
                td_pattern, row_content, re.IGNORECASE | re.DOTALL
            ):
                cell_content = td_match.group(1).strip()
                # Unescape HTML entities
                cell_content = html.unescape(cell_content)
                cells.append(cell_content)
            if cells:
                rows.append(cells)

        if not rows:
            return table_html  # Return original if parsing failed

        # Convert to markdown table
        markdown_rows = []
        for i, row in enumerate(rows):
            markdown_row = "| " + " | ".join(row) + " |"
            markdown_rows.append(markdown_row)
            if i == 0:  # Add separator after header
                separator = "| " + " | ".join(["---"] * len(row)) + " |"
                markdown_rows.append(separator)

        return "\n".join(markdown_rows) + "\n"

    # Find and replace HTML tables
    table_pattern = r"<table[^>]*>.*?</table>"
    markdown_text = re.sub(
        table_pattern,
        html_table_to_markdown,
        markdown_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 3. Convert inline math \( ... \) to $ ... $
    inline_math_pattern = r"\\\((.*?)\\\)"
    markdown_text = re.sub(
        inline_math_pattern, r"$\1$", markdown_text, flags=re.DOTALL
    )

    # 4. Convert display math \[ ... \] to $$ ... $$
    display_math_pattern = r"\\\[(.*?)\\\]"
    markdown_text = re.sub(
        display_math_pattern, r"$$\1$$", markdown_text, flags=re.DOTALL
    )

    logger.info("Post-processing completed")
    return markdown_text


def parse_pdf_to_markdown(
    pdf_path: Path,
    bbox: Optional[str] = None,
    parser_type: str = "llamaparse",
    vllm_tensor_parallel_size: int = 1,
) -> str:
    """Parse PDF and merge all pages into a single markdown text.

    Args:
        pdf_path: Path to the PDF file
        bbox: Bounding box string in format "top,left,bottom,right" (default: None)
        parser_type: Parser type - "llamaparse" or "vllm" (default: "llamaparse")
        vllm_tensor_parallel_size: Tensor parallel size for vllm (default: 1)

    Returns:
        Merged markdown text from all pages
    """
    if parser_type == "vllm":
        return parse_pdf_to_markdown_vllm(
            pdf_path=pdf_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
        )

    # Default to LlamaParse
    logger.info(f"Parsing PDF: {pdf_path} (bbox={bbox})")

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "LLAMA_CLOUD_API_KEY environment variable is not set."
        )

    parser_kwargs = {
        "api_key": api_key,
        "result_type": "markdown",
        "verbose": True,
        "language": "en",
        "high_res_ocr": True,
        "use_vendor_multimodal_model": True,
        "vendor_multimodal_model_name": "gemini-2.0-flash",
    }

    if bbox:
        parser_kwargs["bounding_box"] = bbox

    parser = LlamaParse(**parser_kwargs)

    # Parse the PDF
    try:
        documents = parser.load_data(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise
    logger.debug(f"Raw documents from LlamaParse: {len(documents)} documents")

    # Merge all pages into a single markdown document
    full_markdown = "\n\n".join([doc.text for doc in documents])
    logger.info(f"Merged markdown length: {len(full_markdown)} characters")

    return full_markdown


def markdown_header_chunking(text: str) -> List[str]:
    """Perform chunking based on markdown headers.

    This function uses LangChain's MarkdownHeaderTextSplitter to split
    text at markdown header boundaries (#, ##, ###, etc.), then filters out
    chunks with fewer than 300 characters or header-only content.

    Args:
        text: Input text to chunk

    Returns:
        List of text chunks
    """
    logger.info("Performing markdown header-based chunking")

    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )

    chunks = text_splitter.split_text(text)

    # Extract text from documents
    chunk_texts = [doc.page_content for doc in chunks]

    # Filter out chunks with fewer than 300 characters or header-only content
    MIN_CHUNK_LENGTH = 300
    filtered_chunks = []
    removed_count = 0

    for chunk in chunk_texts:
        # Check if chunk is header-only (only contains header lines)
        lines = chunk.strip().split("\n")
        non_header_lines = [
            line for line in lines if not line.strip().startswith("#")
        ]

        chunk_length = len(chunk)

        # Remove if:
        # 1. Character count is less than 300
        # 2. Chunk only contains header lines (no actual content)
        if chunk_length < MIN_CHUNK_LENGTH:
            logger.debug(
                f"Removed chunk (char_count={chunk_length} < {MIN_CHUNK_LENGTH}): "
                f"{chunk[:50]}..."
            )
            removed_count += 1
        elif not non_header_lines:
            logger.debug(f"Removed header-only chunk: {chunk[:50]}...")
            removed_count += 1
        else:
            filtered_chunks.append(chunk)

    logger.info(
        f"Filtered {removed_count} chunks (char_count < {MIN_CHUNK_LENGTH} or header-only), "
        f"remaining: {len(filtered_chunks)} chunks"
    )

    return filtered_chunks


def save_chunks_to_json(
    chunks: List[str], source_file: str, output_path: Path
) -> None:
    """
    Save chunks to JSON file with specified format.

    Args:
        chunks: List of text chunks
        source_file: Source PDF filename
        output_path: Path to save the JSON file
    """
    chunk_data = []
    for i, chunk_text in enumerate(chunks):
        chunk_data.append(
            {
                "id": i,
                "text": chunk_text,
                "metadata": {
                    "source": source_file,
                    "char_count": len(chunk_text),
                },
            }
        )

    data = {
        "chunks": chunk_data,
        "total_chunks": len(chunk_data),
        "source_file": source_file,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunk_data)} chunks to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Textbook Preprocessing Script"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to input file (PDF for parse/all steps, markdown for chunking step)",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "parse", "chunking"],
        default="all",
        help="Processing step to execute: 'all' (parse PDF + chunking), 'parse' (only parse PDF to markdown), 'chunking' (only chunk existing markdown file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for 'all' step (both markdown and JSON will be placed here with default names)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Output markdown file path for 'parse' step (required for parse, optional for all)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file path for 'chunking' step (required for chunking, optional for all)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box for PDF parsing in format 'top,left,bottom,right' (e.g., '0.1,0,0.2,0')",
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["llamaparse", "vllm"],
        default="llamaparse",
        help="Parser type: 'llamaparse' (default) or 'vllm' for local OCR with VLM",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vllm (default: 1)",
    )

    args = parser.parse_args()

    # Validate parser-specific requirements
    if args.parser == "vllm":
        if not VLLM_AVAILABLE:
            parser.error(
                "vllm is not installed. Install with: pip install vllm"
            )

    input_file = Path(args.file)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return

    # Validate step-specific requirements
    if args.step == "all":
        if not args.output_dir:
            parser.error("--output-dir is required for step 'all'")
        # If output-md or output-json provided, they override default names
    elif args.step == "parse":
        if not args.output_md:
            parser.error("--output-md is required for step 'parse'")
    elif args.step == "chunking":
        if not args.output_json:
            parser.error("--output-json is required for step 'chunking'")

    # File type warnings
    if args.step in ["all", "parse"] and input_file.suffix.lower() != ".pdf":
        logger.warning(f"Input file is not a PDF (.pdf): {input_file}")
    if args.step == "chunking" and input_file.suffix.lower() != ".md":
        logger.warning(f"Input file is not a markdown (.md): {input_file}")

    full_text = ""
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None

    # Determine output paths based on step
    if args.step == "all":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Markdown path
        if args.output_md:
            markdown_path = Path(args.output_md)
        else:
            markdown_path = output_dir / f"{input_file.stem}.md"
        # JSON path
        if args.output_json:
            json_path = Path(args.output_json)
        else:
            json_path = output_dir / f"{input_file.stem}.json"
    elif args.step == "parse":
        markdown_path = Path(args.output_md)
        json_path = None
    elif args.step == "chunking":
        markdown_path = None
        json_path = Path(args.output_json)

    # Step 1: Parse PDF (if needed)
    if args.step in ["all", "parse"]:
        full_text = parse_pdf_to_markdown(
            pdf_path=input_file,
            bbox=args.bbox,
            parser_type=args.parser,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        )
        # Save full markdown file
        if markdown_path is not None:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            logger.info(f"Saved full markdown to {markdown_path}")
        else:
            logger.error("Markdown path not defined for parse step")
            return
    elif args.step == "chunking":
        # Load existing markdown file
        with open(input_file, "r", encoding="utf-8") as f:
            full_text = f.read()
        logger.info(
            f"Loaded markdown from {input_file} (length: {len(full_text)} characters)"
        )

    # Step 2: Chunking (if needed)
    chunks = []
    if args.step in ["all", "chunking"]:
        if not full_text:
            logger.error("No text available for chunking")
            return

        logger.info("Using markdown header-based chunking method")
        chunks = markdown_header_chunking(full_text)

        # Save to JSON
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            save_chunks_to_json(chunks, input_file.name, json_path)
            logger.info(f"Saved {len(chunks)} chunks to {json_path}")
        else:
            logger.error("JSON path not defined for chunking step")
            return

    logger.info(f"Step '{args.step}' completed successfully")


if __name__ == "__main__":
    main()
