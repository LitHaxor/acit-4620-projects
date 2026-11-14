#!/usr/bin/env python3
"""
convert_papers_to_markdown.py
-----------------------------
Converts research paper PDFs into clean, agent-ready Markdown files.

Features:
- Preserves document layout, multi-column reading order, tables, and math equations.
- Adds structured YAML frontmatter (title, authors, DOI/subject, pages, word counts, section outline).
- Injects explicit page demarcation tags (e.g., `<!-- PAGE X OF Y -->`) so agents can cite precise page numbers.
- Generates an `INDEX.md` catalog linking all papers for quick agent discovery and navigation.
- Generates a machine-readable `papers_catalog.json` for agent toolchains and RAG indexing.
- Supports batch conversion or single-file conversion via CLI flags.

Usage:
    # Convert all papers in default directory (papers/ -> papers/markdown/)
    python convert_papers_to_markdown.py

    # Custom directories
    python convert_papers_to_markdown.py --papers-dir papers --output-dir papers/markdown

    # Convert a single paper
    python convert_papers_to_markdown.py --file papers/Beyond_here_and_now.pdf
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pymupdf  # PyMuPDF
    import pymupdf4llm
except ImportError:
    print(
        "Error: 'pymupdf' and 'pymupdf4llm' are required.\n"
        "Install them with: pip install pymupdf pymupdf4llm rapidocr-onnxruntime",
        file=sys.stderr,
    )
    sys.exit(1)


def clean_title(raw_title: str) -> str:
    """Clean title string by removing extra whitespace and HTML/XML tags."""
    if not raw_title:
        return ""
    # Remove HTML-like tags (e.g. <sub>, <sup>)
    clean = re.sub(r"<[^>]+>", "", raw_title)
    # Remove line breaks and normalize spaces
    clean = " ".join(clean.split()).strip()
    return clean


def extract_metadata_and_outline(
    pdf_path: Path, doc: pymupdf.Document, chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract structured metadata, word count, character count, and section outline.
    """
    pdf_meta = doc.metadata or {}
    meta_title = clean_title(pdf_meta.get("title", ""))
    meta_author = clean_title(pdf_meta.get("author", ""))
    meta_subject = clean_title(pdf_meta.get("subject", ""))
    page_count = len(doc)

    # Calculate total length & extract markdown headings
    total_chars = 0
    total_words = 0
    all_headings: List[str] = []
    first_h1: Optional[str] = None

    for chunk in chunks:
        text = chunk.get("text", "")
        total_chars += len(text)
        total_words += len(text.split())

        for line in text.splitlines():
            line_str = line.strip()
            if line_str.startswith("#"):
                # Clean header formatting
                header_text = re.sub(r"^#+\s*", "", line_str)
                header_text = re.sub(r"[\*_`]", "", header_text).strip()
                if header_text:
                    if line_str.startswith("# ") and first_h1 is None:
                        first_h1 = header_text
                    if header_text not in all_headings:
                        all_headings.append(header_text)

    # Determine best title
    if meta_title and meta_title.lower() not in ["untitled", "paper title", "none"]:
        title = meta_title
    elif first_h1:
        title = first_h1
    else:
        title = pdf_path.stem.replace("_", " ")

    # Identify abstract snippet if present
    abstract_snippet = ""
    for chunk in chunks[:2]:
        text = chunk.get("text", "")
        match = re.search(r"\babstract\b[\s—:-]*(.+?)(?=\n\n|\bkeywords\b|\bindex terms\b|##)", text, re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            candidate = " ".join(candidate.split())
            if len(candidate) > 40:
                abstract_snippet = candidate[:400] + ("..." if len(candidate) > 400 else "")
                break

    return {
        "title": title,
        "author": meta_author if meta_author else "N/A",
        "subject_or_doi": meta_subject if meta_subject else "N/A",
        "pdf_filename": pdf_path.name,
        "pdf_size_bytes": pdf_path.stat().st_size,
        "pdf_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
        "page_count": page_count,
        "character_count": total_chars,
        "word_count": total_words,
        "sections": all_headings[:20],  # Keep top 20 structural headings
        "abstract_snippet": abstract_snippet,
    }


def format_frontmatter(metadata: Dict[str, Any]) -> str:
    """Format metadata dictionary as standard YAML frontmatter."""
    escaped_title = metadata["title"].replace('"', '\\"')
    escaped_author = metadata["author"].replace('"', '\\"')
    escaped_subject = metadata["subject_or_doi"].replace('"', '\\"')

    lines = [
        "---",
        f'title: "{escaped_title}"',
        f'pdf_file: "{metadata["pdf_filename"]}"',
        f'pages: {metadata["page_count"]}',
        f'words: {metadata["word_count"]}',
        f'characters: {metadata["character_count"]}',
        f'filesize_mb: {metadata["pdf_size_mb"]}',
        f'doi_or_subject: "{escaped_subject}"',
        f'author: "{escaped_author}"',
        "sections:",
    ]
    for sec in metadata["sections"]:
        sec_clean = sec.replace('"', '\\"')
        lines.append(f'  - "{sec_clean}"')
    lines.append("---\n")
    return "\n".join(lines)


def convert_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    extract_images: bool = False,
    force: bool = True,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Convert a single PDF file to Markdown with frontmatter and page demarcation.
    Returns: (success: bool, metadata: dict, message: str)
    """
    output_md_path = output_dir / f"{pdf_path.stem}.md"

    if output_md_path.exists() and not force:
        return True, None, f"Skipped (already exists): {output_md_path.name}"

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        return False, None, f"Failed to open PDF {pdf_path.name}: {e}"

    try:
        # Extract page chunks with layout parsing & OCR on diagrams
        chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            write_images=extract_images,
            image_path=str(output_dir / "images" / pdf_path.stem) if extract_images else "",
        )
    except Exception as e:
        doc.close()
        return False, None, f"Conversion error on {pdf_path.name}: {e}"

    metadata = extract_metadata_and_outline(pdf_path, doc, chunks)
    doc.close()

    # Build agent-optimized markdown document
    content_parts: List[str] = [format_frontmatter(metadata)]

    total_pages = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        page_header = (
            f"\n\n<!-- ============================================================ -->\n"
            f"<!-- PAGE {idx} OF {total_pages}                                                  -->\n"
            f"<!-- ============================================================ -->\n\n"
        )
        content_parts.append(page_header)
        content_parts.append(chunk.get("text", "").strip())

    full_markdown = "".join(content_parts) + "\n"

    try:
        output_md_path.write_text(full_markdown, encoding="utf-8")
        rel_path = output_md_path.relative_to(Path.cwd()) if output_md_path.is_relative_to(Path.cwd()) else output_md_path
        metadata["markdown_relative_path"] = str(rel_path)
        metadata["markdown_filename"] = output_md_path.name
        return True, metadata, f"Generated: {rel_path} ({metadata['word_count']:,} words, {metadata['page_count']} pages)"
    except Exception as e:
        return False, None, f"Write error for {output_md_path.name}: {e}"


def generate_index_files(catalog: List[Dict[str, Any]], output_dir: Path, papers_dir: Path) -> None:
    """
    Generate INDEX.md and papers_catalog.json in the output directory.
    """
    total_papers = len(catalog)
    total_pages = sum(p["page_count"] for p in catalog)
    total_words = sum(p["word_count"] for p in catalog)

    # Sort catalog by title alphabetically
    catalog = sorted(catalog, key=lambda x: x["title"].lower())

    # 1. Write machine-readable JSON catalog
    json_path = output_dir / "papers_catalog.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    # 2. Write human/agent readable INDEX.md
    index_lines = [
        "# Research Papers Markdown Catalog",
        "",
        "> This catalog lists research papers converted from PDF into structured, agent-ready Markdown format.",
        "> Each markdown document contains YAML frontmatter, preserved tables/formulas, and page demarcation tags (`<!-- PAGE X OF Y -->`).",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Papers:** {total_papers}",
        f"- **Total Pages:** {total_pages:,}",
        f"- **Total Word Count:** {total_words:,} words",
        f"- **Catalog JSON:** [`papers_catalog.json`](./papers_catalog.json)",
        "",
        "## Papers Directory",
        "",
        "| # | Paper Title | Pages | Words | Markdown Link | Original PDF |",
        "|---|-------------|-------|-------|---------------|--------------|",
    ]

    for idx, item in enumerate(catalog, start=1):
        md_name = item["markdown_filename"]
        pdf_name = item["pdf_filename"]
        title = item["title"]
        pages = item["page_count"]
        words = f"{item['word_count']:,}"
        
        # Relative link from output_dir to original pdf
        try:
            pdf_rel = os.path.relpath(papers_dir / pdf_name, output_dir)
        except ValueError:
            pdf_rel = str(papers_dir / pdf_name)

        index_lines.append(
            f"| {idx} | **{title}** | {pages} | {words} | [{md_name}](./{md_name}) | [PDF]({pdf_rel}) |"
        )

    index_lines.append("\n## Paper Section Outlines\n")
    for idx, item in enumerate(catalog, start=1):
        index_lines.append(f"### {idx}. {item['title']}\n")
        index_lines.append(f"- **File:** [`{item['markdown_filename']}`](./{item['markdown_filename']})")
        index_lines.append(f"- **Pages:** {item['page_count']} | **Words:** {item['word_count']:,}")
        if item.get("subject_or_doi") and item["subject_or_doi"] != "N/A":
            index_lines.append(f"- **Subject / DOI:** {item['subject_or_doi']}")
        if item.get("abstract_snippet"):
            index_lines.append(f"- **Abstract Snippet:** {item['abstract_snippet']}")
        if item.get("sections"):
            sections_str = ", ".join(f"`{s}`" for s in item["sections"][:8])
            index_lines.append(f"- **Key Sections:** {sections_str}")
        index_lines.append("")

    index_md_path = output_dir / "INDEX.md"
    index_md_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert research paper PDFs into clean agent-friendly Markdown format."
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("papers"),
        help="Directory containing source PDF files (default: papers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers/markdown"),
        help="Directory to save converted Markdown files (default: papers/markdown)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Convert a specific single PDF file instead of entire directory",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract embedded images to output_dir/images/<paper_stem>/ (default: False)",
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Skip files that already exist in output directory",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        pdf_files = [args.file]
        if not args.file.exists():
            print(f"Error: Target file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    else:
        papers_dir: Path = args.papers_dir
        if not papers_dir.exists():
            print(f"Error: Papers directory not found: {papers_dir}", file=sys.stderr)
            sys.exit(1)
        pdf_files = sorted([f for f in papers_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])

    if not pdf_files:
        print("No PDF files found to convert.")
        sys.exit(0)

    print("=" * 70)
    print(" Research Paper PDF -> Agent-Ready Markdown Converter")
    print(f" Source PDFs:     {len(pdf_files)} file(s)")
    print(f" Output Directory: {output_dir.resolve()}")
    print(f" Extract Images:  {args.extract_images}")
    print(f" Overwrite Mode:  {args.force}")
    print("=" * 70)

    start_time = time.time()
    catalog: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Converting: {pdf_path.name}...")
        t0 = time.time()
        ok, metadata, msg = convert_single_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            extract_images=args.extract_images,
            force=args.force,
        )
        elapsed = time.time() - t0
        if ok and metadata:
            success_count += 1
            catalog.append(metadata)
            print(f"       -> OK in {elapsed:.2f}s | {metadata['word_count']:,} words | {metadata['page_count']} pages")
        elif ok and not metadata:
            print(f"       -> {msg}")
        else:
            fail_count += 1
            print(f"       -> ERROR: {msg}")

    # Generate INDEX.md and papers_catalog.json if batch mode
    if catalog:
        papers_dir = args.papers_dir if not args.file else args.file.parent
        generate_index_files(catalog, output_dir, papers_dir)
        print(f"\nCreated master index: {output_dir / 'INDEX.md'}")
        print(f"Created catalog JSON: {output_dir / 'papers_catalog.json'}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Conversion Complete in {total_time:.2f}s!")
    print(f"Successfully converted: {success_count}/{len(pdf_files)} papers")
    if fail_count > 0:
        print(f"Failed: {fail_count} papers")
    print("=" * 70)


if __name__ == "__main__":
    main()
