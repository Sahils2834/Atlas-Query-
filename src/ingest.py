from pathlib import Path
from collections import Counter
import hashlib
import re
import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Only this test collection is recreated.
COLLECTION_NAME = "rag_documents_v2"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 60
MIN_CHUNK_WORDS = 35

BATCH_SIZE = 64
SKIP_REFERENCES = True

# Fallback reference detection settings.
REFERENCE_FALLBACK_MIN_LINES = 4
REFERENCE_FALLBACK_WINDOW = 8


# ==================================================
# SECTION DETECTION
# ==================================================

SECTION_PATTERNS = {
    "abstract": r"abstract",
    "introduction": r"introduction",

    "related_work": (
        r"related works?|background|literature review|"
        r"previous work|state of the art"
    ),

    "methodology": (
        r"methodology|methods?|proposed method|proposed approach|"
        r"approach|system architecture|framework|design"
    ),

    "experiments": (
        r"experiments?|experimental setup|implementation|"
        r"experimental methodology|experimental design"
    ),

    "results": (
        r"results?|experimental results|evaluation|"
        r"performance evaluation|findings"
    ),

    "discussion": (
        r"discussion|analysis|results and discussion"
    ),

    "limitations_future_work": (
        r"limitations? and future (?:work|directions)|"
        r"limitations?, future work"
    ),

    "limitations": (
        r"limitations?|threats to validity"
    ),

    "future_work": (
        r"future work|future directions|future scope"
    ),

    "conclusion": (
        r"conclusions?|summary|concluding remarks"
    ),

    "references": (
        r"references|bibliography|works cited|"
        r"literature cited|references and notes"
    ),

    "appendix": r"appendix(?:es)?(?:\s+[a-z])?",
}


# ==================================================
# WEBSITE / INTERFACE NOISE
# ==================================================

UI_NOISE_PATTERNS = [
    r"copy page",
    r"was this page helpful\??",
    r"yes",
    r"no",
    r"suggest edits",
    r"previous",
    r"next",
    r"back to top",
    r"skip to (?:main )?content",
    r"menu",
    r"search",
    r"navigation",
    r"table of contents",
    r"on this page",
    r"print",
    r"share",
    r"edit this page",
    r"report an issue",
    r"copy",
    r"copied",
    r"feedback",
    r"light mode",
    r"dark mode",
    r"sign in",
    r"sign up",

    # Common documentation-site fragments.
    r"see all \d+ lines",
    r"install with ai",
    r"watch on",
    r"getting started page",
    r"python typescript",
    r"view source",
    r"edit on github",
    r"getting started quickstart example projects(?: open source)?",
]

TIMESTAMP_PATTERNS = [
    r"\d{1,2}/\d{1,2}/\d{2,4}(?:,\s*\d{1,2}:\d{2}\s*[ap]m)?",
    r"\d{1,2}:\d{2}\s*[ap]m",
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
]


# ==================================================
# REFERENCE DETECTION
# ==================================================

REFERENCE_ENTRY_PATTERNS = [
    re.compile(r"^\s*\[\s*\d{1,4}\s*\]\s+"),

    re.compile(
        r"^\s*\d{1,3}[.)]\s+[A-Z][A-Za-z-]+(?:,|\s+[A-Z]\.?)"
    ),

    re.compile(
        r"\b(?:doi\s*:\s*10\.|https?://doi\.org/|arxiv\.org/abs/)\S+",
        re.I,
    ),

    re.compile(
        r"\b(?:vol\.|volume|no\.|issue|pp\.|pages)\s*\d+",
        re.I,
    ),
]


def looks_like_reference_entry(line: str) -> bool:
    """Identify a possible bibliography entry."""

    line = line.strip()

    if not line:
        return False

    signals = sum(
        bool(pattern.search(line))
        for pattern in REFERENCE_ENTRY_PATTERNS
    )

    # Example: Smith, J. (2021). Research paper title...
    author_year = bool(
        re.search(
            r"^[A-Z][A-Za-z'’-]+"
            r"(?:,\s*[A-Z]\.?)?"
            r"(?:,|\s+and\s+)[^.!?]{0,100}"
            r"\(?(?:19|20)\d{2}\)?",
            line,
        )
    )

    return signals > 0 or author_year


def has_reference_run(lines, start_index: int) -> bool:
    """
    Require multiple reference-like lines close together.

    A single DOI, citation, or author-year mention
    must not end ingestion.
    """

    end_index = min(
        len(lines),
        start_index + REFERENCE_FALLBACK_WINDOW,
    )

    reference_count = 0
    examined_lines = 0

    for index in range(start_index, end_index):
        candidate = lines[index].strip()

        if not candidate:
            continue

        examined_lines += 1

        if looks_like_reference_entry(candidate):
            reference_count += 1

        elif reference_count > 0:
            break

    return (
        reference_count >= REFERENCE_FALLBACK_MIN_LINES
        and examined_lines >= REFERENCE_FALLBACK_MIN_LINES
    )


# ==================================================
# HEADING UTILITIES
# ==================================================

def normalize_heading(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" :.-\t")


def strip_heading_number(text: str) -> str:
    return re.sub(
        r"^\s*\d+(?:\.\d+)*\s*[.)]?\s*",
        "",
        text,
    ).strip()


def detect_section_heading(line: str):
    heading = normalize_heading(line)
    heading = strip_heading_number(heading).casefold()
    heading = re.sub(r"\s+", " ", heading)

    for section, pattern in SECTION_PATTERNS.items():
        if re.fullmatch(pattern, heading, re.IGNORECASE):
            return section

    return None


# ==================================================
# PDF TEXT CLEANING
# ==================================================

def is_noise_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip()

    if not normalized:
        return True

    lowered = normalized.casefold()

    if any(
        re.fullmatch(pattern, lowered, re.I)
        for pattern in UI_NOISE_PATTERNS
    ):
        return True

    if any(
        re.fullmatch(pattern, normalized, re.I)
        for pattern in TIMESTAMP_PATTERNS
    ):
        return True

    if re.fullmatch(r"(?:page\s*)?\d{1,3}", lowered):
        return True

    if re.fullmatch(r"https?://\S+|www\.\S+", lowered):
        return True

    return False


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")

    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Repair words split across PDF lines.
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            cleaned_lines.append("")
            continue

        # Remove interface labels at the beginning.
        line = re.sub(
            r"^(?:Copy page|Copy|Skip to content)\s+",
            "",
            line,
            flags=re.IGNORECASE,
        )

        # Remove interface labels at the end.
        line = re.sub(
            r"\s+(?:Was this page helpful\??|Suggest edits)$",
            "",
            line,
            flags=re.IGNORECASE,
        )

        line = line.strip()

        if line and not is_noise_line(line):
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==================================================
# REPEATED HEADER / FOOTER DETECTION
# ==================================================

def line_fingerprint(line: str) -> str:
    line = line.casefold().strip()
    line = re.sub(r"\d+", "#", line)
    return re.sub(r"\s+", " ", line)


def find_repeated_lines(page_texts):
    counts = Counter()
    total_pages = len(page_texts)

    for text in page_texts:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        edge_lines = lines[:3] + lines[-3:]

        for line in set(edge_lines):
            if len(line) <= 100:
                counts[line_fingerprint(line)] += 1

    threshold = max(2, int(total_pages * 0.3))

    return {
        line
        for line, count in counts.items()
        if count >= threshold
    }


def remove_repeated_lines(text: str, repeated_lines: set) -> str:
    cleaned = []

    for line in text.splitlines():
        fingerprint = line_fingerprint(line)

        if fingerprint in repeated_lines:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ==================================================
# PDF EXTRACTION
# ==================================================

def load_pdf_pages():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}"
        )

    pages = []

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        with pymupdf.open(pdf_path) as pdf:
            raw_pages = [
                clean_text(page.get_text("text"))
                for page in pdf
            ]

            repeated_lines = find_repeated_lines(raw_pages)
            readable = 0

            for page_index, raw_text in enumerate(raw_pages):
                text = remove_repeated_lines(
                    raw_text,
                    repeated_lines,
                )

                text = clean_text(text)

                if not text:
                    continue

                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_index + 1,
                })

                readable += 1

            print(f"Total PDF pages: {len(pdf)}")
            print(f"Readable pages: {readable}")
            print(f"Repeated lines removed: {len(repeated_lines)}")

    if not pages:
        raise ValueError("No readable PDF text found.")

    print(f"\nTotal extracted pages: {len(pages)}")

    return pages


# ==================================================
# HEADING DETECTION
# ==================================================

def is_probable_heading(line: str) -> bool:
    """
    Conservative heading detection.

    Accept numbered and uppercase headings.
    Avoid guessing from arbitrary title-case text.
    """

    line = normalize_heading(line)
    words = line.split()

    if not line or len(line) > 90 or len(words) > 9:
        return False

    if is_noise_line(line):
        return False

    if line.endswith((".", ",", ";", "?", "!")):
        return False

    if re.search(
        r"https?://|www\.|\bdoi\b|arxiv|et al\.",
        line,
        re.I,
    ):
        return False

    if re.search(r"\b(?:19|20)\d{2}\b", line):
        return False

    if re.search(r"\[[0-9,; -]+\]", line):
        return False

    # Reject likely author/reference lines.
    if line.count(",") >= 2:
        return False

    if re.search(r"\b[A-Z]\.\s*[A-Z]?(?:\.|\s)", line):
        return False

    # Numbered section headings.
    if re.match(
        r"^\d+(?:\.\d+)*\s*[.)]?\s+[A-Za-z]",
        line,
    ):
        return True

    # Short uppercase headings.
    if (
        line.isupper()
        and len(words) <= 7
        and any(char.isalpha() for char in line)
    ):
        return True

    return False


# ==================================================
# PARAGRAPH AND SECTION PROCESSING
# ==================================================

def extract_units(page_data, section_state):
    """
    Extract paragraphs and headings.

    Section state is shared across pages belonging to the same PDF.

    Once a references boundary is detected, the remainder of that PDF
    is ignored, including any later headings or appendix labels.
    """

    lines = page_data["text"].splitlines()
    units = []

    current_section = section_state["section"]
    current_heading = section_state["heading"]
    skip_references = section_state["skip_references"]

    paragraph_lines = []
    skipped_reference_lines = 0

    def flush_paragraph():
        nonlocal paragraph_lines

        if not paragraph_lines:
            return

        paragraph = " ".join(
            line.strip()
            for line in paragraph_lines
            if line.strip()
        )

        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if paragraph:
            units.append({
                "text": paragraph,
                "section": current_section,
                "section_heading": current_heading,
                "chunk_type": "paragraph",
            })

        paragraph_lines = []

    for line_index, line in enumerate(lines):
        line = line.strip()

        if not line:
            if not skip_references:
                flush_paragraph()
            continue

        if is_noise_line(line):
            continue

        # Permanent references boundary.
        # Do not process headings or fallback detection afterward.
        if skip_references:
            skipped_reference_lines += 1
            continue

        section = detect_section_heading(line)

        if section:
            flush_paragraph()

            current_section = section
            current_heading = normalize_heading(line)

            if SKIP_REFERENCES and section == "references":
                skip_references = True

                print(
                    f"  References boundary detected: "
                    f"{page_data['source']} "
                    f"page {page_data['page']} "
                    f"({line})"
                )

            continue

        # Fallback detection for PDFs without an explicit heading.
        if (
            SKIP_REFERENCES
            and has_reference_run(lines, line_index)
        ):
            flush_paragraph()

            skip_references = True
            skipped_reference_lines += 1

            print(
                f"  Probable references boundary: "
                f"{page_data['source']} "
                f"page {page_data['page']} "
                f"(multiple consecutive citation-like entries)"
            )

            continue

        if is_probable_heading(line):
            flush_paragraph()

            current_heading = normalize_heading(line)

            units.append({
                "text": current_heading,
                "section": current_section,
                "section_heading": current_heading,
                "chunk_type": "heading",
            })

            continue

        paragraph_lines.append(line)

    if not skip_references:
        flush_paragraph()
    else:
        paragraph_lines = []

    # Persist section state for the next page of this PDF.
    section_state["section"] = current_section
    section_state["heading"] = current_heading
    section_state["skip_references"] = skip_references

    section_state["skipped_reference_lines"] = (
        section_state.get("skipped_reference_lines", 0)
        + skipped_reference_lines
    )

    return units


# ==================================================
# INTELLIGENT CHUNKING
# ==================================================

def split_large_paragraph(text: str):
    words = text.split()

    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    if step <= 0:
        raise ValueError(
            "CHUNK_SIZE_WORDS must be greater than "
            "CHUNK_OVERLAP_WORDS."
        )

    pieces = []

    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE_WORDS]

        if piece:
            pieces.append(" ".join(piece))

        if start + CHUNK_SIZE_WORDS >= len(words):
            break

    return pieces


def chunk_page(page_data, section_state):
    units = extract_units(page_data, section_state)

    chunks = []

    current_texts = []
    current_words = 0

    current_section = None
    current_heading = None

    def flush():
        nonlocal current_texts, current_words

        if not current_texts:
            return

        text = "\n\n".join(current_texts).strip()

        if len(text.split()) >= MIN_CHUNK_WORDS:
            chunks.append({
                "text": text,
                "section": current_section or "general",
                "section_heading": current_heading or "General",
                "chunk_type": "content",
            })

        current_texts = []
        current_words = 0

    for unit in units:
        unit_section = unit["section"]
        unit_heading = unit["section_heading"]

        # Never combine different sections or subsections.
        if current_words > 0 and (
            unit_section != current_section
            or unit_heading != current_heading
        ):
            flush()

        current_section = unit_section
        current_heading = unit_heading

        if unit["chunk_type"] == "heading":
            continue

        paragraph = unit["text"]
        word_count = len(paragraph.split())

        # Split unusually large paragraphs separately.
        if word_count > CHUNK_SIZE_WORDS:
            flush()

            for piece in split_large_paragraph(paragraph):
                if len(piece.split()) >= MIN_CHUNK_WORDS:
                    chunks.append({
                        "text": piece,
                        "section": current_section or "general",
                        "section_heading": current_heading or "General",
                        "chunk_type": "content",
                    })

            continue

        # Start a new chunk when the current chunk is full.
        if (
            current_words > 0
            and current_words + word_count > CHUNK_SIZE_WORDS
        ):
            previous_words = " ".join(current_texts).split()
            overlap = previous_words[-CHUNK_OVERLAP_WORDS:]

            flush()

            current_texts = (
                [" ".join(overlap)] if overlap else []
            )

            current_words = len(overlap)

        current_texts.append(paragraph)
        current_words += word_count

    flush()

    return chunks


# ==================================================
# BUILD INDEX RECORDS
# ==================================================

def build_chunks(pages):
    records = []
    seen_content = set()

    # Independent state for each PDF.
    document_states = {}

    for page_data in pages:
        source = page_data["source"]

        if source not in document_states:
            document_states[source] = {
                "section": "general",
                "heading": "General",
                "skip_references": False,
                "skipped_reference_lines": 0,
            }

        state = document_states[source]
        chunks = chunk_page(page_data, state)

        for chunk_index, chunk in enumerate(chunks):
            text = chunk["text"].strip()

            normalized = re.sub(
                r"\s+",
                " ",
                text,
            ).casefold()

            # Avoid duplicate chunks within one PDF.
            duplicate_key = (
                source,
                normalized,
            )

            if duplicate_key in seen_content:
                continue

            seen_content.add(duplicate_key)

            identity = (
                f"{source}|"
                f"{page_data['page']}|"
                f"{chunk_index}|"
                f"{normalized}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:24]

            records.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "source": source,
                    "page": int(page_data["page"]),
                    "chunk_index": int(chunk_index),
                    "word_count": len(text.split()),
                    "section": chunk["section"],
                    "section_heading": (
                        chunk["section_heading"] or "General"
                    ),
                    "chunk_type": chunk["chunk_type"],
                },
            })

    # Reference detection diagnostics.
    for source, state in document_states.items():
        print(
            f"Reference lines skipped for {source}: "
            f"{state['skipped_reference_lines']}"
        )

        if SKIP_REFERENCES and not state["skip_references"]:
            print(
                f"  INFO: no references boundary detected "
                f"in {source}; this may be normal for documentation PDFs."
            )

    return records


# ==================================================
# STORE IN CHROMADB
# ==================================================

def store_records(records):
    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    existing = [
        item.name
        for item in client.list_collections()
    ]

    # Only recreate the v2 test collection.
    if COLLECTION_NAME in existing:
        print(
            f"\nDeleting previous test collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "chunking": (
                "permanent-reference-boundary-clean-ui-"
                "section-aware-350w-overlap60-v6"
            ),
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        texts = [
            record["text"]
            for record in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=[
                record["id"]
                for record in batch
            ],
            documents=texts,
            metadatas=[
                record["metadata"]
                for record in batch
            ],
            embeddings=embeddings,
        )

        print(
            f"Stored "
            f"{min(start + BATCH_SIZE, len(records))}"
            f"/{len(records)} chunks"
        )

    return collection


# ==================================================
# MAIN
# ==================================================

def main():
    print("========== PDF INGESTION ==========")

    print("Target collection:", COLLECTION_NAME)

    print(
        "Original rag_documents collection "
        "will remain untouched."
    )

    pages = load_pdf_pages()
    records = build_chunks(pages)

    if not records:
        raise ValueError("No chunks were created.")

    print("\n========== CHUNK PREVIEW ==========")

    for index, record in enumerate(records[:8], start=1):
        metadata = record["metadata"]

        print(f"\nChunk {index}")
        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Section:", metadata["section"])
        print("Heading:", metadata["section_heading"])
        print("Words:", metadata["word_count"])
        print("Text:", record["text"][:500])

    print("\nTotal chunks created:", len(records))

    collection = store_records(records)

    print("\n========== INGESTION COMPLETE ==========")

    print("Collection:", COLLECTION_NAME)
    print("Stored chunks:", collection.count())
    print("Database:", CHROMA_DIR)
    print("Original collection remains untouched.")


if __name__ == "__main__":
    main()