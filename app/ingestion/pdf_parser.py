"""
PDF Ingestion Pipeline for NCERT Class 12 Physics Part 1
Extracts structured content with metadata: chapters, headings, formulas, tables
"""

import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class PhysicsChunk:
    chunk_id: str
    page: int
    chapter: str
    chapter_num: int
    heading: str
    subheading: str
    content: str
    formulas: List[str]
    tables: List[str]
    source: str = "NCERT Physics Class 12 Part 1"
    chunk_type: str = "text"  # text | formula | definition | table


# NCERT Physics Part 1 chapter map (pages approximate)
CHAPTER_MAP = {
    1: {"name": "Electric Charges and Fields", "start": 1, "end": 44},
    2: {"name": "Electrostatic Potential and Capacitance", "start": 45, "end": 90},
    3: {"name": "Current Electricity", "start": 91, "end": 136},
    4: {"name": "Moving Charges and Magnetism", "start": 137, "end": 180},
    5: {"name": "Magnetism and Matter", "start": 181, "end": 216},
    6: {"name": "Electromagnetic Induction", "start": 217, "end": 252},
    7: {"name": "Alternating Current", "start": 253, "end": 294},
    8: {"name": "Electromagnetic Waves", "start": 295, "end": 318},
}

# Patterns for formula detection
FORMULA_PATTERNS = [
    r'[A-Za-z]\s*=\s*[\d\w\s\+\-\*/\^\(\)\.]+',  # basic equations
    r'F\s*=\s*k[Qq]',          # Coulomb's law
    r'E\s*=\s*',               # Electric field
    r'V\s*=\s*',               # Potential
    r'∇[²×·]',                 # Vector calculus
    r'∮|∫|∑',                  # Integral/sum notation
    r'\d+\s*×\s*10\^?\d+',     # Scientific notation
    r'ε₀|μ₀|π\s*ε',            # Physical constants
]

HEADING_PATTERNS = [
    r'^\d+\.\d+\s+[A-Z]',           # Section numbers like 2.3 HEADING
    r'^[A-Z][A-Z\s]{4,}$',          # ALL CAPS headings
    r'^\d+\.\s+[A-Z]',              # Chapter headings
]


class PDFIngestionPipeline:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.chunks: List[PhysicsChunk] = []
        self.raw_pages: List[Dict] = []

    def infer_chapter(self, page_num: int, text: str) -> tuple[int, str]:
        """Infer chapter number and name from page number and content."""
        # Try to detect from text first
        ch_match = re.search(r'Chapter\s+(\d+)', text, re.IGNORECASE)
        if ch_match:
            num = int(ch_match.group(1))
            if num in CHAPTER_MAP:
                return num, CHAPTER_MAP[num]["name"]

        # Fall back to page range
        for ch_num, ch_info in CHAPTER_MAP.items():
            if ch_info["start"] <= page_num <= ch_info["end"]:
                return ch_num, ch_info["name"]

        return 0, "Preliminary / Appendix"

    def extract_formulas(self, text: str) -> List[str]:
        """Extract physics formulas from text."""
        formulas = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for pattern in FORMULA_PATTERNS:
                if re.search(pattern, line):
                    if len(line) < 200:  # filter out full paragraphs
                        formulas.append(line)
                    break
        return list(set(formulas))

    def extract_headings(self, text: str) -> tuple[str, str]:
        """Extract heading and subheading from text block."""
        heading = ""
        subheading = ""
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Section heading like "2.3 Electric Field Lines"
            sec_match = re.match(r'^(\d+\.\d+)\s+(.+)$', line)
            if sec_match and len(line) < 100:
                subheading = line
                continue

            # Chapter heading like "ELECTRIC CHARGES AND FIELDS"
            if re.match(r'^[A-Z][A-Z\s]{5,}$', line) and len(line) < 80:
                heading = line.title()
                continue

        return heading, subheading

    def parse_pdf(self) -> List[Dict]:
        """Parse PDF and extract structured page content."""
        logger.info(f"Parsing PDF: {self.pdf_path}")

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        doc = fitz.open(str(self.pdf_path))
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)

            ch_num, ch_name = self.infer_chapter(page_num + 1, text)
            heading, subheading = self.extract_headings(text)
            formulas = self.extract_formulas(text)

            # Extract tables (blocks with grid-like structure)
            tables = []
            for block in blocks:
                if block[6] == 1:  # image block (tables often rendered as images)
                    tables.append(f"[Table on page {page_num + 1}]")

            pages.append({
                "page": page_num + 1,
                "chapter_num": ch_num,
                "chapter": ch_name,
                "heading": heading,
                "subheading": subheading,
                "content": text.strip(),
                "formulas": formulas,
                "tables": tables,
                "source": "NCERT Physics Class 12 Part 1",
                "word_count": len(text.split()),
            })

        doc.close()
        self.raw_pages = pages
        logger.success(f"Parsed {len(pages)} pages successfully")
        return pages

    def section_aware_chunk(self, pages: List[Dict],
                             chunk_size: int = 900,
                             chunk_overlap: int = 175) -> List[PhysicsChunk]:
        """
        Section-aware + semantic chunking strategy.

        Tradeoffs:
        - Section-aware: Preserves context within logical sections, better for 
          topic-based retrieval. Risk: sections can be very long or very short.
        - Semantic chunking: Groups by meaning. Needs embedding pass.
        - Hybrid approach used here: split by section boundaries first, 
          then enforce token window with overlap for continuity.
        """
        logger.info("Chunking with section-aware strategy...")
        chunks = []
        chunk_counter = 0

        # Group pages by chapter
        chapter_groups: Dict[int, List[Dict]] = {}
        for page in pages:
            ch = page["chapter_num"]
            if ch not in chapter_groups:
                chapter_groups[ch] = []
            chapter_groups[ch].append(page)

        for ch_num, ch_pages in chapter_groups.items():
            # Build a full chapter text with page markers
            full_text = ""
            page_markers = {}  # char_position -> page_num

            for pg in ch_pages:
                marker = f"\n[PAGE:{pg['page']}]\n"
                page_markers[len(full_text)] = pg['page']
                full_text += marker + pg['content'] + "\n"

            # Split on section boundaries first
            section_splits = re.split(
                r'(?=\n\d+\.\d+\s+[A-Z]|\n[A-Z][A-Z\s]{5,}\n)',
                full_text
            )

            for section in section_splits:
                if not section.strip():
                    continue

                words = section.split()
                if not words:
                    continue

                # Determine current page from markers in section
                current_page = ch_pages[0]["page"] if ch_pages else 1
                page_match = re.search(r'\[PAGE:(\d+)\]', section)
                if page_match:
                    current_page = int(page_match.group(1))

                # Get heading from section
                _, subheading = self.extract_headings(section)
                formulas = self.extract_formulas(section)

                # Sliding window chunking within section
                i = 0
                while i < len(words):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)

                    # Clean up page markers from content
                    chunk_text_clean = re.sub(r'\[PAGE:\d+\]', '', chunk_text).strip()

                    if len(chunk_text_clean) < 50:  # skip tiny fragments
                        i += chunk_size - chunk_overlap
                        continue

                    chunk_id = f"ch{ch_num:02d}_p{current_page:03d}_c{chunk_counter:04d}"
                    chunk_counter += 1

                    chunk = PhysicsChunk(
                        chunk_id=chunk_id,
                        page=current_page,
                        chapter=ch_pages[0]["chapter"],
                        chapter_num=ch_num,
                        heading=ch_pages[0]["heading"],
                        subheading=subheading,
                        content=chunk_text_clean,
                        formulas=formulas if i == 0 else [],  # attach formulas to first chunk
                        tables=ch_pages[0]["tables"] if i == 0 else [],
                        chunk_type="formula" if len(formulas) > 2 else "text",
                    )
                    chunks.append(chunk)
                    i += chunk_size - chunk_overlap

        self.chunks = chunks
        logger.success(f"Created {len(chunks)} chunks across {len(chapter_groups)} chapters")
        return chunks

    def run(self, chunk_size: int = 900, chunk_overlap: int = 175) -> List[PhysicsChunk]:
        """Run complete ingestion pipeline."""
        pages = self.parse_pdf()
        chunks = self.section_aware_chunk(pages, chunk_size, chunk_overlap)
        return chunks

    def save_chunks(self, output_path: str):
        """Save chunks to JSON for inspection."""
        data = [asdict(c) for c in self.chunks]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} chunks to {output_path}")

    def get_stats(self) -> Dict:
        """Return ingestion statistics."""
        if not self.chunks:
            return {}
        chapters = set(c.chapter for c in self.chunks)
        avg_len = sum(len(c.content.split()) for c in self.chunks) / len(self.chunks)
        formula_chunks = sum(1 for c in self.chunks if c.chunk_type == "formula")
        return {
            "total_chunks": len(self.chunks),
            "total_pages": len(self.raw_pages),
            "chapters": len(chapters),
            "avg_chunk_words": round(avg_len, 1),
            "formula_chunks": formula_chunks,
            "chapter_list": sorted(chapters),
        }
