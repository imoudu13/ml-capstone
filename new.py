from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ML model
model = SentenceTransformer('all-MiniLM-L6-v2')
keywords = [
    "Bodily Injury", "Property Damage", "Liability", "Coverage",
    "Deductible", "Limit of Insurance", "Premium", "Exclusion", "Endorsement"
]
keyword_embeddings = model.encode(keywords)


def extract_text_with_metadata(pdf_path):
    """
    Extract text, page numbers, and line metadata from a PDF.
    """
    content_metadata = []
    reader = PdfReader(pdf_path)

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            lines = page_text.splitlines()
            for line_num, line in enumerate(lines):
                content_metadata.append({
                    "page": page_num + 1,
                    "line": line_num + 1,
                    "text": line.strip()
                })
    return content_metadata


def extract_coverage_lines(pdf_content):
    """
    Extract lines with coverage-related terms or numbers.
    Returns a dictionary with metadata (e.g., page, line) and text.
    """
    coverage_lines = {}
    coverage_pattern = re.compile(r"\b(?:CAD|USD)?\s*\$?\d+(?:,\d{3})*(?:\.\d+)?\b")  # Match amounts

    for entry in pdf_content:
        text = entry['text']
        if any(keyword in text for keyword in keywords) or coverage_pattern.search(text):
            key = f"page-{entry['page']}_line-{entry['line']}"
            coverage_lines[key] = {
                "page": entry['page'],
                "line": entry['line'],
                "text": text
            }
    return coverage_lines


def extract_values(text):
    """
    Extract numerical values and their associated currencies from a text line.
    """
    value_pattern = re.compile(r"(\b(?:CAD|USD)?\s*\$?\d+(?:,\d{3})*(?:\.\d+)?\b)")
    matches = value_pattern.findall(text)
    return [match.replace(",", "") for match in matches]


def compare_coverage_lines(pdf1_lines, pdf2_lines):
    """
    Compare extracted coverage lines between two PDFs using embeddings.
    Also checks for value differences.
    Returns a list of differences with metadata.
    """
    differences = []

    for key1, line1 in pdf1_lines.items():
        embedding1 = model.encode([line1['text']])
        best_match = None
        best_similarity = 0

        for key2, line2 in pdf2_lines.items():
            embedding2 = model.encode([line2['text']])
            similarity = cosine_similarity(embedding1, embedding2)[0][0]

            if similarity > best_similarity:
                best_match = line2
                best_similarity = similarity

        # Check if values differ for similar lines
        if best_similarity >= 0.8:
            values1 = extract_values(line1['text'])
            values2 = extract_values(best_match['text']) if best_match else []

            if values1 != values2:
                differences.append({
                    "pdf1": line1,
                    "pdf2": best_match,
                    "similarity": best_similarity,
                    "value_difference": {
                        "pdf1_values": values1,
                        "pdf2_values": values2
                    }
                })
        else:
            differences.append({
                "pdf1": line1,
                "pdf2": best_match or {"page": None, "line": None, "text": "Not Found"},
                "similarity": best_similarity
            })

    return differences


def process_and_compare_pdfs(pdf1_path, pdf2_path):
    """
    Process two PDFs and compare coverage data, including value mismatches.
    """
    # Extract content with metadata
    pdf1_content = extract_text_with_metadata(pdf1_path)
    pdf2_content = extract_text_with_metadata(pdf2_path)

    # Extract coverage-related lines
    pdf1_lines = extract_coverage_lines(pdf1_content)
    pdf2_lines = extract_coverage_lines(pdf2_content)

    # Compare the coverage lines
    differences = compare_coverage_lines(pdf1_lines, pdf2_lines)
    return differences
