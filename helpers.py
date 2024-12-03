import re
from PyPDF2 import PdfReader
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define keywords to look for
keywords = [
    "Bodily Injury", "Property Damage", "Liability", "Coverage",
    "Deductible", "Limit of Insurance", "Premium", "Exclusion", "Endorsement"
]

keyword_embeddings = model.encode(keywords)

# Step 1: Extract text and tables from PDFs
def extract_content_from_pdf(pdf_path):
    text_content = []
    table_content = []
    reader = PdfReader(pdf_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)

    # For tables
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                table_content.append(table)

    return " ".join(text_content), table_content

# Step 2: Preprocess text
def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
    return text

# Step 3: Extract coverage-related data
def extract_coverage_data(text, tables):
    extracted_data = []

    # Process paragraphs
    lines = text.split(". ")
    for line in lines:
        line_embedding = model.encode([line])
        similarity = cosine_similarity(line_embedding, keyword_embeddings)
        if similarity.max() > 0.7:  # Threshold for relevance
            extracted_data.append(("Paragraph", line.strip()))

    # Process tables
    for table_index, table in enumerate(tables):
        for row in table:
            for cell in row:
                cell = preprocess_text(cell)
                cell_embedding = model.encode([cell])
                similarity = cosine_similarity(cell_embedding, keyword_embeddings)
                if similarity.max() > 0.7:
                    extracted_data.append((f"Table {table_index + 1}", cell.strip()))

    return extracted_data


# Step 4: Normalize and structure coverage data
def normalize_coverage_data(data):
    coverage_dict = {}
    pattern = re.compile(r"(?P<coverage>[a-zA-Z\s]+):?\s*(?P<value>[0-9,\.]+)")

    for location, line in data:
        match = pattern.search(line)
        if match:
            coverage = match.group("coverage").strip()
            value = match.group("value").strip().replace(",", "")
            coverage_dict[(location, coverage)] = value

    return coverage_dict

# Step 5: Compare coverage data
def compare_coverage_data(data1, data2):
    all_keys = set(data1.keys()).union(data2.keys())
    differences = []

    for key in all_keys:
        value1 = data1.get(key, "Not Present")
        value2 = data2.get(key, "Not Present")
        if value1 != value2:
            differences.append({
                "location": key[0],
                "coverage": key[1],
                "policy_1": value1,
                "policy_2": value2
            })
    return differences

# Step 6: Process both PDFs
def process_and_compare_pdfs(pdf1_path, pdf2_path):
    # Extract content from PDFs
    text1, tables1 = extract_content_from_pdf(pdf1_path)
    text2, tables2 = extract_content_from_pdf(pdf2_path)

    # Preprocess text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Extract and normalize coverage data
    data1 = extract_coverage_data(text1, tables1)
    data2 = extract_coverage_data(text2, tables2)
    normalized_data1 = normalize_coverage_data(data1)
    normalized_data2 = normalize_coverage_data(data2)

    # Compare data
    differences = compare_coverage_data(normalized_data1, normalized_data2)

    return differences
