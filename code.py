import os
import re
import pymupdf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

CV_FOLDER = "cvs"
JOB_FOLDER = "job_description"
JOB_FILE = os.path.join(JOB_FOLDER, "job.txt")


# ==========================================
# 2. CLEAN TEXT
# ==========================================

def clean_text(text):
    """Clean extracted text before comparison."""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#./ -]", " ", text)

    return text.strip()


# ==========================================
# 3. EXTRACT CANDIDATE NAME
# ==========================================

def extract_candidate_name(pdf_path):
    """
    Try to identify the candidate's name from
    the first page of a PDF CV.
    """

    try:
        with pymupdf.open(pdf_path) as pdf:

            if len(pdf) == 0:
                return "Name Not Found"

            text = pdf[0].get_text("text")

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        # Common headings and details that are not names
        excluded_exact = {
            "resume",
            "curriculum vitae",
            "professional summary",
            "summary",
            "contact",
            "contact information",
            "work experience",
            "professional experience",
            "education",
            "technical skills",
            "skills",
            "projects",
            "profile",
            "references",
        }

        excluded_words = {
            "developer",
            "engineer",
            "analyst",
            "manager",
            "specialist",
            "intern",
            "consultant",
            "designer",
            "scientist",
            "administrator",
        }

        for line in lines[:12]:

            lower_line = line.lower().strip()

            # Skip common headings
            if lower_line in excluded_exact:
                continue

            # Skip contact information
            if any(marker in lower_line for marker in [
                "@",
                "http://",
                "https://",
                "www.",
                "linkedin",
                "phone:",
                "email:",
                "mobile:",
                "address:",
                "location:",
            ]):
                continue

            # Skip lines containing digits
            if re.search(r"\d", line):
                continue

            # Remove common punctuation around a name
            cleaned_line = re.sub(
                r"^[\W_]+|[\W_]+$",
                "",
                line
            ).strip()

            words = cleaned_line.split()

            # A basic name heuristic: 2–4 alphabetic words
            if not 2 <= len(words) <= 4:
                continue

            if not all(
                re.fullmatch(
                    r"[A-Za-z]+(?:[-'][A-Za-z]+)*",
                    word
                )
                for word in words
            ):
                continue

            # Avoid returning obvious job titles
            if any(
                word.lower() in excluded_words
                for word in words
            ):
                continue

            return cleaned_line

        return "Name Not Found"

    except Exception as error:
        print(
            f"Name extraction error in "
            f"{os.path.basename(pdf_path)}: {error}"
        )

        return "Name Not Found"


# ==========================================
# 4. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_path):
    """Extract text from every page of a PDF."""

    try:
        text = ""

        with pymupdf.open(pdf_path) as pdf:

            for page in pdf:
                text += page.get_text("text") + "\n"

        return clean_text(text)

    except Exception as error:

        print(
            f"Error reading "
            f"{os.path.basename(pdf_path)}: {error}"
        )

        return ""


# ==========================================
# 5. READ JOB DESCRIPTION
# ==========================================

def read_job_description():
    """Read the job description from job.txt."""

    if not os.path.isfile(JOB_FILE):
        raise FileNotFoundError(
            f"Job description not found: {JOB_FILE}"
        )

    with open(
        JOB_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    text = clean_text(text)

    if not text:
        raise ValueError(
            "The job description file is empty."
        )

    return text


# ==========================================
# 6. ANALYZE CVs
# ==========================================

def analyze_cvs():
    """Compare CVs with the job description."""

    print("\n" + "=" * 65)
    print("                CV ANALYZER")
    print("=" * 65)

    # Check CV folder
    if not os.path.isdir(CV_FOLDER):
        print(f"CV folder not found: {CV_FOLDER}")
        return

    # Read job description
    job_description = read_job_description()

    # Find PDF files
    cv_files = sorted([
        filename
        for filename in os.listdir(CV_FOLDER)
        if filename.lower().endswith(".pdf")
        and os.path.isfile(
            os.path.join(CV_FOLDER, filename)
        )
    ])

    if not cv_files:
        print("No PDF CVs found in the cvs folder.")
        return

    print(f"\nPDF files found: {len(cv_files)}")
    print("Extracting candidate information...\n")

    # Each candidate stores:
    # filename, candidate name, extracted CV text
    candidates = []

    for filename in cv_files:

        file_path = os.path.join(
            CV_FOLDER,
            filename
        )

        candidate_name = extract_candidate_name(
            file_path
        )

        cv_text = extract_text_from_pdf(
            file_path
        )

        if not cv_text:
            print(f"Skipping unreadable CV: {filename}")
            continue

        candidates.append((
            filename,
            candidate_name,
            cv_text
        ))

        print(
            f"Processed: {filename} "
            f"| Candidate: {candidate_name}"
        )

    if not candidates:
        print("\nNo readable CVs were found.")
        return

    # ======================================
    # 7. CONVERT TEXT INTO TF-IDF VECTORS
    # ======================================

    documents = (
        [job_description]
        + [
            cv_text
            for _, _, cv_text in candidates
        ]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    # First vector = job description
    job_vector = tfidf_matrix[0:1]

    # Remaining vectors = candidate CVs
    cv_vectors = tfidf_matrix[1:]

    # ======================================
    # 8. CALCULATE SIMILARITY SCORES
    # ======================================

    scores = cosine_similarity(
        job_vector,
        cv_vectors
    )[0]

    # ======================================
    # 9. SORT CANDIDATES
    # ======================================

    results = []

    for candidate, score in zip(candidates, scores):

        filename, candidate_name, cv_text = candidate

        results.append((
            filename,
            candidate_name,
            float(score)
        ))

    results.sort(
        key=lambda item: item[2],
        reverse=True
    )

    # ======================================
    # 10. DISPLAY RESULTS
    # ======================================

    print("\n" + "=" * 65)
    print("               ANALYSIS RESULTS")
    print("=" * 65)

    print(f"\nJob Description: {JOB_FILE}")
    print(f"CVs discovered: {len(cv_files)}")
    print(f"CVs analyzed: {len(candidates)}")

    print("\nCandidates ranked by text similarity:\n")

    for rank, (filename, name, score) in enumerate(
        results,
        start=1
    ):

        percentage = score * 100

        if percentage >= 50:
            category = "High Text Similarity"

        elif percentage >= 25:
            category = "Moderate Text Similarity"

        else:
            category = "Low Text Similarity"

        print(f"Rank: {rank}")
        print(f"Candidate Name: {name}")
        print(f"CV File: {filename}")
        print(f"Similarity Score: {percentage:.2f}%")
        print(f"Category: {category}")
        print("-" * 65)

    print("\nCV analysis completed successfully!")


# ==========================================
# 11. START THE PROGRAM
# ==========================================

if __name__ == "__main__":

    try:
        analyze_cvs()

    except FileNotFoundError as error:
        print(f"\nFile Error: {error}")

    except ValueError as error:
        print(f"\nInput Error: {error}")

    except Exception as error:
        print(f"\nUnexpected Error: {error}")
