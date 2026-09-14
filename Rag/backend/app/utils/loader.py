from pathlib import Path

from pypdf import PdfReader

def load_document(file_path: str) -> list[dict]:
    """Load a document while retaining the page or section boundary."""
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [
            {"text": page.extract_text() or "", "page_number": page_number}
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    if path.suffix.lower() in {".md", ".txt"}:
        return [{"text": path.read_text(encoding="utf-8"), "page_number": None}]

    raise ValueError(f"Unsupported document type: {path.suffix}")

def load_pdf(file_path: str) -> str:
    """Backward-compatible PDF loader used by older examples."""
    return "\n".join(page["text"] for page in load_document(file_path))


# Page 1
# Page 1 – Company Overview, Vision, Mission &
# Core Values
# Welcome to NextGen Tech Solutions Pvt Ltd. This Human Resources Policy document defines
# the principles, rules, and structured processes that govern employment within the organization.
# Vision:
# To become a globally respected technology company known for ethical innovation, operational
# excellence,
# and employee empowerment.
# Mission:
# To design scalable, secure, and intelligent digital solutions while fostering a high-performance
# culture
# that encourages ownership, accountability, and continuous improvement.
# Core Values:
# 1. Integrity – We act with honesty and transparency in all professional dealings.
# 2. Ownership – Employees are encouraged to take responsibility beyond assigned tasks.
# 3. Innovation – Continuous experimentation and improvement are encouraged.
# 4. Collaboration – Cross-functional teamwork drives company success.
# 5. Customer Excellence – Delivering measurable value to clients is a priority.
# Organizational Structure:
# The company operates through structured departments including Engineering, DevOps, HR, Sales,
# Marketing,
# Finance, and Operations. Each department functions under defined KPIs aligned with company
# goals.


