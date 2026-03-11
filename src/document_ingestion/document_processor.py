"""Document processing module for loading and splitting documents."""

import json
import csv
from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ---------- loaders ----------

    def load_from_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        # First try PyPDFLoader (fast, works for text-based PDFs)
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))

        # Check if extracted text is meaningful (not empty/placeholder)
        total_text = "".join(d.page_content.strip() for d in docs)
        # Remove bullet-only content
        cleaned = total_text.replace("•", "").replace("\n", "").strip()

        if len(cleaned) > 100:
            # PyPDFLoader got real text, use it
            # Filter out empty pages
            return [d for d in docs if len(d.page_content.strip().replace("•", "").replace("\n", "").strip()) > 10]

        # Fallback: PDF is image-based (scanned). Use OCR via pymupdf + easyocr
        print(f"[DOC PROCESSOR] PyPDFLoader got no usable text from {file_path}. Attempting OCR...")
        try:
            return self._ocr_pdf(file_path)
        except Exception as e:
            print(f"[DOC PROCESSOR] OCR failed: {e}. Returning empty docs.")
            return docs

    def _ocr_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Extract text from image-based PDFs using pymupdf + pytesseract/easyocr."""
        import pymupdf
        from PIL import Image
        import io

        doc = pymupdf.open(str(file_path))
        total_pages = len(doc)
        all_docs = []

        # Try to get an OCR function (pytesseract preferred, easyocr fallback)
        ocr_fn = self._get_ocr_function()
        if ocr_fn is None:
            doc.close()
            raise RuntimeError(
                "No OCR backend available. Install Tesseract OCR "
                "(conda install -c conda-forge tesseract) or easyocr (pip install easyocr) "
                "to process scanned/image-based PDFs."
            )

        for page_num in range(total_pages):
            page = doc[page_num]
            # Render page to image at 200 DPI for good OCR quality
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # Run OCR
            page_text = ocr_fn(img)

            if page_text and len(page_text.strip()) > 10:
                all_docs.append(Document(
                    page_content=page_text.strip(),
                    metadata={
                        "source": str(file_path),
                        "page": page_num,
                        "extraction_method": "ocr",
                    }
                ))

        doc.close()
        print(f"[DOC PROCESSOR] OCR extracted text from {len(all_docs)}/{total_pages} pages")
        return all_docs

    @staticmethod
    def _get_ocr_function():
        """Return a callable that takes a PIL Image and returns extracted text string."""
        # Try pytesseract first (faster, more reliable if Tesseract binary exists)
        try:
            import pytesseract
            import shutil

            # Find Tesseract binary
            tesseract_path = shutil.which("tesseract")
            if not tesseract_path:
                # Common conda install location on Windows
                import os
                conda_path = os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "bin", "tesseract.exe")
                if os.path.isfile(conda_path):
                    tesseract_path = conda_path
                # Also check anaconda3 base
                home = os.path.expanduser("~")
                for candidate in [
                    os.path.join(home, "anaconda3", "Library", "bin", "tesseract.exe"),
                    os.path.join(home, "miniconda3", "Library", "bin", "tesseract.exe"),
                ]:
                    if os.path.isfile(candidate):
                        tesseract_path = candidate
                        break

            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                # Verify it works
                pytesseract.get_tesseract_version()
                print(f"[DOC PROCESSOR] Using pytesseract with {tesseract_path}")
                return lambda img: pytesseract.image_to_string(img)
        except Exception as e:
            print(f"[DOC PROCESSOR] pytesseract not available: {e}")

        # Try easyocr as fallback
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[DOC PROCESSOR] Using easyocr")
            return lambda img: "\n".join(reader.readtext(np.array(img), detail=0, paragraph=True))
        except Exception as e:
            print(f"[DOC PROCESSOR] easyocr not available: {e}")

        return None

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
        return docs

    def load_from_docx(self, file_path: Union[str, Path]) -> List[Document]:
        """Load Microsoft Word documents."""
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
            d.metadata["file_type"] = "docx"
        return docs

    def load_from_csv(self, file_path: Union[str, Path]) -> List[Document]:
        """Load CSV files."""
        loader = CSVLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
            d.metadata["file_type"] = "csv"
        return docs

    def load_from_json(self, file_path: Union[str, Path]) -> List[Document]:
        """Load JSON files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to text representation
        if isinstance(data, list):
            content = "\n\n".join([json.dumps(item, indent=2) for item in data])
        else:
            content = json.dumps(data, indent=2)
        
        doc = Document(
            page_content=content,
            metadata={"source": str(file_path), "file_type": "json"}
        )
        return [doc]

    def load_from_markdown(self, file_path: Union[str, Path]) -> List[Document]:
        """Load Markdown files."""
        loader = UnstructuredMarkdownLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
            d.metadata["file_type"] = "markdown"
        return docs

    def load_from_html(self, file_path: Union[str, Path]) -> List[Document]:
        """Load HTML files."""
        loader = UnstructuredHTMLLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
            d.metadata["file_type"] = "html"
        return docs

    def load_from_python(self, file_path: Union[str, Path]) -> List[Document]:
        """Load Python source files."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
            d.metadata["file_type"] = "python"
        return docs

    def load_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Load documents from URLs, PDFs, TXT, DOCX, CSV, JSON, MD, HTML, or Python files."""
        docs: List[Document] = []
        
        # Supported extensions mapping
        extension_loaders = {
            ".pdf": self.load_from_pdf,
            ".txt": self.load_from_txt,
            ".docx": self.load_from_docx,
            ".doc": self.load_from_docx,
            ".csv": self.load_from_csv,
            ".json": self.load_from_json,
            ".md": self.load_from_markdown,
            ".markdown": self.load_from_markdown,
            ".html": self.load_from_html,
            ".htm": self.load_from_html,
            ".py": self.load_from_python,
        }
        
        for src in sources:
            src_str = str(src)
            if src_str.startswith("http://") or src_str.startswith("https://"):
                docs.extend(self.load_from_url(src_str))
                continue

            path = Path(src_str)
            suffix = path.suffix.lower()
            
            if suffix in extension_loaders:
                docs.extend(extension_loaders[suffix](path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src_str}. "
                    f"Supported: URL, .pdf, .txt, .docx, .csv, .json, .md, .html, .py"
                )
        return docs

    # ---------- splitter ----------

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    # ---------- end-to-end ----------

    def process_sources(self, sources: List[Union[str, Path]]) -> List[Document]:
        docs = self.load_documents(sources)
        return self.split_documents(docs)
    
    
    
