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
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", str(file_path))
        return docs

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
