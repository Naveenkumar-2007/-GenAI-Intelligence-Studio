"""YouTube video transcript loader + timestamp-based chunking."""

from __future__ import annotations

from typing import List, Dict
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class VideoProcessor:
    """Extract transcript and convert into timestamp-aware document chunks."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract the YouTube video ID from a URL."""
        if "watch?v=" in url:
            return url.split("watch?v=")[-1].split("&")[0].strip()
        if "youtu.be/" in url:
            return url.split("youtu.be/")[-1].split("?")[0].strip()
        raise ValueError("Invalid YouTube link")

    def load_transcript(self, url: str) -> List[Dict]:
        """Load YouTube transcript."""
        video_id = self.extract_video_id(url)
        # New API: use fetch() method and convert to list of dicts
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        # Convert FetchedTranscript to list of dicts
        return [{"text": entry.text, "start": entry.start, "duration": entry.duration} for entry in transcript]

    def transcript_to_document(self, transcript: List[Dict], url: str) -> Document:
        """Convert transcript list → single Document with timestamps."""
        full_text = []
        for entry in transcript:
            t = entry.get("text", "")
            start = entry.get("start", 0)
            full_text.append(f"[{start:.1f}s] {t}")

        combined = "\n".join(full_text)
        return Document(
            page_content=combined,
            metadata={
                "source": url,
                "type": "youtube_transcript",
            },
        )

    def chunk_document(self, doc: Document) -> List[Document]:
        """Split transcript doc into timestamp-aware chunks."""
        return self.splitter.split_documents([doc])

    def process_video(self, url: str) -> List[Document]:
        """Full pipeline: URL → transcript → doc → chunks."""
        transcript = self.load_transcript(url)
        base_doc = self.transcript_to_document(transcript, url)
        chunks = self.chunk_document(base_doc)

        # keep timestamps in metadata
        for ch in chunks:
            # extract first timestamp from text
            match = re.search(r"\[(\d+\.\d+)s\]", ch.page_content)
            if match:
                ch.metadata["timestamp_start"] = float(match.group(1))

        return chunks
