"""YouTube video transcript loader + timestamp-based chunking with proxy support."""

from __future__ import annotations

from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os

# Import the API
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig


class VideoProcessor:
    """Extract transcript and convert into timestamp-aware document chunks."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Initialize API without proxy first (for local use)
        self.api = YouTubeTranscriptApi()
        
        # Check for proxy configuration from environment / Streamlit secrets
        self.proxy_api = None
        self._webshare_key = None
        webshare_api_key = os.getenv("WEBSHARE_API_KEY")
        if not webshare_api_key:
            try:
                import streamlit as st
                webshare_api_key = st.secrets.get("WEBSHARE_API_KEY")
            except Exception:
                pass
        if webshare_api_key:
            self._webshare_key = webshare_api_key
            try:
                self.proxy_api = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(webshare_api_key)
                )
                print(f"[VIDEO] Webshare proxy configured successfully")
            except Exception as e:
                print(f"[VIDEO] Webshare proxy config failed: {e}")
                self.proxy_api = None

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract the YouTube video ID from a URL."""
        # Handle various YouTube URL formats
        if "watch?v=" in url:
            video_id = url.split("watch?v=")[-1].split("&")[0].strip()
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0].strip()
        elif "/shorts/" in url:
            video_id = url.split("/shorts/")[-1].split("?")[0].strip()
        else:
            # Maybe it's just the video ID
            video_id = url.strip()
        
        if not video_id or len(video_id) < 5:
            raise ValueError("Invalid YouTube link")
        return video_id

    def _fetch_with_api(self, api: YouTubeTranscriptApi, video_id: str) -> List[Dict]:
        """Attempt to fetch transcript using given API instance."""
        # Try fetching English first
        try:
            transcript = api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
            return [{"text": e.text, "start": e.start, "duration": e.duration} for e in transcript]
        except Exception:
            pass
        
        # Fallback: List all and translate if needed
        transcript_list = api.list(video_id)
        for t in transcript_list:
            try:
                if t.is_translatable:
                    translated = t.translate('en').fetch()
                    return [{"text": e.text, "start": e.start, "duration": e.duration} for e in translated]
                else:
                    original = t.fetch()
                    return [{"text": e.text, "start": e.start, "duration": e.duration} for e in original]
            except Exception:
                continue
        
        raise ValueError("No usable transcript found.")

    def load_transcript(self, url: str) -> List[Dict]:
        """Load YouTube transcript with automatic proxy fallback."""
        video_id = self.extract_video_id(url)
        
        # Strategy 1: Try direct connection first (works locally and sometimes on cloud)
        try:
            return self._fetch_with_api(self.api, video_id)
        except Exception as direct_error:
            direct_msg = str(direct_error)
            print(f"[VIDEO] Direct fetch failed: {direct_msg[:200]}")
        
        # Strategy 2: If we have proxy configured, try with proxy
        if self.proxy_api:
            try:
                print("[VIDEO] Trying with Webshare proxy...")
                return self._fetch_with_api(self.proxy_api, video_id)
            except Exception as proxy_error:
                raise ValueError(
                    f"Both direct and proxy failed.\n"
                    f"Direct: {direct_msg[:150]}\n"
                    f"Proxy: {str(proxy_error)[:150]}"
                )

        # Strategy 3: If we have the key but proxy_api wasn't created, try creating now
        if self._webshare_key and not self.proxy_api:
            try:
                print("[VIDEO] Retrying Webshare proxy creation...")
                proxy_api = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(self._webshare_key)
                )
                return self._fetch_with_api(proxy_api, video_id)
            except Exception as retry_error:
                raise ValueError(
                    f"Direct connection blocked by YouTube and Webshare proxy failed.\n"
                    f"Direct: {direct_msg[:150]}\n"
                    f"Proxy: {str(retry_error)[:150]}\n"
                    f"Your Webshare free plan may only have datacenter proxies which YouTube also blocks. "
                    f"Try upgrading to Webshare residential proxies, or use the app locally."
                )
        
        # No proxy key at all
        raise ValueError(
            f"YouTube is blocking requests from this IP (Streamlit Cloud). "
            f"Set WEBSHARE_API_KEY in Streamlit Secrets with a Webshare.io API key "
            f"(residential proxies recommended), or run locally."
        )

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
            match = re.search(r"\[(\d+\.\d+)s\]", ch.page_content)
            if match:
                ch.metadata["timestamp_start"] = float(match.group(1))

        return chunks
