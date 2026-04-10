"""YouTube video transcript loader + timestamp-based chunking with proxy support."""

from __future__ import annotations

from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os
import json
import tempfile
import shutil
import subprocess

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
        webshare_api_key = os.getenv("WEBSHARE_API_KEY")
        if not webshare_api_key:
            try:
                import streamlit as st
                webshare_api_key = st.secrets.get("WEBSHARE_API_KEY")
            except Exception:
                pass
        if webshare_api_key:
            try:
                self.proxy_api = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(webshare_api_key)
                )
                print("[VIDEO] Webshare proxy configured")
            except Exception as e:
                print(f"[VIDEO] Webshare proxy config failed: {e}")

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract the YouTube video ID from a URL."""
        if "watch?v=" in url:
            video_id = url.split("watch?v=")[-1].split("&")[0].strip()
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0].strip()
        elif "/shorts/" in url:
            video_id = url.split("/shorts/")[-1].split("?")[0].strip()
        else:
            video_id = url.strip()
        
        if not video_id or len(video_id) < 5:
            raise ValueError("Invalid YouTube link")
        return video_id

    def _fetch_with_api(self, api: YouTubeTranscriptApi, video_id: str) -> List[Dict]:
        """Attempt to fetch transcript using given API instance."""
        try:
            transcript = api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
            return [{"text": e.text, "start": e.start, "duration": e.duration} for e in transcript]
        except Exception:
            pass
        
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

    @staticmethod
    def _fetch_with_ytdlp(video_id: str) -> List[Dict]:
        """Fallback: use yt-dlp to extract subtitles (works on cloud servers)."""
        url = f"https://www.youtube.com/watch?v={video_id}"
        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "subs")
            used_strategy = None
            strategy_errors = []

            cookie_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
            extractor_variants = [
                "youtube:player_client=android,web;skip=dash,hls",
                "youtube:player_client=web,ios",
                "youtube:player_client=tv_embedded",
            ]

            def _build_opts(extractor_args: str) -> Dict:
                opts = {
                    "skip_download": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": ["en", "en-US", "en-GB", "en-*", "hi", "ta", "te", "ml", "kn"],
                    "subtitlesformat": "json3/vtt/best",
                    "outtmpl": outtmpl,
                    "quiet": True,
                    "no_warnings": True,
                    "extractor_args": {"youtube": extractor_args.replace("youtube:", "")},
                    "force_ipv4": True,
                }
                if cookie_file and os.path.exists(cookie_file):
                    opts["cookiefile"] = cookie_file
                return opts

            def _parse_saved_subtitles() -> List[Dict]:
                # Find subtitle files saved by yt-dlp and parse them.
                for fname in os.listdir(tmpdir):
                    if fname.endswith(".json3"):
                        with open(os.path.join(tmpdir, fname), "r", encoding="utf-8") as f:
                            data = json.load(f)
                        events = data.get("events", [])
                        result = []
                        for ev in events:
                            segs = ev.get("segs", [])
                            text = "".join(s.get("utf8", "") for s in segs).strip()
                            if text and text != "\n":
                                start_ms = ev.get("tStartMs", 0)
                                dur_ms = ev.get("dDurationMs", 0)
                                result.append({
                                    "text": text,
                                    "start": start_ms / 1000.0,
                                    "duration": dur_ms / 1000.0,
                                })
                        if result:
                            return result

                for fname in os.listdir(tmpdir):
                    if fname.endswith(".vtt"):
                        with open(os.path.join(tmpdir, fname), "r", encoding="utf-8") as f:
                            content = f.read()
                        parsed = VideoProcessor._parse_vtt(content)
                        if parsed:
                            return parsed
                return []

            # Strategy A: python module
            try:
                import yt_dlp  # type: ignore[import-not-found]

                for extractor_args in extractor_variants:
                    try:
                        ydl_opts = _build_opts(extractor_args)
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])
                        used_strategy = f"python-module[{extractor_args}]"
                        parsed = _parse_saved_subtitles()
                        if parsed:
                            return parsed
                    except Exception as e:
                        strategy_errors.append(f"py:{extractor_args}:{str(e)[:120]}")
                        continue
            except ImportError:
                pass

            if not used_strategy:
                # Strategy B: CLI binary (common in some deployments)
                ytdlp_cmd = shutil.which("yt-dlp") or shutil.which("yt_dlp")
                if not ytdlp_cmd:
                    raise RuntimeError(
                        "yt-dlp not installed (module + CLI missing). "
                        "Add 'yt-dlp' to requirements.txt for deployment."
                    )

                for extractor_args in extractor_variants:
                    cmd = [
                        ytdlp_cmd,
                        "--skip-download",
                        "--write-subs",
                        "--write-auto-subs",
                        "--sub-langs",
                        "en,en-US,en-GB,en-*,hi,ta,te,ml,kn",
                        "--sub-format",
                        "json3/vtt/best",
                        "--force-ipv4",
                        "--extractor-args",
                        extractor_args,
                        "-o",
                        outtmpl,
                        url,
                    ]
                    if cookie_file and os.path.exists(cookie_file):
                        cmd.extend(["--cookies", cookie_file])

                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        used_strategy = f"cli[{extractor_args}]"
                        parsed = _parse_saved_subtitles()
                        if parsed:
                            return parsed
                    except subprocess.CalledProcessError as e:
                        err = (e.stderr or e.stdout or "")[:220]
                        strategy_errors.append(f"cli:{extractor_args}:{err}")
                        continue

            # Final parse attempt in case files were written by a previous attempt.
            parsed = _parse_saved_subtitles()
            if parsed:
                return parsed

        if used_strategy:
            raise ValueError(f"yt-dlp ({used_strategy}) ran but no subtitle tracks were available.")
        detail = " | ".join(strategy_errors[:3]) if strategy_errors else "no strategy details"
        raise ValueError(f"yt-dlp could not extract subtitles for this video ({detail}).")

    @staticmethod
    def _parse_vtt(content: str) -> List[Dict]:
        """Parse WebVTT subtitle content into transcript entries."""
        lines = content.split("\n")
        result = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Look for timestamp lines like "00:00:01.000 --> 00:00:04.000"
            if "-->" in line:
                parts = line.split("-->")
                start_str = parts[0].strip()
                # Parse HH:MM:SS.mmm or MM:SS.mmm
                start_secs = VideoProcessor._vtt_time_to_secs(start_str)
                # Collect text lines
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    clean = re.sub(r"<[^>]+>", "", lines[i].strip())
                    if clean:
                        text_lines.append(clean)
                    i += 1
                text = " ".join(text_lines)
                if text:
                    result.append({"text": text, "start": start_secs, "duration": 0})
            else:
                i += 1
        return result

    @staticmethod
    def _vtt_time_to_secs(time_str: str) -> float:
        """Convert VTT timestamp to seconds."""
        time_str = time_str.strip()
        parts = time_str.replace(",", ".").split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return 0.0

    def load_transcript(self, url: str) -> List[Dict]:
        """Load YouTube transcript with multiple fallback strategies."""
        video_id = self.extract_video_id(url)
        errors = []
        
        # Strategy 1: Direct youtube-transcript-api
        try:
            return self._fetch_with_api(self.api, video_id)
        except Exception as e:
            errors.append(f"Direct: {str(e)[:120]}")
            print(f"[VIDEO] Direct fetch failed: {errors[-1]}")
        
        # Strategy 2: youtube-transcript-api with Webshare proxy
        if self.proxy_api:
            try:
                print("[VIDEO] Trying Webshare proxy...")
                return self._fetch_with_api(self.proxy_api, video_id)
            except Exception as e:
                errors.append(f"Proxy: {str(e)[:120]}")
                print(f"[VIDEO] Proxy fetch failed: {errors[-1]}")

        # Strategy 3: yt-dlp (most reliable from cloud servers)
        try:
            print("[VIDEO] Trying yt-dlp fallback...")
            return self._fetch_with_ytdlp(video_id)
        except Exception as e:
            errors.append(f"yt-dlp: {str(e)[:120]}")
            print(f"[VIDEO] yt-dlp failed: {errors[-1]}")

        # All strategies failed
        raise ValueError(
            f"Could not fetch transcript after trying all methods.\n" +
            "\n".join(errors) +
            "\nTip: Some videos have no subtitles/captions, are region-restricted, or require cookies. "
            "If needed, set YTDLP_COOKIES_FILE in .env to a valid YouTube cookies.txt file."
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
