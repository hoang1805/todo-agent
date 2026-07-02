"""Load a document from the web — fetch a URL and reduce it to plain text.

Stdlib only (``urllib`` + ``html.parser``): the app already runs fully local, and
a document fetch doesn't justify a new HTTP dependency. HTML is stripped to its
visible text (script/style dropped); anything else text-like passes through as-is.
The scheme allowlist and download cap are input guardrails in the §3 sense —
they bound what an arbitrary user-supplied URL can pull into the ingestion path.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_DOWNLOAD_BYTES = 2_000_000  # cap the fetch, not just the parsed text
_TIMEOUT_SECONDS = 15
_USER_AGENT = "todo-agent/1.0 (+document ingestion)"

#: Tags whose content is never visible text.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "head"})


class LoadError(Exception):
    """A fetch/extraction failure carrying a user-facing message."""


#: Tags that mark the primary content of a page. When present, everything outside
#: them (navigation, header, footer, cookie banners) is noise we'd rather not
#: ingest — it pollutes retrieval with link-text fragments.
_MAIN_TAGS = frozenset({"main", "article"})

#: Below this much main-tag text, assume the tag was decorative and keep the
#: whole page instead.
_MIN_MAIN_CHARS = 200


class _TextExtractor(HTMLParser):
    """Collect the visible text of an HTML page (and its <title>).

    Keeps two streams: everything, and only what's inside <main>/<article> —
    the caller prefers the latter when it's substantial.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._main_depth = 0
        self._in_title = False
        self.title = ""
        self.parts: list[str] = []
        self.main_parts: list[str] = []

    def _emit(self, piece: str) -> None:
        self.parts.append(piece)
        if self._main_depth:
            self.main_parts.append(piece)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _MAIN_TAGS:
            self._main_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "br":  # usually never closed — break on open
            self._emit("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        # Block-level closers become paragraph breaks so structure survives —
        # the chunking router uses blank lines to detect "structured" text.
        if tag in ("p", "div", "section", "article", "li", "tr",
                   "h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("\n\n")
        if tag in _MAIN_TAGS and self._main_depth:
            self._main_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if not self._skip_depth and data.strip():
            self._emit(data)


def _clean(parts: list[str]) -> str:
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)  # collapse break runs
    return text.strip()


def _html_to_text(html: str) -> tuple[str, str]:
    """Strip *html* to ``(title, visible_text)`` with paragraph breaks kept.

    Prefers the page's <main>/<article> content when it carries substantial text,
    falling back to the full body otherwise.
    """
    parser = _TextExtractor()
    parser.feed(html)
    main_text = _clean(parser.main_parts)
    text = main_text if len(main_text) >= _MIN_MAIN_CHARS else _clean(parser.parts)
    return parser.title.strip(), text


def fetch_url_text(url: str) -> tuple[str, str]:
    """Fetch *url* and return ``(title, plain_text)``; raise :class:`LoadError` on failure.

    Only ``http(s)`` is allowed (no ``file://`` escapes into the local disk), and
    the download is capped at :data:`MAX_DOWNLOAD_BYTES`.
    """
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LoadError("Please give a full http(s):// URL.")

    try:
        request = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_DOWNLOAD_BYTES + 1)
            content_type = (response.headers.get("Content-Type") or "").lower()
    except Exception as exc:  # noqa: BLE001 — network errors become one user-facing kind
        raise LoadError(f"Could not fetch the URL ({exc}).") from exc

    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise LoadError(f"The page is too large (> {MAX_DOWNLOAD_BYTES // 1_000_000} MB).")

    charset_match = re.search(r"charset=([\w-]+)", content_type)
    encoding = charset_match.group(1) if charset_match else "utf-8"
    try:
        body = raw.decode(encoding, errors="replace")
    except LookupError:  # unknown charset name from the server
        body = raw.decode("utf-8", errors="replace")

    if "html" in content_type or body.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
        title, text = _html_to_text(body)
    else:
        title, text = "", body.strip()

    if not text:
        raise LoadError("The page contained no readable text.")
    return title or parsed.netloc, text
