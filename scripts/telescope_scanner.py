#!/usr/bin/env python3
"""
Research Telescope Paper Scanner
================================
Scans research papers across ALL scientific domains to find revolutionary
research opportunities, gaps, and cross-domain connections.

Usage:
    python telescope_scanner.py --scan                  # Scan only, save results
    python telescope_scanner.py --scan --email           # Scan and send digest
    python telescope_scanner.py --scan --dry-run         # Scan without email
    python telescope_scanner.py --email-only             # Email latest results
    python telescope_scanner.py --scan --sources arxiv   # Single source
    python telescope_scanner.py --scan --max-papers 50   # Limit results

Sources: ArXiv, bioRxiv, Semantic Scholar, CrossRef
Output:  JSON files in telescope/results/, HTML email digest
"""

import argparse
import hashlib
import json
import logging
import os
import re
import smtplib
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Load .env file if present (no dependency needed)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "telescope"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"
DIGESTS_DIR = BASE_DIR / "digests"

RECIPIENT_EMAIL = "naladala@uw.edu"

# SMTP config — set these env vars or edit directly
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.office365.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "naladala@uw.edu")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", SMTP_USER)

# Rate limiting (seconds between API calls)
RATE_LIMITS = {
    "arxiv": 3.0,
    "biorxiv": 1.0,
    "semantic_scholar": 3.0,
    "crossref": 1.0,
}

# ArXiv categories to scan
ARXIV_CATEGORIES = [
    "cs.NE", "cs.LG", "cs.AI", "cs.CV",
    "q-bio.NC", "q-bio.QM",
    "physics.comp-ph", "cond-mat", "astro-ph",
    "stat.ML", "math.OC", "eess.SP",
]

# Search keywords organized by theme
SEARCH_KEYWORDS = [
    # Core connectome/neuro
    "connectome neural topology",
    "biological neural network architecture",
    "reservoir computing echo state",
    "brain-inspired computing neuromorphic",
    # Gaps and benchmarks
    "computational gap analysis methodological",
    "novel benchmark systematic comparison",
    "underexplored dataset large-scale evaluation",
    # Cross-domain applications
    "drug discovery machine learning gap",
    "protein folding computational gap",
    "materials discovery neural network",
    "astronomical data neural networks",
    "particle physics machine learning",
    "social network topology computational",
    "economic modeling computational gap",
    # Specific techniques
    "liquid state machine biological",
    "graph neural architecture search",
    "small-world network computation",
]

# Gap detection patterns (regex)
GAP_PATTERNS = [
    r"(?:we|this)\s+leave[s]?\s+(?:this|it|the)\s+(?:for|to)\s+future\s+work",
    r"future\s+(?:work|direction|research|investigation)",
    r"remain[s]?\s+(?:an?\s+)?open\s+(?:question|problem|challenge)",
    r"beyond\s+the\s+scope\s+of\s+(?:this|the\s+current)",
    r"limit(?:ation|ed)\s+(?:by|to|in)",
    r"not\s+(?:yet\s+)?(?:been\s+)?(?:explored|investigated|studied|addressed)",
    r"unexplored",
    r"underexplored",
    r"gap\s+in\s+(?:the\s+)?(?:literature|research|knowledge)",
    r"lack(?:s|ing)?\s+(?:of\s+)?(?:systematic|comprehensive|large-scale)",
    r"no\s+(?:existing|prior|previous)\s+(?:work|study|benchmark)",
    r"(?:surprisingly|remarkably)\s+(?:little|few)\s+(?:work|attention|research)",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telescope")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def paper_id(title: str, authors: str = "") -> str:
    """Generate a deterministic hash ID for deduplication."""
    raw = (title.lower().strip() + "|" + authors.lower().strip()).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def rate_limit(source: str):
    """Sleep to respect API rate limits."""
    delay = RATE_LIMITS.get(source, 1.0)
    time.sleep(delay)


def safe_get(url: str, params: dict = None, headers: dict = None,
             timeout: int = 30, source: str = "generic") -> Optional[requests.Response]:
    """GET with error handling and rate limiting."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        rate_limit(source)
        return resp
    except requests.RequestException as e:
        log.warning(f"[{source}] Request failed: {e}")
        return None


def time_of_day() -> str:
    """Return morning/afternoon/evening based on current hour."""
    h = datetime.now().hour
    if h < 12:
        return "morning"
    elif h < 18:
        return "afternoon"
    return "evening"


# ---------------------------------------------------------------------------
# Paper data model
# ---------------------------------------------------------------------------

def make_paper(
    title: str,
    authors: list[str],
    abstract: str,
    url: str,
    source: str,
    published: str = "",
    categories: list[str] = None,
    doi: str = "",
    citation_count: int = 0,
    extra: dict = None,
) -> dict:
    """Create a normalized paper record."""
    return {
        "id": paper_id(title, ", ".join(authors[:3])),
        "title": title.strip(),
        "authors": authors,
        "abstract": abstract.strip(),
        "url": url,
        "source": source,
        "published": published,
        "categories": categories or [],
        "doi": doi,
        "citation_count": citation_count,
        "extra": extra or {},
        "fetched_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Source: ArXiv
# ---------------------------------------------------------------------------

class ArXivFetcher:
    """Fetch papers from ArXiv API (Atom/XML)."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def fetch(self, categories: list[str], keywords: list[str],
              max_results: int = 100, days_back: int = 7) -> list[dict]:
        papers = []
        seen_ids = set()

        # Strategy 1: category-based recent papers
        for cat in categories:
            query = f"cat:{cat}"
            per_cat = max(1, min(max_results // len(categories), 30))
            batch = self._query(query, max_results=per_cat)
            for p in batch:
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    papers.append(p)

        # Strategy 2: keyword search across all categories
        for kw in keywords[:8]:  # limit to avoid excessive API calls
            query = f'all:"{kw}"'
            batch = self._query(query, max_results=10)
            for p in batch:
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    papers.append(p)

        log.info(f"[ArXiv] Fetched {len(papers)} papers")
        return papers

    def _query(self, search_query: str, max_results: int = 20) -> list[dict]:
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = safe_get(self.BASE_URL, params=params, source="arxiv", timeout=60)
        if not resp:
            return []
        return self._parse_atom(resp.text)

    def _parse_atom(self, xml_text: str) -> list[dict]:
        """Parse ArXiv Atom XML response without xml.etree (simple regex parse)."""
        papers = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
        for entry in entries:
            title = self._extract(entry, "title").replace("\n", " ").strip()
            abstract = self._extract(entry, "summary").replace("\n", " ").strip()
            published = self._extract(entry, "published")[:10]

            # Extract authors
            authors = re.findall(r"<name>(.*?)</name>", entry)

            # Extract URL
            links = re.findall(r'<id>(.*?)</id>', entry)
            url = links[0] if links else ""

            # Extract categories
            cats = re.findall(r'<category[^>]*term="([^"]*)"', entry)

            if title and abstract:
                papers.append(make_paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    source="arxiv",
                    published=published,
                    categories=cats,
                ))
        return papers

    @staticmethod
    def _extract(text: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Source: bioRxiv
# ---------------------------------------------------------------------------

class BioRxivFetcher:
    """Fetch papers from bioRxiv API."""

    BASE_URL = "https://api.biorxiv.org/details/biorxiv"

    SUBJECTS = ["neuroscience", "bioinformatics", "computational-biology"]

    def fetch(self, max_results: int = 50, days_back: int = 7) -> list[dict]:
        papers = []
        seen = set()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/{start_date}/{end_date}/0/100"
        resp = safe_get(url, source="biorxiv", timeout=60)
        if not resp:
            return []

        try:
            data = resp.json()
        except Exception:
            log.warning("[bioRxiv] Failed to parse JSON")
            return []

        collection = data.get("collection", [])
        for item in collection:
            category = item.get("category", "").lower().replace(" ", "-")
            # Filter to relevant subjects
            if not any(s in category for s in self.SUBJECTS):
                continue

            title = item.get("title", "")
            pid = paper_id(title)
            if pid in seen:
                continue
            seen.add(pid)

            authors_raw = item.get("authors", "")
            authors = [a.strip() for a in authors_raw.split(";") if a.strip()]

            papers.append(make_paper(
                title=title,
                authors=authors,
                abstract=item.get("abstract", ""),
                url=f"https://doi.org/{item.get('doi', '')}",
                source="biorxiv",
                published=item.get("date", ""),
                categories=[category],
                doi=item.get("doi", ""),
            ))

            if len(papers) >= max_results:
                break

        log.info(f"[bioRxiv] Fetched {len(papers)} papers")
        return papers


# ---------------------------------------------------------------------------
# Source: Semantic Scholar
# ---------------------------------------------------------------------------

class SemanticScholarFetcher:
    """Fetch papers from Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    FIELDS = "title,authors,abstract,url,year,citationCount,externalIds,fieldsOfStudy"

    def fetch(self, keywords: list[str], max_results: int = 50) -> list[dict]:
        papers = []
        seen = set()

        for kw in keywords[:6]:  # limit queries
            batch = self._search(kw, limit=min(max_results // 6, 15))
            for p in batch:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    papers.append(p)

        log.info(f"[Semantic Scholar] Fetched {len(papers)} papers")
        return papers

    def _search(self, query: str, limit: int = 10) -> list[dict]:
        params = {
            "query": query,
            "limit": limit,
            "fields": self.FIELDS,
            "year": f"{datetime.now().year - 1}-",
        }
        resp = safe_get(self.SEARCH_URL, params=params, source="semantic_scholar")
        if not resp:
            return []

        try:
            data = resp.json()
        except Exception:
            return []

        papers = []
        for item in data.get("data", []):
            if not item.get("title") or not item.get("abstract"):
                continue

            authors = [
                a.get("name", "") for a in (item.get("authors") or [])
            ]
            ext_ids = item.get("externalIds") or {}

            papers.append(make_paper(
                title=item["title"],
                authors=authors,
                abstract=item.get("abstract", ""),
                url=item.get("url", ""),
                source="semantic_scholar",
                published=str(item.get("year", "")),
                categories=item.get("fieldsOfStudy") or [],
                doi=ext_ids.get("DOI", ""),
                citation_count=item.get("citationCount", 0),
            ))
        return papers

    def get_citations(self, paper_doi_or_id: str, limit: int = 5) -> list[dict]:
        """Follow citation graph for a given paper."""
        url = f"{self.BASE_URL}/paper/{paper_doi_or_id}/citations"
        params = {"fields": self.FIELDS, "limit": limit}
        resp = safe_get(url, params=params, source="semantic_scholar")
        if not resp:
            return []
        try:
            data = resp.json()
        except Exception:
            return []

        papers = []
        for item in data.get("data", []):
            citing = item.get("citingPaper", {})
            if not citing.get("title"):
                continue
            authors = [a.get("name", "") for a in (citing.get("authors") or [])]
            papers.append(make_paper(
                title=citing["title"],
                authors=authors,
                abstract=citing.get("abstract", ""),
                url=citing.get("url", ""),
                source="semantic_scholar_citation",
                published=str(citing.get("year", "")),
                citation_count=citing.get("citationCount", 0),
            ))
        return papers


# ---------------------------------------------------------------------------
# Source: CrossRef
# ---------------------------------------------------------------------------

class CrossRefFetcher:
    """Fetch papers from CrossRef API."""

    SEARCH_URL = "https://api.crossref.org/works"

    def fetch(self, keywords: list[str], max_results: int = 30) -> list[dict]:
        papers = []
        seen = set()

        for kw in keywords[:4]:
            batch = self._search(kw, rows=min(max_results // 4, 10))
            for p in batch:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    papers.append(p)

        log.info(f"[CrossRef] Fetched {len(papers)} papers")
        return papers

    def _search(self, query: str, rows: int = 10) -> list[dict]:
        params = {
            "query": query,
            "rows": rows,
            "sort": "deposited",
            "order": "desc",
            "filter": f"from-pub-date:{datetime.now().year - 1}",
            "select": "DOI,title,author,abstract,URL,published-print,subject,is-referenced-by-count",
        }
        headers = {"User-Agent": "ResearchTelescope/1.0 (mailto:naladala@uw.edu)"}
        resp = safe_get(self.SEARCH_URL, params=params, headers=headers, source="crossref")
        if not resp:
            return []

        try:
            data = resp.json()
        except Exception:
            return []

        papers = []
        for item in data.get("message", {}).get("items", []):
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""
            if not title:
                continue

            abstract = item.get("abstract", "")
            # CrossRef abstracts often have JATS XML tags
            abstract = re.sub(r"<[^>]+>", "", abstract)

            authors = []
            for a in (item.get("author") or []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)

            pub_date = ""
            pp = item.get("published-print") or item.get("published-online") or {}
            parts = pp.get("date-parts", [[]])[0]
            if parts:
                pub_date = "-".join(str(p) for p in parts)

            papers.append(make_paper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=item.get("URL", ""),
                source="crossref",
                published=pub_date,
                doi=item.get("DOI", ""),
                citation_count=item.get("is-referenced-by-count", 0),
                categories=item.get("subject", []),
            ))
        return papers


# ---------------------------------------------------------------------------
# Analysis Engine
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """Score and analyze papers for research opportunities."""

    def __init__(self, existing_knowledge: dict = None):
        self.knowledge = existing_knowledge or {}
        self._compiled_gaps = [re.compile(p, re.IGNORECASE) for p in GAP_PATTERNS]

    def analyze_paper(self, paper: dict) -> dict:
        """Run full analysis on a single paper, returning scores and flags."""
        abstract = paper.get("abstract", "")
        title = paper.get("title", "")
        text = f"{title}. {abstract}"

        # Gap detection
        gaps_found = self._detect_gaps(text)

        # Keyword relevance
        relevance_hits = self._keyword_relevance(text)

        # Compute scores
        capitalize = self._capitalize_score(paper, gaps_found, relevance_hits)
        novelty = self._novelty_score(paper, gaps_found)
        feasibility = self._feasibility_score(paper, text)
        connection = self._connection_score(paper, relevance_hits)

        overall = round((capitalize + novelty + feasibility + connection) / 4, 1)

        return {
            **paper,
            "scores": {
                "capitalize": capitalize,
                "novelty": novelty,
                "feasibility": feasibility,
                "connection": connection,
                "overall": overall,
            },
            "gaps_detected": gaps_found,
            "relevance_hits": relevance_hits,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _detect_gaps(self, text: str) -> list[str]:
        """Find gap/future-work patterns in the text."""
        found = []
        for pattern in self._compiled_gaps:
            matches = pattern.findall(text)
            if matches:
                found.extend(matches)
        return found[:10]  # cap at 10

    def _keyword_relevance(self, text: str) -> list[str]:
        """Check which of our research keywords appear in the text."""
        text_lower = text.lower()
        hits = []
        kw_atoms = [
            "connectome", "neural topology", "reservoir computing", "echo state",
            "neuromorphic", "brain-inspired", "biological neural", "graph neural",
            "benchmark", "systematic comparison", "drug discovery", "protein folding",
            "materials discovery", "particle physics", "social network topology",
            "small-world", "scale-free", "liquid state machine",
            "computational gap", "underexplored", "large-scale evaluation",
        ]
        for kw in kw_atoms:
            if kw in text_lower:
                hits.append(kw)
        return hits

    def _capitalize_score(self, paper: dict, gaps: list, hits: list) -> int:
        """How likely can we turn this into a published paper? (1-10)"""
        score = 3  # baseline
        score += min(len(gaps), 3)           # gaps detected boost
        score += min(len(hits), 2)           # relevance boost
        if paper.get("citation_count", 0) > 10:
            score += 1                       # building on cited work
        if any(kw in " ".join(hits) for kw in ["benchmark", "systematic comparison"]):
            score += 1                       # benchmark papers are easier to extend
        return min(score, 10)

    def _novelty_score(self, paper: dict, gaps: list) -> int:
        """How novel is the gap/opportunity? (1-10)"""
        score = 4
        score += min(len(gaps), 3)
        abstract = paper.get("abstract", "").lower()
        novelty_signals = ["first", "novel", "unprecedented", "new approach", "we propose"]
        for sig in novelty_signals:
            if sig in abstract:
                score += 1
        return min(score, 10)

    def _feasibility_score(self, paper: dict, text: str) -> int:
        """Can we do this with public datasets and our compute? (1-10)"""
        score = 5
        text_lower = text.lower()
        # Public data signals
        public_signals = [
            "publicly available", "open source", "github", "kaggle",
            "mnist", "cifar", "imagenet", "openml", "zenodo",
            "public dataset", "open access",
        ]
        for sig in public_signals:
            if sig in text_lower:
                score += 1
        # Compute cost signals (negative)
        heavy_signals = ["tpu pod", "1000 gpu", "petabyte", "supercomputer"]
        for sig in heavy_signals:
            if sig in text_lower:
                score -= 2
        return max(1, min(score, 10))

    def _connection_score(self, paper: dict, hits: list) -> int:
        """How well does this connect to our existing knowledge? (1-10)"""
        score = 2
        # Direct connectome relevance
        core_hits = [h for h in hits if h in [
            "connectome", "neural topology", "reservoir computing",
            "biological neural", "brain-inspired", "small-world",
            "echo state", "neuromorphic", "graph neural",
        ]]
        score += min(len(core_hits) * 2, 6)
        # Cross-domain bonus
        cats = paper.get("categories", [])
        domains = set()
        for c in cats:
            if c.startswith("cs."):
                domains.add("cs")
            elif c.startswith("q-bio"):
                domains.add("bio")
            elif c.startswith("physics") or c.startswith("astro"):
                domains.add("physics")
            elif c.startswith("stat"):
                domains.add("stat")
            else:
                domains.add("other")
        if len(domains) > 1:
            score += 1  # cross-domain paper
        return min(score, 10)


# ---------------------------------------------------------------------------
# Cross-Domain Connection Detector
# ---------------------------------------------------------------------------

class ConnectionDetector:
    """Detect cross-domain connections between papers."""

    DOMAIN_MAP = {
        "cs": "Computer Science",
        "q-bio": "Biology",
        "physics": "Physics",
        "stat": "Statistics",
        "math": "Mathematics",
        "astro": "Astrophysics",
        "cond-mat": "Condensed Matter",
        "eess": "Engineering",
    }

    def find_connections(self, papers: list[dict]) -> list[dict]:
        """Group papers by domain and find cross-domain technique overlaps."""
        domain_papers = defaultdict(list)
        for p in papers:
            for cat in p.get("categories", []):
                prefix = cat.split(".")[0] if "." in cat else cat
                domain_papers[prefix].append(p)

        connections = []
        domains = list(domain_papers.keys())

        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                d1_name = self.DOMAIN_MAP.get(d1, d1)
                d2_name = self.DOMAIN_MAP.get(d2, d2)
                # Find papers sharing method keywords
                for p1 in domain_papers[d1][:10]:
                    for p2 in domain_papers[d2][:10]:
                        overlap = self._method_overlap(p1, p2)
                        if overlap:
                            connections.append({
                                "paper_a": {"title": p1["title"], "domain": d1_name, "id": p1["id"]},
                                "paper_b": {"title": p2["title"], "domain": d2_name, "id": p2["id"]},
                                "shared_methods": overlap,
                                "connection_type": "cross_domain_technique",
                            })
        log.info(f"[Connections] Found {len(connections)} cross-domain links")
        return connections[:30]  # cap

    def _method_overlap(self, p1: dict, p2: dict) -> list[str]:
        """Find shared methodological keywords between two papers."""
        methods = [
            "graph neural", "transformer", "attention mechanism",
            "reinforcement learning", "variational", "diffusion",
            "monte carlo", "bayesian", "kernel", "embedding",
            "contrastive learning", "self-supervised", "autoencoder",
            "recurrent", "convolutional", "generative", "optimization",
            "simulation", "topology", "spectral", "clustering",
        ]
        t1 = (p1.get("title", "") + " " + p1.get("abstract", "")).lower()
        t2 = (p2.get("title", "") + " " + p2.get("abstract", "")).lower()
        shared = [m for m in methods if m in t1 and m in t2]
        return shared


# ---------------------------------------------------------------------------
# Citation Traversal
# ---------------------------------------------------------------------------

class CitationTraverser:
    """Follow citation chains from high-scoring papers."""

    def __init__(self):
        self.ss = SemanticScholarFetcher()

    def traverse(self, papers: list[dict], score_threshold: float = 7.0,
                 max_papers: int = 5) -> list[dict]:
        """Get citing papers for highest-scored findings."""
        top = [p for p in papers if p.get("scores", {}).get("overall", 0) >= score_threshold]
        top = sorted(top, key=lambda x: x["scores"]["overall"], reverse=True)[:max_papers]

        discovered = []
        for p in top:
            doi = p.get("doi", "")
            if not doi:
                continue
            log.info(f"[Citations] Traversing citations for: {p['title'][:60]}...")
            citing = self.ss.get_citations(f"DOI:{doi}", limit=3)
            for c in citing:
                c["extra"]["discovered_via"] = p["id"]
                c["extra"]["discovery_method"] = "citation_traversal"
            discovered.extend(citing)

        log.info(f"[Citations] Discovered {len(discovered)} papers via traversal")
        return discovered


# ---------------------------------------------------------------------------
# Results Storage
# ---------------------------------------------------------------------------

class ResultsStore:
    """Save and load scan results as JSON."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.dir = results_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: dict) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.dir / f"scan_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"Results saved to {path}")
        return path

    def load_latest(self) -> Optional[dict]:
        files = sorted(self.dir.glob("scan_*.json"), reverse=True)
        if not files:
            return None
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_paper_ids(self) -> set:
        """Load all previously seen paper IDs for dedup across scans."""
        ids = set()
        for f in self.dir.glob("scan_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for p in data.get("papers", []):
                    ids.add(p.get("id", ""))
            except Exception:
                continue
        return ids


# ---------------------------------------------------------------------------
# Email Digest
# ---------------------------------------------------------------------------

class DigestEmailer:
    """Format and send HTML email digest."""

    def format_digest(self, results: dict) -> str:
        """Generate HTML email from scan results."""
        papers = results.get("papers", [])
        connections = results.get("connections", [])
        meta = results.get("metadata", {})

        # Sort by overall score
        papers = sorted(papers, key=lambda x: x.get("scores", {}).get("overall", 0), reverse=True)
        top_papers = papers[:15]

        tod = time_of_day()
        date_str = datetime.now().strftime("%B %d, %Y")

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
.header p {{ margin: 0; opacity: 0.8; font-size: 14px; }}
.stats {{ display: flex; gap: 15px; margin-top: 15px; }}
.stat {{ background: rgba(255,255,255,0.15); padding: 8px 14px; border-radius: 8px; font-size: 13px; }}
.paper {{ background: white; border-radius: 10px; padding: 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.paper h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
.paper h3 a {{ color: #0f3460; text-decoration: none; }}
.paper h3 a:hover {{ text-decoration: underline; }}
.meta {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
.abstract {{ font-size: 13px; line-height: 1.5; color: #555; margin-bottom: 10px; }}
.scores {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.score {{ background: #e8f4f8; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.score.high {{ background: #d4edda; color: #155724; }}
.score.mid {{ background: #fff3cd; color: #856404; }}
.score.low {{ background: #f8d7da; color: #721c24; }}
.gaps {{ margin-top: 8px; padding: 8px; background: #fffbe6; border-radius: 6px; font-size: 12px; }}
.section-title {{ font-size: 18px; font-weight: 700; margin: 25px 0 12px 0; padding-bottom: 6px; border-bottom: 2px solid #0f3460; }}
.connection {{ background: #f0f7ff; border-left: 3px solid #0f3460; padding: 12px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 13px; }}
.footer {{ text-align: center; font-size: 12px; color: #999; margin-top: 30px; padding: 15px; }}
</style></head>
<body>
<div class="header">
    <h1>Research Telescope Digest</h1>
    <p>{date_str} &mdash; {tod.capitalize()} scan</p>
    <div class="stats">
        <div class="stat">Papers scanned: {meta.get('total_fetched', 0)}</div>
        <div class="stat">After dedup: {meta.get('after_dedup', 0)}</div>
        <div class="stat">Cross-domain links: {len(connections)}</div>
        <div class="stat">Sources: {', '.join(meta.get('sources', []))}</div>
    </div>
</div>
"""

        # Top papers
        html += '<div class="section-title">Top Research Opportunities</div>\n'
        for i, p in enumerate(top_papers, 1):
            scores = p.get("scores", {})
            overall = scores.get("overall", 0)

            def score_class(v):
                if v >= 7: return "high"
                if v >= 4: return "mid"
                return "low"

            abstract_short = p.get("abstract", "")[:300]
            if len(p.get("abstract", "")) > 300:
                abstract_short += "..."

            gaps_html = ""
            if p.get("gaps_detected"):
                gap_list = ", ".join(p["gaps_detected"][:3])
                gaps_html = f'<div class="gaps"><strong>Gaps detected:</strong> {gap_list}</div>'

            html += f"""
<div class="paper">
    <h3>#{i}. <a href="{p.get('url', '#')}" target="_blank">{p['title']}</a></h3>
    <div class="meta">{', '.join(p.get('authors', [])[:3])} &bull; {p.get('source', '')} &bull; {p.get('published', '')}</div>
    <div class="abstract">{abstract_short}</div>
    <div class="scores">
        <span class="score {score_class(overall)}">Overall: {overall}/10</span>
        <span class="score {score_class(scores.get('capitalize', 0))}">Capitalize: {scores.get('capitalize', 0)}/10</span>
        <span class="score {score_class(scores.get('novelty', 0))}">Novelty: {scores.get('novelty', 0)}/10</span>
        <span class="score {score_class(scores.get('feasibility', 0))}">Feasibility: {scores.get('feasibility', 0)}/10</span>
        <span class="score {score_class(scores.get('connection', 0))}">Connection: {scores.get('connection', 0)}/10</span>
    </div>
    {gaps_html}
</div>
"""

        # Cross-domain connections
        if connections:
            html += '<div class="section-title">Cross-Domain Connections</div>\n'
            for conn in connections[:10]:
                pa = conn["paper_a"]
                pb = conn["paper_b"]
                methods = ", ".join(conn["shared_methods"])
                html += f"""
<div class="connection">
    <strong>{pa['domain']}</strong>: {pa['title'][:80]}...<br>
    <strong>{pb['domain']}</strong>: {pb['title'][:80]}...<br>
    <em>Shared methods: {methods}</em>
</div>
"""

        html += f"""
<div class="footer">
    Research Telescope v1.0 &mdash; Automated scan at {datetime.now().strftime('%H:%M UTC')}<br>
    Scanning {len(ARXIV_CATEGORIES)} ArXiv categories + bioRxiv + Semantic Scholar + CrossRef
</div>
</body></html>
"""

        return html

    def send(self, html: str, subject: str = None, dry_run: bool = False) -> bool:
        """Send the digest email."""
        if not subject:
            tod = time_of_day()
            date_str = datetime.now().strftime("%Y-%m-%d")
            subject = f"Research Telescope Digest -- {date_str} {tod}"

        if dry_run:
            # Save to file instead
            path = DIGESTS_DIR / f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            log.info(f"[DRY RUN] Digest saved to {path}")
            return True

        if not SMTP_USER or not SMTP_PASS:
            log.warning(
                "[Email] SMTP credentials not configured. "
                "Set SMTP_USER and SMTP_PASS env vars. "
                "Saving digest to file instead."
            )
            path = DIGESTS_DIR / f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            log.info(f"Digest saved to {path} (configure SMTP to enable email)")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
            log.info(f"[Email] Digest sent to {RECIPIENT_EMAIL}")
            return True
        except Exception as e:
            log.error(f"[Email] Failed to send: {e}")
            # Fallback: save to file
            path = DIGESTS_DIR / f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            log.info(f"Digest saved to {path}")
            return False


# ---------------------------------------------------------------------------
# Main Scanner Orchestrator
# ---------------------------------------------------------------------------

class TelescopeScanner:
    """Main orchestrator that ties all components together."""

    def __init__(self, sources: list[str] = None, max_papers: int = 200):
        self.sources = sources or ["arxiv", "biorxiv", "semantic_scholar", "crossref"]
        self.max_papers = max_papers
        self.store = ResultsStore()
        self.analyzer = AnalysisEngine()
        self.connector = ConnectionDetector()
        self.emailer = DigestEmailer()

    def scan(self) -> dict:
        """Run a full scan across all configured sources."""
        log.info("=" * 60)
        log.info("RESEARCH TELESCOPE SCAN STARTING")
        log.info(f"Sources: {', '.join(self.sources)}")
        log.info(f"Max papers: {self.max_papers}")
        log.info("=" * 60)

        all_papers = []
        per_source = self.max_papers // len(self.sources)

        # Fetch from each source
        if "arxiv" in self.sources:
            fetcher = ArXivFetcher()
            all_papers.extend(fetcher.fetch(
                ARXIV_CATEGORIES, SEARCH_KEYWORDS, max_results=per_source
            ))

        if "biorxiv" in self.sources:
            fetcher = BioRxivFetcher()
            all_papers.extend(fetcher.fetch(max_results=per_source))

        if "semantic_scholar" in self.sources:
            fetcher = SemanticScholarFetcher()
            all_papers.extend(fetcher.fetch(SEARCH_KEYWORDS, max_results=per_source))

        if "crossref" in self.sources:
            fetcher = CrossRefFetcher()
            all_papers.extend(fetcher.fetch(SEARCH_KEYWORDS, max_results=per_source))

        total_fetched = len(all_papers)
        log.info(f"Total fetched: {total_fetched}")

        # Deduplicate
        seen = set()
        unique = []
        for p in all_papers:
            if p["id"] not in seen:
                seen.add(p["id"])
                unique.append(p)
        all_papers = unique
        log.info(f"After dedup: {len(all_papers)}")

        # Analyze each paper
        log.info("Analyzing papers...")
        analyzed = []
        for p in all_papers:
            analyzed.append(self.analyzer.analyze_paper(p))

        # Sort by overall score
        analyzed.sort(key=lambda x: x.get("scores", {}).get("overall", 0), reverse=True)

        # Cross-domain connections
        log.info("Detecting cross-domain connections...")
        connections = self.connector.find_connections(analyzed)

        # Citation traversal for top papers
        log.info("Traversing citations for top papers...")
        traverser = CitationTraverser()
        cited_papers = traverser.traverse(analyzed, score_threshold=6.0, max_papers=3)
        for cp in cited_papers:
            cp_analyzed = self.analyzer.analyze_paper(cp)
            if cp_analyzed["id"] not in seen:
                analyzed.append(cp_analyzed)
                seen.add(cp_analyzed["id"])

        # Re-sort after adding citation-discovered papers
        analyzed.sort(key=lambda x: x.get("scores", {}).get("overall", 0), reverse=True)

        # Build results
        results = {
            "metadata": {
                "scan_time": datetime.utcnow().isoformat(),
                "sources": self.sources,
                "total_fetched": total_fetched,
                "after_dedup": len(analyzed),
                "categories_scanned": ARXIV_CATEGORIES,
                "keyword_count": len(SEARCH_KEYWORDS),
            },
            "papers": analyzed,
            "connections": connections,
            "summary": {
                "top_5_titles": [p["title"] for p in analyzed[:5]],
                "avg_overall_score": round(
                    sum(p["scores"]["overall"] for p in analyzed) / max(len(analyzed), 1), 2
                ),
                "high_scoring_count": sum(
                    1 for p in analyzed if p["scores"]["overall"] >= 7
                ),
                "gaps_found_total": sum(
                    len(p.get("gaps_detected", [])) for p in analyzed
                ),
                "cross_domain_connections": len(connections),
            },
        }

        # Save
        save_path = self.store.save(results)

        # Print summary
        self._print_summary(results)

        return results

    def send_digest(self, results: dict = None, dry_run: bool = False) -> bool:
        """Format and send email digest."""
        if results is None:
            results = self.store.load_latest()
            if results is None:
                log.error("No scan results found. Run --scan first.")
                return False

        html = self.emailer.format_digest(results)
        return self.emailer.send(html, dry_run=dry_run)

    def _print_summary(self, results: dict):
        """Print a CLI summary of results."""
        s = results["summary"]
        print("\n" + "=" * 60)
        print("  RESEARCH TELESCOPE SCAN COMPLETE")
        print("=" * 60)
        print(f"  Papers analyzed:        {results['metadata']['after_dedup']}")
        print(f"  Average score:          {s['avg_overall_score']}/10")
        print(f"  High-scoring (>=7):     {s['high_scoring_count']}")
        print(f"  Gaps detected:          {s['gaps_found_total']}")
        print(f"  Cross-domain links:     {s['cross_domain_connections']}")
        print()
        print("  TOP 5 OPPORTUNITIES:")
        for i, title in enumerate(s["top_5_titles"], 1):
            score = results["papers"][i - 1]["scores"]["overall"]
            print(f"    {i}. [{score}/10] {title[:70]}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Research Telescope Paper Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python telescope_scanner.py --scan                   Scan all sources
  python telescope_scanner.py --scan --email            Scan and email digest
  python telescope_scanner.py --scan --dry-run          Scan, save digest as HTML file
  python telescope_scanner.py --email-only              Email the latest scan results
  python telescope_scanner.py --scan --sources arxiv    Scan ArXiv only
  python telescope_scanner.py --scan --max-papers 50    Limit to 50 papers
        """,
    )
    parser.add_argument("--scan", action="store_true", help="Run a full paper scan")
    parser.add_argument("--email", action="store_true", help="Send email digest after scan")
    parser.add_argument("--email-only", action="store_true", help="Email latest results without scanning")
    parser.add_argument("--dry-run", action="store_true", help="Save digest as HTML file instead of emailing")
    parser.add_argument(
        "--sources", nargs="+",
        choices=["arxiv", "biorxiv", "semantic_scholar", "crossref"],
        default=None,
        help="Which sources to scan (default: all)",
    )
    parser.add_argument("--max-papers", type=int, default=200, help="Max papers to fetch (default: 200)")

    args = parser.parse_args()

    if not args.scan and not args.email_only:
        parser.print_help()
        sys.exit(0)

    scanner = TelescopeScanner(sources=args.sources, max_papers=args.max_papers)

    if args.scan:
        results = scanner.scan()
        if args.email or args.dry_run:
            scanner.send_digest(results, dry_run=args.dry_run)
    elif args.email_only:
        scanner.send_digest(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
