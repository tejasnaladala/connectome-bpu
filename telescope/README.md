# Research Telescope Paper Scanner

Automated system that scans research papers across ALL scientific domains to find
revolutionary research opportunities, gaps, and cross-domain connections.

## Quick Start

```bash
# From the connectome_bpu directory:

# Full scan, save digest as HTML file (no email):
python scripts/telescope_scanner.py --scan --dry-run

# Full scan + email digest:
python scripts/telescope_scanner.py --scan --email

# Scan ArXiv only, limit to 50 papers:
python scripts/telescope_scanner.py --scan --sources arxiv --max-papers 50

# Email the latest scan results (no new scan):
python scripts/telescope_scanner.py --email-only
```

## Dependencies

Only `requests` is required beyond the standard library:

```bash
pip install requests
```

## Email Configuration

To enable email delivery, set these environment variables:

```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASS="your-app-password"
export SENDER_EMAIL="your-email@gmail.com"
```

For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833)
(not your regular password). Go to Google Account > Security > 2-Step Verification >
App Passwords.

Without SMTP credentials, digests are saved as HTML files in `telescope/digests/`.

## Sources

| Source | API | What it covers |
|--------|-----|---------------|
| ArXiv | export.arxiv.org | 12 categories: cs.NE, cs.LG, cs.AI, cs.CV, q-bio.NC, q-bio.QM, physics.comp-ph, cond-mat, astro-ph, stat.ML, math.OC, eess.SP |
| bioRxiv | api.biorxiv.org | Neuroscience, computational biology, bioinformatics |
| Semantic Scholar | api.semanticscholar.org | Keyword search with citation traversal |
| CrossRef | api.crossref.org | Broad academic search, DOI metadata |

## Scoring System

Each paper is rated 1-10 on four dimensions:

- **Capitalize**: How likely can this be turned into a published paper by us?
- **Novelty**: How novel is the gap/opportunity?
- **Feasibility**: Can we do this with public datasets and our compute?
- **Connection**: How well does this connect to our existing knowledge base?

The **Overall** score is the average of all four.

## Output

- **JSON results**: `telescope/results/scan_YYYYMMDD_HHMMSS.json`
- **HTML digests**: `telescope/digests/digest_YYYYMMDD_HHMMSS.html`
- **Email**: Sent to naladala@uw.edu (when SMTP is configured)

## Directory Structure

```
telescope/
  cache/       # API response cache (future use)
  digests/     # Saved HTML email digests
  results/     # JSON scan results (accumulated over time)
  README.md    # This file
```

## Scheduling (3 scans/day)

Use cron (Linux/Mac) or Task Scheduler (Windows) to run every 8 hours:

### Linux/Mac cron:
```
0 6,14,22 * * * cd /path/to/connectome_bpu && python scripts/telescope_scanner.py --scan --email
```

### Windows Task Scheduler:
Create a task that runs at 6:00, 14:00, and 22:00 daily with:
```
python C:\Users\tejas\OneDrive\Desktop\connectome_bpu\scripts\telescope_scanner.py --scan --email
```

## How It Works

1. **Fetch** papers from all configured sources (ArXiv, bioRxiv, Semantic Scholar, CrossRef)
2. **Deduplicate** by title+author hash
3. **Analyze** each paper: detect gaps (regex on "future work"/"limitations"/etc.), score relevance
4. **Cross-reference** papers across domains to find technique overlaps
5. **Traverse citations** for the highest-scoring papers to discover related work
6. **Save** results as JSON, render HTML digest, optionally send email
