# OCR Document Intelligence Platform

An asynchronous document intelligence system that extracts structured data
from invoices and receipts (PDF or image). Every page is run through
**three independent OCR engines** — Tesseract, EasyOCR, and PaddleOCR —
whose outputs are reconciled through a weighted consensus algorithm, checked
against business rules, and automatically routed to either straight-through
approval or human review based on confidence.

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow](#workflow)
3. [Technical Architecture](#technical-architecture)
4. [Core Features](#core-features)
5. [Multi-Engine OCR & Extraction](#multi-engine-ocr--extraction)
6. [Validation, Review & Database](#validation-review--database)
7. [Tech Stack by Layer](#tech-stack-by-layer)
8. [Project Directory Structure](#project-directory-structure)
9. [Core API Endpoints](#core-api-endpoints)
10. [Frontend Views](#frontend-views)
11. [Running Locally](#running-locally)

---

## Overview

No single OCR engine is reliably accurate across every kind of document.
Tesseract, EasyOCR, and PaddleOCR each have different strengths — typed
text, varied fonts, multilingual receipts, layout diversity. Instead of
trusting one engine's output blindly, this system runs all three in
parallel on every page and lets them vote on the correct value for each
field. Fields all three engines agree on get high confidence; fields where
they disagree get flagged for a human reviewer instead of silently guessing.

The result is a pipeline that is both **automated** (clean documents flow
straight through with no human involved) and **safe** (ambiguous or
inconsistent documents are caught and routed to a person before anything
downstream trusts the extracted numbers).

---

## Workflow

```
Upload  ─▶  Quality Analysis  ─▶  Preprocessing  ─▶  Multi-Engine OCR  ─▶  Field Extraction  ─▶  Consensus  ─▶  Validation  ─▶  Routing  ─▶  Human Review (if needed)
```

1. **Upload** — 1 to 10 files accepted at once (PDF, PNG, or JPG).
   Multi-page PDFs are split into individual pages automatically.
2. **Quality Analysis** — each page is scored for blur, brightness,
   contrast, skew angle, estimated DPI, and color mode.
3. **Preprocessing** — based on the quality score, the page is
   automatically deskewed, contrast-enhanced, sharpened, or denoised
   before OCR ever sees it.
4. **Multi-Engine OCR** — the preprocessed page is run through all three
   engines independently, each returning raw text and a confidence score.
5. **Field Extraction** — each engine's raw text is parsed (regex +
   spatial heuristics) into structured fields: invoice number, date, tax
   ID, subtotal, tax amount, grand total, payment method, line items.
6. **Consensus** — for each field, the three engines' values are compared
   and weighted-voted on; the winning value, its confidence, and which
   engines agreed on it are all recorded.
7. **Validation** — business rules check for missing required fields and
   verify `subtotal + tax == grand_total` within a small tolerance.
8. **Routing** — combining field confidence and validation results, the
   document is marked `AUTO_APPROVED` or `HUMAN_REVIEW_REQUIRED`.
9. **Human Review** (if required) — a reviewer inspects flagged fields in
   the UI, corrects values if needed, and approves or rejects the document.

The frontend mirrors this as a 3-step wizard: **Upload → Preprocessing &
Quality → Results**, plus a **Batch Queue** view for multi-file uploads and
a **History** view for revisiting past documents.

---

## Technical Architecture

The system is organized into four layers.

**Frontend layer.** A single HTML file (`index.html`) containing the upload
wizard, batch queue, history view, and review UI. It communicates with the
backend entirely over REST — JSON for status/results, multipart form data
for uploads.

**API layer.** `main.py`, built with FastAPI, exposes all endpoints:
upload, status, result, review, export, and document history. It accepts
requests, persists an initial document record, and hands off processing to
a background task rather than blocking the response on OCR — since OCR can
take several seconds per page, this keeps the API responsive immediately
after upload (`202 Accepted`).

**Orchestration and service layer.** `services/jobs.py` is the
orchestrator: for each document, it runs quality analysis, applies
preprocessing, invokes the OCR consensus engine, extracts fields, computes
consensus, validates business rules, and determines routing — in that
order, per page. Each of these responsibilities lives in its own service
module (quality analyzer, preprocessing pipeline, OCR consensus engine,
field extraction parser, validation rules, confidence router), so the
orchestrator coordinates them rather than implementing the logic itself.

**Data layer.** SQLAlchemy over SQLite persists one row per document,
storing per-page and consolidated results as JSON columns so the schema
stays flexible across different document shapes and page counts.

Processing runs as a FastAPI `BackgroundTask` so the upload endpoint
responds immediately rather than blocking on OCR.

---

## Core Features

- **Multi-engine OCR consensus** — Tesseract, EasyOCR, and PaddleOCR run on
  every page; a weighted vote resolves each field.
- **Image quality analysis** — blur, brightness, contrast, skew, DPI, and
  color mode scored per page before OCR.
- **Adaptive preprocessing** — deskewing, CLAHE contrast enhancement,
  sharpening, and denoising applied automatically based on quality score.
- **Multi-page PDF support** — each page processed and viewable
  independently in the UI, with its own quality metrics, OCR results, and
  extracted fields.
- **Confidence-based routing** — documents are auto-approved or flagged for
  human review based on field-level agreement and validation outcomes.
- **Business rule validation** — required-field checks and arithmetic
  consistency checks (subtotal + tax = total).
- **Human-in-the-loop review** — reviewers can view, correct, approve, or
  reject flagged documents directly in the UI.
- **Batch upload** — up to 10 documents processed concurrently with a live
  status queue.
- **Document history** — every processed document remains browsable and
  re-viewable.
- **Export** — results downloadable as JSON or CSV.

---

## Multi-Engine OCR & Extraction

### Engines

| Engine     | Library      | Default weight |
|------------|--------------|-----------------|
| PaddleOCR  | `paddleocr`  | 0.40            |
| Tesseract  | `pytesseract`| 0.35            |
| EasyOCR    | `easyocr`    | 0.25            |

Weights are configurable in `OCRConsensusEngine.__init__` and reflect each
engine's relative reliability observed during development. Each engine also
reports its own per-page confidence score (an average of per-token/per-line
confidences where the underlying library exposes one).

### Field extraction

Each engine's raw text is parsed independently (`services/extraction/parser.py`)
using regex patterns and, where token-level bounding boxes are available,
spatial proximity heuristics (`services/extraction/layout.py`) to find
values near their labels. Extracted fields:

- Invoice number, invoice date, tax ID (GSTIN pattern supported)
- Currency, subtotal, tax amount, grand total
- Payment method
- Line items (description, quantity, unit price, line total)

### Consensus algorithm

For each field, every engine's extracted value is treated as a weighted
vote. The value with the highest combined engine weight wins; its
confidence is the winning weight divided by the total possible weight
across engines that returned a value. Line items are resolved similarly,
factoring in both engine trust weight and the completeness of the item list.

---

## Validation, Review & Database

### Business rule validation (`services/validation/rules.py`)

| Rule | Severity |
|---|---|
| Invoice number missing | HIGH |
| Invoice date missing | MEDIUM |
| `subtotal + tax_amount ≠ grand_total` (tolerance ±0.05) | HIGH |

### Confidence-based routing (`services/validation/confidence_router.py`)

Fields below the confidence threshold (default `0.85`) or documents with
any validation issues are routed to `HUMAN_REVIEW_REQUIRED`; otherwise
`AUTO_APPROVED`.

### Human review

`PUT /documents/{id}/review` accepts an approval decision plus optional
field- and line-item-level corrections, recording the reviewer's identity
and updating the document's status.

### Database

SQLite via SQLAlchemy (`app/core/database.py`), a single `documents` table
(`app/models/document.py`):

| Column | Type | Purpose |
|---|---|---|
| `id` | String (UUID) | Primary key |
| `filename`, `file_type`, `doc_category` | String | Upload metadata |
| `status`, `progress` | String, Float | Processing/review state |
| `total_pages` | Integer | Page count |
| `overall_confidence` | Float | Routing confidence |
| `routing_reason` | String | Why a document was routed a given way |
| `error_message` | Text | Populated if processing failed |
| `pages_data` | JSON | Per-page quality, OCR, and consensus results |
| `consensus_data` | JSON | Document-level consolidated field values |
| `validation_issues` | JSON | List of validation findings |
| `routing_decision` | JSON | Full routing engine output |

Storing per-page and consolidated results as JSON columns avoids a rigid
relational schema for data whose shape varies by document type and page
count.

---

## Tech Stack by Layer

| Layer | Technology | Role |
|---|---|---|
| Frontend | Vanilla JS, Tailwind CSS (CDN) | Upload wizard, batch queue, history, review UI — no build step |
| API | FastAPI | REST endpoints, request validation (Pydantic), background task dispatch |
| Orchestration | Python (`services/jobs.py`) | Coordinates the pipeline per document, tracks progress |
| OCR | Tesseract (`pytesseract`), EasyOCR, PaddleOCR | Text extraction |
| Image processing | OpenCV, Pillow | Quality analysis, preprocessing |
| PDF handling | PyMuPDF (`fitz`) | PDF → page image conversion, page preview rendering |
| Data | SQLAlchemy, SQLite | Persistence |
| Config | `pydantic-settings` | Environment-based configuration |

---

## Project Directory Structure

```
app/
├── main.py                        # FastAPI app & all API routes (entry point)
├── index.html                     # Frontend (upload wizard, batch, history)
├── core/
│   ├── config.py                  # Settings (DB URL, Poppler path)
│   └── database.py                # SQLAlchemy engine/session setup
├── models/
│   └── document.py                # Document ORM model
├── repositories/
│   └── document_repository.py     # DB read/write operations
├── services/
│   ├── jobs.py                    # Pipeline orchestration (the core of the system)
│   ├── export_service.py          # JSON/CSV export formatting
│   ├── ocr/
│   │   └── consensus.py           # Runs all 3 engines + consensus voting
│   ├── extraction/
│   │   ├── parser.py              # Regex/heuristic field extraction
│   │   └── layout.py              # Spatial proximity matching
│   ├── preprocessing/
│   │   ├── quality.py             # Image quality scoring
│   │   └── pipeline.py            # Deskew/contrast/sharpen/denoise
│   └── validation/
│       ├── rules.py                # Business rule validation
│       └── confidence_router.py    # Routing decision logic
└── schemas/
    └── document.py                # (early-iteration Pydantic schemas)
```

---

## Core API Endpoints

All routes are prefixed under `/documents` except the root.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/documents` | List all documents with status |
| `POST` | `/documents/upload` | Upload 1–10 files (`file1`...`file10`); auto-queues processing |
| `POST` | `/documents/{id}/process` | Manually (re)start processing for a document |
| `GET` | `/documents/{id}/status` | Current status, progress %, confidence, routing reason |
| `GET` | `/documents/{id}/result` | Full results: per-page quality/OCR/consensus, validation, routing |
| `GET` | `/documents/{id}/image?page=N` | Page preview image (renders PDF pages or normalizes image uploads to PNG) |
| `PUT` | `/documents/{id}/review` | Submit reviewer approval/rejection and field corrections |
| `GET` | `/documents/{id}/export?format=json\|csv` | Export final results |

---

## Frontend Views

| View | Purpose |
|---|---|
| **Upload** | Drag-and-drop or browse, up to 10 files |
| **Preprocessing & Quality** | Live progress, phase labels, per-page quality metrics, outcome banner |
| **Results** | Document preview, header fields, line items, engine confidence bars, validation issues, extracted text — all per-page for multi-page documents |
| **Batch Queue** | Shown automatically for multi-file uploads; each document tracked and opened independently |
| **History** | Every previously processed document, reopenable at any time |

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` in a browser.

### Requirements

- Tesseract OCR binary installed and on `PATH` (or configured via
  `pytesseract.pytesseract.tesseract_cmd`)
- Sufficient disk space for EasyOCR/PaddleOCR model downloads on first run