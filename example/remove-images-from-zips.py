#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "python-docx>=1.2.0",
# ]
# ///
"""Clean images out of docx/zip archives and compress PDFs, preserving quality.

Archive handling (no external binaries needed):
  * Pure Python stdlib (`zipfile`, `xml.etree`) - no reliance on unzip/zip.
  * Archives are rebuilt in memory then written atomically (temp file +
    os.replace), so a crash never leaves a half-written file.
  * For .docx it also strips the XML elements that *reference* images
    (<w:drawing>, <w:pict>, <w:object>, pic:pic, ...) and prunes the matching
    relationship entries out of every .rels file.  Deleting media blobs without
    this step leaves dangling relationship IDs, which Word flags as corrupt.
  * docx files are reduced to the main body text: document metadata
    (docProps/*), tracked-change/versioning markup (w:ins/w:del, rsid*
    attributes, trackChanges.xml, people.xml, comments), headers, footers,
    footnotes, endnotes, and embedded objects/media are all dropped.  Inserted
    text is accepted and deleted text removed, so the final accepted text stays.
  * Nested archives (docx/zips inside zips, zips inside docx) are cleaned
    recursively with a depth limit, preserving the outer archive's structure.
  * Untouched entries pass through byte-identical; archives with nothing to
    remove are left exactly as they were.

PDF compression (uses Ghostscript if installed, otherwise skipped):
  * Re-encodes with pdfwrite at a configurable quality preset. The default
    'printer' preset keeps 300 dpi (600 dpi mono) and high JPEG quality, so
    quality loss is not discernible while duplicate/downsampled image data is
    removed.
  * Only replaces the original when the result is actually smaller.
  * PDFs found inside archives are compressed the same way, recursively.

Usage:
    python3 remove-images-from-zips.py [--dir DIR] [--dry-run]
        [--pdf-quality {screen,ebook,printer,prepress}] [--no-pdfs]
"""

import argparse
import io
import os
import posixpath
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET

IMAGE_EXTS = {
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif",
    "svg", "ico", "raw", "cr2", "nef", "arw", "heic", "heif",
}

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PICTURE_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
VML_NS = "urn:schemas-microsoft-com:vml"
OFFICE_NS = "urn:schemas-microsoft-com:office:office"

# Containers that embed images in OOXML - remove them wholesale.
STRIP_TAGS = {
    f"{{{WORD_NS}}}drawing",
    f"{{{WORD_NS}}}pict",
    f"{{{WORD_NS}}}object",
    f"{{{PICTURE_NS}}}pic",
    f"{{{DRAWING_NS}}}blip",
    f"{{{VML_NS}}}imagedata",
}

# Non-body parts of a docx that add bulk/history without contributing the
# main text: metadata, revision/comment tracking, headers/footers, footnotes,
# and embedded media/objects.
DOCX_DROP_DIRS = {"media", "charts", "embeddings", "activex"}
DOCX_DROP_STEMS = {
    "footnotes.xml", "endnotes.xml", "people.xml", "trackchanges.xml",
    "custom.xml",
}

# Tracked-change/comment markers inside word/document.xml.  Insertions are
# kept (their text is accepted) while deletions and annotation plumbing are
# removed, so only the final accepted text remains.
REVISION_UNWRAP_TAGS = {
    f"{{{WORD_NS}}}ins",
    f"{{{WORD_NS}}}moveTo",
}
REVISION_DROP_TAGS = {
    f"{{{WORD_NS}}}del",
    f"{{{WORD_NS}}}moveFrom",
    f"{{{WORD_NS}}}moveToRangeStart",
    f"{{{WORD_NS}}}moveToRangeEnd",
    f"{{{WORD_NS}}}moveFromRangeStart",
    f"{{{WORD_NS}}}moveFromRangeEnd",
    f"{{{WORD_NS}}}commentRangeStart",
    f"{{{WORD_NS}}}commentRangeEnd",
    f"{{{WORD_NS}}}commentReference",
    f"{{{WORD_NS}}}footnoteReference",
    f"{{{WORD_NS}}}endnoteReference",
    f"{{{WORD_NS}}}headerReference",
    f"{{{WORD_NS}}}footerReference",
    f"{{{WORD_NS}}}rPrChange",
    f"{{{WORD_NS}}}pPrChange",
    f"{{{WORD_NS}}}sectPrChange",
    f"{{{WORD_NS}}}trPrChange",
    f"{{{WORD_NS}}}tcPrChange",
    f"{{{WORD_NS}}}cellIns",
    f"{{{WORD_NS}}}cellDel",
    f"{{{WORD_NS}}}cellMerge",
}

# History/settings in word/settings.xml that don't affect the text.
SETTINGS_STRIP_TAGS = {
    f"{{{WORD_NS}}}trackChanges",
    f"{{{WORD_NS}}}rsids",
}

_NAMESPACES = {
    "w": WORD_NS,
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": DRAWING_NS,
    "pic": PICTURE_NS,
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "v": VML_NS,
    "o": OFFICE_NS,
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
for _prefix, _uri in _NAMESPACES.items():
    ET.register_namespace(_prefix, _uri)

MAX_DEPTH = 8

# dpi caps per quality preset. Higher = bigger files but less visible loss.
PDF_QUALITY = {
    "screen":   {"color": 72,  "gray": 72,  "mono": 144, "jpegq": 60, "downsample": True},
    "ebook":    {"color": 150, "gray": 150, "mono": 300, "jpegq": 75, "downsample": True},
    "printer":  {"color": 300, "gray": 300, "mono": 600, "jpegq": 90, "downsample": True},
    "prepress": {"color": 600, "gray": 600, "mono": 1200, "jpegq": 95, "downsample": False},
}


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def is_image(name: str) -> bool:
    ext = os.path.splitext(name)[1].lstrip(".").lower()
    return ext in IMAGE_EXTS


def should_drop_docx_part(name: str) -> bool:
    """True if the part is docx bulk that isn't needed for the main text."""
    lower = name.lower()
    if lower.startswith("docprops/"):
        return True
    if not lower.startswith("word/"):
        return False
    rel = lower[len("word/"):]
    if rel.split("/", 1)[0] in DOCX_DROP_DIRS:
        return True
    stem = rel.rsplit("/", 1)[-1]
    return (stem.startswith("header") or stem.startswith("footer")
            or stem.startswith("comments") or stem in DOCX_DROP_STEMS)


def _remove_all(el, tags):
    removed = 0
    for child in list(el):
        if child.tag in tags:
            el.remove(child)
            removed += 1
        else:
            removed += _remove_all(child, tags)
    return removed


def strip_xml_references(raw: bytes):
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw, False
    if _remove_all(root, STRIP_TAGS) == 0:
        return raw, False
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), True


def rels_base_dir(name: str) -> str:
    parts = name.split("/")
    try:
        i = parts.index("_rels")
    except ValueError:
        return posixpath.dirname(name)
    return "/".join(parts[:i])


def strip_rels(raw: bytes, name: str, drop_targets: set):
    base = rels_base_dir(name)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw, False
    removed = 0
    for rel in list(root):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target") or ""
        resolved = posixpath.normpath(posixpath.join(base, target.lstrip("/")))
        if is_image(resolved) or resolved in drop_targets:
            root.remove(rel)
            removed += 1
    if removed == 0:
        return raw, False
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), True


def strip_content_types(raw: bytes, drop_targets: set):
    """Prune [Content_Types].xml Override entries for removed parts."""
    if not drop_targets:
        return raw, False
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw, False
    removed = 0
    for child in list(root):
        if not child.tag.endswith("}Override"):
            continue
        part = (child.get("PartName") or "").lstrip("/")
        if posixpath.normpath(part) in drop_targets:
            root.remove(child)
            removed += 1
    if removed == 0:
        return raw, False
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), True


def _strip_revision_elems(el):
    removed = 0
    for child in list(el):
        if child.tag in REVISION_UNWRAP_TAGS:
            kids = list(child)
            idx = list(el).index(child)
            el.remove(child)
            for k in reversed(kids):
                el.insert(idx, k)
            removed += 1
            for k in kids:
                removed += _strip_revision_elems(k)
        elif child.tag in REVISION_DROP_TAGS:
            el.remove(child)
            removed += 1
        else:
            removed += _strip_revision_elems(child)
    return removed


def strip_revisions(raw: bytes):
    """Accept insertions, drop deletions/comments and rsid history."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw, False
    changed = False
    for el in root.iter():
        for attr in list(el.attrib):
            if attr.startswith(f"{{{WORD_NS}}}rsid"):
                del el.attrib[attr]
                changed = True
    if _strip_revision_elems(root):
        changed = True
    if not changed:
        return raw, False
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), True


def strip_settings_xml(raw: bytes):
    """Remove track-changes/history settings from word/settings.xml."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw, False
    if _remove_all(root, SETTINGS_STRIP_TAGS) == 0:
        return raw, False
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), True


def clean_archive_bytes(data: bytes, depth: int, gs_path: str = None,
                        pdf_quality: str = "printer"):
    """Return (new_bytes, changed). Only rebuilds if something changed."""
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return data, False

    changed = False
    docx_archive = False
    drop_targets = set()
    entries = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zin:
        comment = zin.comment
        infos = list(zin.infolist())
        if any(i.filename.lower() == "word/document.xml" for i in infos):
            docx_archive = True
            for i in infos:
                if should_drop_docx_part(i.filename):
                    drop_targets.add(posixpath.normpath(i.filename))
                    changed = True
        for info in infos:
            name = info.filename
            lower = name.lower()
            norm = posixpath.normpath(name)
            raw = zin.read(info)

            if norm in drop_targets:
                continue

            # Nested archive: recurse, keep the cleaned bytes.
            if depth < MAX_DEPTH and (lower.endswith(".docx") or lower.endswith(".zip")):
                cleaned, sub = clean_archive_bytes(raw, depth + 1, gs_path,
                                                   pdf_quality)
                if sub:
                    raw, changed = cleaned, True
            # Plain image file: drop it.
            elif is_image(name):
                changed = True
                continue
            # PDF: compress in place.
            elif lower.endswith(".pdf"):
                cleaned, sub = compress_pdf_bytes(raw, pdf_quality, gs_path)
                if sub:
                    raw, changed = cleaned, True
            # Package content types: prune entries for removed parts.
            elif lower == "[content_types].xml":
                cleaned, sub = strip_content_types(raw, drop_targets)
                if sub:
                    raw, changed = cleaned, True
            # XML part: strip drawing/image elements, plus (in docx) the
            # tracked-change/comment markup so only the final text remains.
            elif lower.endswith(".xml"):
                cleaned, sub = strip_xml_references(raw)
                if sub:
                    raw, changed = cleaned, True
                if docx_archive:
                    if lower == "word/document.xml":
                        cleaned, sub = strip_revisions(raw)
                    elif lower == "word/settings.xml":
                        cleaned, sub = strip_settings_xml(raw)
                    else:
                        cleaned, sub = None, False
                    if sub:
                        raw, changed = cleaned, True
            # Relationship part: drop image + removed-part relationships.
            elif lower.endswith(".rels"):
                cleaned, sub = strip_rels(raw, name, drop_targets)
                if sub:
                    raw, changed = cleaned, True

            entries.append((info, raw))

    if not changed:
        return data, False

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.comment = comment
        for info, raw in entries:
            zi = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.comment = info.comment
            zi.external_attr = info.external_attr
            zi.create_system = info.create_system
            zout.writestr(zi, raw)
    return buf.getvalue(), True


def process_archive(path: str, dry_run: bool, processed: set, gs_path: str,
                    pdf_quality: str) -> bool:
    key = os.path.realpath(path)
    if key in processed:
        return False
    processed.add(key)

    with open(path, "rb") as f:
        data = f.read()

    if not zipfile.is_zipfile(io.BytesIO(data)):
        print(f"  skip (not a zip): {path}")
        return False

    new_data, changed = clean_archive_bytes(data, 0, gs_path, pdf_quality)
    if not changed:
        return False

    print(f"  cleaned: {path}")
    if not dry_run:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(new_data)
        os.replace(tmp, path)
    return True


def gs_command(inp: str, outp: str, profile: str) -> list:
    p = PDF_QUALITY[profile]
    cmd = [
        "gs", "-q", "-dNOPAUSE", "-dBATCH", "-dQUIET",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        "-dEmbedAllFonts=true",
        f"-dColorImageResolution={p['color']}",
        f"-dGrayImageResolution={p['gray']}",
        f"-dMonoImageResolution={p['mono']}",
        f"-dJPEGQ={p['jpegq']}",
    ]
    if p["downsample"]:
        cmd += [
            "-dDownsampleColorImages=true",
            "-dDownsampleGrayImages=true",
            "-dDownsampleMonoImages=true",
            "-dColorImageDownsampleType=/Bicubic",
            "-dGrayImageDownsampleType=/Bicubic",
        ]
    else:
        cmd += [
            "-dDownsampleColorImages=false",
            "-dDownsampleGrayImages=false",
            "-dDownsampleMonoImages=false",
        ]
    cmd += ["-dAutoFilterColorImages=true", "-dAutoFilterGrayImages=true",
            f"-sOutputFile={outp}", inp]
    return cmd


def compress_pdf(path: str, dry_run: bool, profile: str, gs_path: str) -> int:
    """Return bytes saved (0 if kept/skipped)."""
    if not gs_path:
        print(f"  skip (Ghostscript not installed): {path}")
        return 0

    old = os.path.getsize(path)
    outp = path + ".tmp.pdf"
    try:
        subprocess.run(gs_command(path, outp, profile), check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  skip (Ghostscript failed on): {path}")
        if os.path.exists(outp):
            os.remove(outp)
        return 0

    new = os.path.getsize(outp)
    if new >= old:
        os.remove(outp)
        print(f"  kept original (not smaller): {path} ({fmt_size(old)})")
        return 0

    pct = 100.0 * (old - new) / old
    print(f"  compressed: {path} {fmt_size(old)} -> {fmt_size(new)} "
          f"(-{pct:.0f}%)")
    if not dry_run:
        os.replace(outp, path)
    else:
        os.remove(outp)
    return old - new


def compress_pdf_bytes(data: bytes, profile: str, gs_path: str):
    """Compress an in-memory PDF. Returns (new_bytes, changed).

    Writes to a temp file for Ghostscript, only keeps the result if smaller.
    """
    if not gs_path:
        return data, False

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        in_path = f.name
        f.write(data)
    out_path = in_path + ".tmp"
    new = None
    try:
        subprocess.run(gs_command(in_path, out_path, profile), check=True,
                       capture_output=True)
        with open(out_path, "rb") as f:
            new = f.read()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    finally:
        os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
    if new is None or len(new) >= len(data):
        return data, False
    return new, True


def main():
    parser = argparse.ArgumentParser(
        description="Strip images and history from docx/zip archives and "
                    "compress PDFs.")
    parser.add_argument("--dir", default=".", help="directory to scan (default: current)")
    parser.add_argument("--dry-run", action="store_true", help="report only, don't rewrite")
    parser.add_argument("--pdf-quality", choices=sorted(PDF_QUALITY),
                        default="printer",
                        help="PDF quality/downsample preset (default: printer)")
    parser.add_argument("--no-pdfs", action="store_true",
                        help="skip PDF compression")
    args = parser.parse_args()

    archives, pdfs = [], []
    for dirpath, _dirs, files in os.walk(args.dir):
        for f in files:
            lower = f.lower()
            if lower.endswith((".docx", ".zip")):
                archives.append(os.path.join(dirpath, f))
            elif lower.endswith(".pdf") and not lower.endswith(".tmp.pdf"):
                pdfs.append(os.path.join(dirpath, f))

    processed = set()
    cleaned = 0
    gs_path = None if args.no_pdfs else shutil.which("gs")
    for path in sorted(archives):
        if process_archive(path, args.dry_run, processed, gs_path,
                           args.pdf_quality):
            cleaned += 1

    saved = 0
    compressed = 0
    for path in sorted(pdfs):
        delta = compress_pdf(path, args.dry_run, args.pdf_quality, gs_path)
        if delta:
            saved += delta
            compressed += 1

    verb = "Would clean" if args.dry_run else "Cleaned"
    print(f"{verb} {cleaned} archive file(s).")
    if not args.no_pdfs:
        verb = "Would compress" if args.dry_run else "Compressed"
        print(f"{verb} {compressed} PDF(s), saving {fmt_size(saved)} "
              f"({args.pdf_quality} quality).")


if __name__ == "__main__":
    main()