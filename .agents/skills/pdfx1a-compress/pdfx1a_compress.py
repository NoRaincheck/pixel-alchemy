#!/usr/bin/env python3
"""
pdfx1a_compress.py — make a print-ready, reproducible PDF/X-1a with minimal size growth.

Pipeline
--------
If the input already uses only CMYK/Gray colour, it is finalised directly.

If the input carries RGB/Lab colour or transparency (e.g. a Quartz export),
Ghostscript first converts everything to DeviceCMYK (fonts embedded,
images re-encoded as JPEG, transparency flattened), then the deterministic
finalizer takes over:

  * image pixels only change when you ask them to (`--quality`);
    otherwise just a lossless `jpegtran -optimize` pass, kept if smaller;
  * PDF/X-1a conformance is injected deterministically:
      - /TrimBox (= /MediaBox) on every page
      - OutputIntent (/GTS_PDFX) with a CMYK ICC profile (N=4)
      - Info dict: GTS_PDFXVersion, Trapped /False, pinned dates
      - XMP metadata consistent with the Info dict
      - deterministic document/instance IDs (derived from input bytes)
  * serialisation is canonical: same input bytes + same options + same toolchain
    => byte-identical output.

Lossy mode (--quality p90|p75|<int>):
  * images are re-encoded with Pillow at the requested quality; a candidate is
    kept only when it is actually smaller than the current best.
  * --subsampling none|422|420|auto controls chroma subsampling of the C channel
    (M, Y and K always stay full-resolution). 'auto' tries both and keeps the
    smaller. Subsampling trades colour sharpness for size.

Examples
--------
  ./pdfx1a_compress.py compress in.pdf out.pdf                     # lossless + PDF/X-1a
  ./pdfx1a_compress.py compress rgb.pdf out.pdf                    # auto RGB->CMYK via Ghostscript
  ./pdfx1a_compress.py compress in.pdf out.pdf --quality p90       # re-encode @ quality 90
  ./pdfx1a_compress.py compress in.pdf out.pdf --quality p75 --subsampling auto
  ./pdfx1a_compress.py verify out.pdf

Requires: python3 (stdlib only for the core pipeline).
Optional: ghostscript (auto RGB->CMYK conversion), jpegtran (lossless savings),
          Pillow for --quality, qpdf + poppler for verify extras.
"""

import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zlib
from datetime import datetime, timezone

TOOL_NAME = "pdfx1a-compress"
TOOL_VERSION = "1.1.0"

OBJ_START = re.compile(rb"(?m)^(\d+)\s+(\d+)\s+obj\b")

SUBSAMPLING_CHOICES = {"none": 0, "422": 1, "420": 2}


class Obj:
    __slots__ = ("num", "gen", "head", "stream")

    def __init__(self, num, gen, head, stream):
        self.num = num
        self.gen = gen
        self.head = head          # bytes between "obj" and "stream"/"endobj"
        self.stream = stream      # raw stream payload bytes or None

    def dict_bytes(self):
        return b" ".join(self.head.split())


# ----------------------------------------------------------------- parsing

def parse_pdf(data):
    """Parse a classic-xref PDF into {num: Obj}. No object streams, no encryption."""
    objs = {}
    n = len(data)
    for m in OBJ_START.finditer(data):
        num, gen = int(m.group(1)), int(m.group(2))
        pos = m.end()
        e = data.find(b"endobj", pos)
        if e == -1:
            raise ValueError("object %d: no endobj" % num)
        seg = data[pos:e]
        stm = re.compile(rb"\bstream(\r\n|\n|\r)").search(seg)
        if stm:
            head = seg[: stm.start()]
            lm = re.search(rb"/Length\s+(\d+)\s+\d+\s+R", head)
            if lm:
                ref = int(lm.group(1))
                m2 = re.search(rb"(?m)^%d\s+\d+\s+obj\s*(\d+)\s*endobj" % ref, data[:n])
                length = int(m2.group(1)) if m2 else None
            else:
                ln = re.search(rb"/Length\s+(\d+)", head)
                length = int(ln.group(1)) if ln else None
            if length is None:
                raise ValueError("object %d: cannot resolve /Length" % num)
            payload_start = e - (len(seg) - stm.end())
            tail = data[payload_start + length : e]
            if b"endstream" not in tail:
                raise ValueError("object %d: endstream mismatch" % num)
            objs[num] = Obj(num, gen, head.rstrip(), data[payload_start : payload_start + length])
        else:
            objs[num] = Obj(num, gen, seg.rstrip(), None)
    return objs


def get_trailer_refs(data):
    t = data.rfind(b"trailer")
    seg = data[t : data.rfind(b"startxref")]
    root = info = None
    m = re.search(rb"/Root\s+(\d+)\s+\d+\s+R", seg)
    if m:
        root = int(m.group(1))
    m = re.search(rb"/Info\s+(\d+)\s+\d+\s+R", seg)
    if m:
        info = int(m.group(1))
    return root, info


def dict_get_ref(head, key):
    m = re.search(re.escape(key) + rb"\s+(\d+)\s+\d+\s+R", head)
    return int(m.group(1)) if m else None


def find_pages(objs, root_num):
    """Walk the page tree; return page object numbers in document order."""
    catalog = objs[root_num]
    pages_ref = dict_get_ref(catalog.dict_bytes(), b"/Pages")
    out = []
    stack = [pages_ref]
    while stack:
        n = stack.pop(0)
        h = objs[n].dict_bytes()
        if re.search(rb"/Type\s*/Pages\b", h):
            kids = re.search(rb"/Kids\s*\[(.*?)\]", h, re.S).group(1)
            refs = re.findall(rb"(\d+)\s+\d+\s+R", kids)
            stack[0:0] = [int(r) for r in refs]
        elif re.search(rb"/Type\s*/Page\b", h):
            out.append(n)
    return out


def inherited_mediabox(objs, page_num):
    """Return MediaBox values as list of byte strings (walks up the tree)."""
    n = page_num
    seen = set()
    while n is not None and n not in seen:
        seen.add(n)
        mb = re.search(
            rb"/MediaBox\s*\[\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\]",
            objs[n].dict_bytes(),
        )
        if mb:
            return [g for g in mb.groups()]
        n = dict_get_ref(objs[n].dict_bytes(), b"/Parent")
    return None


# ----------------------------------------------------------------- writing

def serialize(objs, root_ref, info_ref, doc_hex, inst_hex):
    buf = io.BytesIO()
    buf.write(b"%PDF-1.3\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objs):
        o = objs[num]
        offsets[num] = buf.tell()
        buf.write(b"%d %d obj\n" % (num, o.gen))
        buf.write(o.head.strip() + b"\n")
        if o.stream is not None:
            buf.write(b"stream\n" + o.stream + b"\nendstream\n")
        buf.write(b"endobj\n")
    maxnum = max(objs)
    xref_pos = buf.tell()
    buf.write(b"xref\n0 %d\n" % (maxnum + 1))
    buf.write(b"0000000000 65535 f \n")
    for i in range(1, maxnum + 1):
        if i in offsets:
            buf.write(b"%010d 00000 n \n" % offsets[i])
        else:
            buf.write(b"0000000000 65535 f \n")
    trailer = b"trailer\n<< /Size %d /Root %d 0 R " % (maxnum + 1, root_ref)
    if info_ref:
        trailer += b"/Info %d 0 R " % info_ref
    trailer += b"/ID [<%s><%s>] >>\n" % (doc_hex.encode(), inst_hex.encode())
    buf.write(trailer)
    buf.write(b"startxref\n%d\n%%%%EOF\n" % xref_pos)
    return buf.getvalue()


# ----------------------------------------------------------------- helpers

def sha_hex(b):
    return hashlib.sha256(b).hexdigest()


def uuid_urn(hex32):
    h = hex32.lower()
    return "urn:uuid:%s-%s-%s-%s-%s" % (h[0:8], h[8:12], h[12:16], h[16:20], h[20:32])


def xml_escape(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def normalize_date(s):
    """Accept D:YYYYMMDDHHmmSS... or ISO; return canonical 'D:YYYYMMDDHHmmSSZ'."""
    s = s.strip().strip("()")
    if s.startswith("D:"):
        body = s[2:]
        m = re.match(r"(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", body)
        if m:
            parts = [p or "00" for p in m.groups()]
            return "D:" + "".join(parts) + "Z"
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("D:%Y%m%d%H%M%SZ")
        except ValueError:
            pass
    return None


def epoch_to_date(epoch):
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("D:%Y%m%d%H%M%SZ")


def resolve_date(cli_date, source_info_head):
    """Deterministic date resolution order: CLI > SOURCE_DATE_EPOCH > source file's
    own CreationDate > fixed epoch constant. Never the current wall clock."""
    if cli_date:
        d = normalize_date(cli_date)
        if not d:
            sys.exit("error: --date must look like D:20260823120000Z")
        return d
    env = os.environ.get("SOURCE_DATE_EPOCH")
    if env and env.strip().isdigit():
        return epoch_to_date(env.strip())
    if source_info_head:
        m = re.search(rb"/CreationDate\s*\(([^)]*)\)", source_info_head)
        if m:
            d = normalize_date(m.group(1).decode("latin1"))
            if d:
                return d
    return "D:19700101000000Z"


def optimize_jpeg_lossless(data, jpegtran):
    """jpegtran -optimize: lossless; returns smaller bytes or None."""
    if not jpegtran:
        return None
    try:
        p = subprocess.run(
            ["jpegtran", "-optimize", "-copy", "all"],
            input=data,
            capture_output=True,
            timeout=120,
        )
    except Exception:
        return None
    if p.returncode != 0 or len(p.stdout) == 0 or len(p.stdout) >= len(data):
        return None
    return p.stdout


def recompress_jpeg(data, quality, subsampling):
    """Lossy Pillow re-encode; returns bytes (may be larger — caller decides)."""
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    if im.mode != "CMYK":
        return None
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, subsampling=subsampling, optimize=True)
    return buf.getvalue() or None


def pick_image_candidate(original, jpegtran, args):
    """Return (best_bytes_or_None, method) where best beats `original` in size.

    method is one of: 'original', 'jpegtran', 'recompress'.
    """
    best = optimize_jpeg_lossless(original, jpegtran)
    method = "jpegtran" if best is not None else "original"

    if args.quality is not None or getattr(args, "preset", None) == "recompress":
        try:
            from PIL import Image  # noqa: F401  (availability check)
        except ImportError:
            sys.exit("error: --quality / --preset recompress require Pillow "
                     "(python3 -m pip install pillow)")
        q = args.quality if args.quality is not None else args.jpeg_quality
        subs = (
            [SUBSAMPLING_CHOICES[args.subsampling]]
            if args.subsampling != "auto"
            else [SUBSAMPLING_CHOICES[s] for s in ("none", "422", "420")]
        )
        for sub in subs:
            try:
                cand = recompress_jpeg(original, q, sub)
            except Exception:
                cand = None
            if cand and (best is None or len(cand) < len(best)):
                best = cand
                method = "recompress"

    if best is not None and len(best) < len(original):
        return best, method
    return None, "original"


# ----------------------------------------------------------------- conversion

def find_conversion_reasons(objs):
    """List reasons why a document cannot be finalised to PDF/X-1a as-is
    (RGB/Lab colour, transparency). Empty list == already colour-clean."""
    reasons = {}
    rgb_images = rgb_ops = transp = 0
    for o in objs.values():
        h = o.dict_bytes()
        if re.search(rb"/Subtype\s*/Image", h) and o.stream is not None:
            is_rgb = bool(re.search(rb"/DeviceRGB|/CalRGB|/Lab\b", h))
            cs_ref = re.search(rb"/ColorSpace\s*(\d+)\s+\d+\s+R", h)
            if not is_rgb and cs_ref:
                # /ColorSpace <ref> where object is [/ICCBased <stream>]
                arr = objs.get(int(cs_ref.group(1)))
                m2 = re.search(rb"/ICCBased\s+(\d+)\s+\d+\s+R", arr.head) if arr else None
                if m2:
                    icc_obj = objs.get(int(m2.group(1)))
                    if icc_obj and re.search(rb"/N\s+3\b", icc_obj.dict_bytes()):
                        is_rgb = True
            if not is_rgb:
                m3 = re.search(
                    rb"/ColorSpace\s*\[\s*/ICCBased\s+(\d+)\s+\d+\s+R", h)
                if m3:
                    icc_obj = objs.get(int(m3.group(1)))
                    if icc_obj and re.search(rb"/N\s+3\b", icc_obj.dict_bytes()):
                        is_rgb = True
            if is_rgb:
                rgb_images += 1
        if re.search(rb"/Type\s*/ExtGState", h):
            for key in (b"/CA", b"/ca"):
                vals = re.findall(key + rb"\s+([\d.]+)", h)
                if any(float(x) < 1.0 for x in vals):
                    transp += 1
            bm = re.search(rb"/BM\s*/(?!Normal)(\w+)", h)
            if bm:
                transp += 1
        if o.stream is None or b"FlateDecode" not in h:
            continue
        try:
            body = zlib.decompress(o.stream)
        except Exception:
            continue
        if b"Tj" in body or b"TJ" in body or b"Do " in body:
            for op in (b"rg", b"RG"):
                rgb_ops += len(re.findall(rb"(?:^|[\s])" + re.escape(op) + rb"[\s]", body))
    if rgb_images:
        reasons["%d RGB/Lab image(s)" % rgb_images] = True
    if rgb_ops:
        reasons["%d RGB fill/stroke operators" % rgb_ops] = True
    if transp:
        reasons["%d transparency state(s)" % transp] = True
    return list(reasons)


def ghostscript_convert(src_bytes, workdir, args):
    """Stage 1: convert any colour model / flatten transparency to DeviceCMYK
    with Ghostscript pdfwrite. Returns path of converted PDF.

    Image quality here is a fixed print-grade hand-off (QFactor 0.4, Adobe
    'printer' anchor ≈ JPEG q80); --quality refinement happens afterwards in
    pick_image_candidate(), which beats this encoder anyway.
    """
    gs = shutil.which("gs")
    if not gs:
        sys.exit("error: input needs colour conversion but Ghostscript is not "
                 "installed. Install it (brew install ghostscript) or convert "
                 "the file to CMYK first.")
    samples_h = {"none": "[1 1 1 1]", "422": "[2 1 1 1]",
                 "auto": "[1 1 1 1]", "420": "[2 2 1 1]"}[args.subsampling]
    samples_v = {"none": "[1 1 1 1]", "422": "[1 1 1 1]",
                 "auto": "[1 1 1 1]", "420": "[2 1 1 1]"}[args.subsampling]
    out_path = os.path.join(workdir, "stage1_cmyk.pdf")
    cmd = [
        gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
        "-sDEVICE=pdfwrite",
        "-o", out_path,
        "-dCompatibilityLevel=1.3",
        "-sColorConversionStrategy=CMYK",
        "-dProcessColorModel=/DeviceCMYK",
        "-dEmbedAllFonts=true", "-dSubsetFonts=true", "-dCompressFonts=true",
        "-dAutoRotatePages=/None",
        "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/DCTEncode",
        "-dDownsampleColorImages=false", "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dPassThroughJPEGImages=false",
        "-c",
        "<< /ColorImageDict << /QFactor 0.4 /HSamples %s /VSamples %s >> "
        "/GrayImageDict << /QFactor 0.4 /HSamples [1 1] /VSamples [1 1] >> "
        ">> setdistillerparams" % (samples_h, samples_v),
        "-f", os.path.abspath(args.input),
    ]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not os.path.isfile(out_path):
        err = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()
        sys.exit("error: Ghostscript conversion failed:\n  " +
                 "\n  ".join(err[-6:]))
    return out_path


# ----------------------------------------------------------------- XMP

XMP_TEMPLATE = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{tool} {ver}">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
    xmlns:pdf="http://ns.adobe.com/pdf/1.3/"
    xmlns:pdfx="http://ns.adobe.com/pdfx/1.3/"
    xmlns:pdfxid="http://www.npes.org/pdfx/ns/id/">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
   <xmp:CreatorTool>{creator}</xmp:CreatorTool>
   <xmp:CreateDate>{iso_date}</xmp:CreateDate>
   <xmp:ModifyDate>{iso_date}</xmp:ModifyDate>
   <xmpMM:DocumentID>{doc_uuid}</xmpMM:DocumentID>
   <xmpMM:InstanceID>{inst_uuid}</xmpMM:InstanceID>
   <pdf:Producer>{producer}</pdf:Producer>
   <pdfxid:GTS_PDFXVersion>{pdfx_version}</pdfxid:GTS_PDFXVersion>
   <pdfx:Trapped>False</pdfx:Trapped>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def build_xmp(title, creator, date_d, producer, pdfx_version, doc_hex, inst_hex):
    iso = datetime.strptime(date_d, "D:%Y%m%d%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ")
    return XMP_TEMPLATE.format(
        tool=TOOL_NAME,
        ver=TOOL_VERSION,
        title=xml_escape(title),
        creator=xml_escape(creator),
        iso_date=iso,
        doc_uuid=uuid_urn(doc_hex),
        inst_uuid=uuid_urn(inst_hex),
        producer=xml_escape(producer),
        pdfx_version=xml_escape(pdfx_version),
    ).encode("utf-8")


# ----------------------------------------------------------------- compress

def cmd_compress(args):
    src_path = os.path.abspath(args.input)
    with open(src_path, "rb") as f:
        orig_data = f.read()

    # ---- stage 0: parse + decide whether colour conversion is needed ------
    try:
        objs = parse_pdf(orig_data)
    except ValueError:
        objs = None
    reasons = find_conversion_reasons(objs) if objs is not None else [
        "structure not directly parseable (object streams / newer PDF)"
    ]
    converted = False
    data = orig_data
    if reasons and not args.no_convert:
        import tempfile
        workdir = tempfile.mkdtemp(prefix="pdfx1a-", dir=os.path.dirname(os.path.abspath(args.output)) or None)
        print("colour audit      : conversion needed (%s)" % "; ".join(reasons))
        stage1 = ghostscript_convert(orig_data, workdir, args)
        with open(stage1, "rb") as f:
            data = f.read()
        converted = True
        try:
            shutil.rmtree(workdir)
        except OSError:
            pass
    elif reasons:
        print("colour audit      : WARNING proceeding without conversion (%s); "
              "output will not be X-1a conforming" % "; ".join(reasons))

    try:
        objs = parse_pdf(data)
    except ValueError as ex:
        sys.exit("error: cannot parse PDF (%s).\n"
                 "This tool supports classic-xref PDFs without encryption or "
                 "object streams; run it through Ghostscript/qpdf first." % ex)
    root_ref, info_ref = get_trailer_refs(data)
    if root_ref is None:
        sys.exit("error: no trailer /Root found (encrypted or unusual PDF?)")

    jpegtran = None if args.no_jpeg_opt else shutil.which("jpegtran")

    icc_path = os.path.abspath(args.icc)
    if not os.path.isfile(icc_path):
        sys.exit("error: ICC profile not found: %s" % icc_path)
    with open(icc_path, "rb") as f:
        icc = f.read()

    pdfx_version = args.pdfx_version

    # Determinism: dates/IDs always derive from the ORIGINAL bytes, never
    # from a Ghostscript intermediate (which embeds wall-clock stamps).
    orig_root, orig_info_ref = get_trailer_refs(orig_data)
    orig_info_head = b""
    if orig_info_ref:
        try:
            orig_info_head = parse_pdf(orig_data)[orig_info_ref].head
        except ValueError:
            pass
    date_d = resolve_date(args.date, orig_info_head)

    def paren_str(key, default=""):
        kb = key.encode("latin1")
        info_head = objs[info_ref].head if info_ref else b""
        m = re.search(re.escape(kb) + rb"\s*\(([^)]*)\)", info_head)
        return m.group(1).decode("latin1") if m else default

    title = args.title or paren_str("/Title", os.path.basename(src_path))
    creator = args.creator or paren_str("/Creator", "")

    doc_hex = sha_hex(orig_data)[:32]
    inst_hex = sha_hex(b"instance:" + orig_data)[:32]

    # After RGB->CMYK conversion the image data is already a lossy generation
    # (Ghostscript re-encode), so keeping its large streams verbatim has no
    # fidelity value. Converted documents therefore always run the size
    # optimiser: explicit --quality wins, otherwise default q80 + auto.
    img_args = argparse.Namespace(**vars(args))
    if converted and img_args.quality is None:
        img_args.quality = 80
        if img_args.subsampling == "none":
            img_args.subsampling = "auto"
        print("optimise          : converted input -> defaulting to "
              "--quality 80 --subsampling auto (override with --quality)")

    # ---- 1. image optimisation -------------------------------------------
    stats = {"total": 0, "original": 0, "jpegtran": 0, "recompress": 0}
    saved_by = {"jpegtran": 0, "recompress": 0}
    for o in sorted(objs.values(), key=lambda x: x.num):
        h = o.dict_bytes()
        if o.stream is None or not re.search(rb"/Subtype\s*/Image", h):
            continue
        if not re.search(rb"/Filter\s*/?DCTDecode", h.replace(b"[", b"")):
            continue
        stats["total"] += 1
        new, method = pick_image_candidate(o.stream, jpegtran, img_args)
        if new is not None:
            saved_by[method] += len(o.stream) - len(new)
            stats[method] += 1
            o.head = re.sub(rb"/Length\s+\d+", b"/Length %d" % len(new), o.head, count=1)
            o.stream = new
        else:
            stats["original"] += 1

    # ---- 2. TrimBox on every page -----------------------------------------
    pages = find_pages(objs, root_ref)
    for pnum in pages:
        o = objs[pnum]
        mb = inherited_mediabox(objs, pnum)
        if mb is None:
            sys.exit("error: page object %d has no MediaBox anywhere up the tree" % pnum)
        if b"/TrimBox" not in o.head:
            ins = b"/TrimBox [ %s %s %s %s ] " % tuple(mb)
            i = o.head.rindex(b">>")
            o.head = o.head[:i] + ins + o.head[i:]

    # ---- 3. new objects: reserve all numbers up-front -----------------------
    nextnum = max(objs) + 1
    icc_num = nextnum; nextnum += 1
    xmp_num = nextnum; nextnum += 1
    intent_num = nextnum; nextnum += 1
    reserved_info_num = nextnum
    objs[icc_num] = Obj(icc_num, 0, b"<< /N 4 /Length %d >>" % len(icc), icc)
    xmp = build_xmp(title, creator, date_d, "%s %s" % (TOOL_NAME, TOOL_VERSION),
                    pdfx_version, doc_hex, inst_hex)
    xmp_head = b"<< /Type /Metadata /Subtype /XML /Length %d >>" % len(xmp)

    # ---- 4. catalog: attach Metadata (overwrite in place) + OutputIntents --
    cat = objs[root_ref]
    intent_dict = (
        b"<< /Type /OutputIntent /S /GTS_PDFX "
        b"/OutputCondition (Commercial and specialty printing) "
        b"/OutputConditionIdentifier (" + args.condition_id.encode("latin1") + b") "
        b"/RegistryName (http://www.color.org) "
        b"/DestOutputProfile %d 0 R >>" % icc_num
    )
    objs[intent_num] = Obj(intent_num, 0, intent_dict, None)

    existing_md = dict_get_ref(cat.dict_bytes(), b"/Metadata")
    if existing_md:
        objs[existing_md].head = xmp_head
        objs[existing_md].stream = xmp
    else:
        objs[xmp_num] = Obj(xmp_num, 0, xmp_head, xmp)
    i = cat.head.rindex(b">>")
    add = b""
    if b"/OutputIntents" not in cat.head:
        add += b"/OutputIntents [ %d 0 R ] " % intent_num
    if not existing_md:
        add += b"/Metadata %d 0 R " % xmp_num
    cat.head = cat.head[:i] + add + cat.head[i:]

    # ---- 5. info dict: PDF/X required keys ---------------------------------
    new_info_num = None
    if info_ref:
        ih = objs[info_ref].head
        for key in (b"/CreationDate", b"/ModDate", b"/Trapped", b"/GTS_PDFXVersion"):
            ih = re.sub(re.escape(key) + rb"\s*(\([^)]*\)|/[A-Za-z]+)", b"", ih, count=1)
        add = (
            b"/CreationDate (%s) /ModDate (%s) /GTS_PDFXVersion (%s) /Trapped /False "
            % (date_d.encode(), date_d.encode(), pdfx_version.encode())
        )
        j = ih.rindex(b">>")
        objs[info_ref].head = ih[:j] + b" " + add + ih[j:]
    else:
        new_info_num = reserved_info_num
        objs[new_info_num] = Obj(
            new_info_num,
            0,
            b"<< /Title (%s) /CreationDate (%s) /ModDate (%s) "
            b"/GTS_PDFXVersion (%s) /Trapped /False "
            % (
                title.encode("latin1"),
                date_d.encode(),
                date_d.encode(),
                pdfx_version.encode(),
            )
            + b">>",
            None,
        )

    # ---- 6. serialize -------------------------------------------------------
    out = serialize(objs, root_ref, new_info_num or info_ref, doc_hex, inst_hex)
    with open(os.path.abspath(args.output), "wb") as f:
        f.write(out)

    # ---- report --------------------------------------------------------------
    ib, ob = len(orig_data), len(out)
    print("input           : %s (%s bytes)" % (src_path, format(ib, ",")))
    print("output          : %s (%s bytes)" % (os.path.abspath(args.output), format(ob, ",")))
    print("size delta      : %+d bytes (%+.2f%%)" % (ob - ib, 100.0 * (ob - ib) / ib))
    if converted:
        print("colour convert  : RGB/Lab/transparency -> DeviceCMYK via Ghostscript "
              "(intermediate: %s bytes)" % format(len(data), ","))
    print(
        "jpeg streams    : %d | kept original: %d | lossless jpegtran: %d (-%s B)"
        % (stats["total"], stats["original"], stats["jpegtran"],
           format(saved_by["jpegtran"], ","))
    )
    if img_args.quality is not None or getattr(img_args, "preset", None) == "recompress":
        q = img_args.quality if img_args.quality is not None else img_args.jpeg_quality
        print(
            "lossy re-encode : quality %d, subsampling %s | applied to %d images (-%s B)"
            % (q, img_args.subsampling, stats["recompress"], format(saved_by["recompress"], ","))
        )
        if stats["total"] and stats["recompress"] == 0:
            print("note            : no image got smaller at this quality — "
                  "the originals were already better compressed. Try a lower "
                  "--quality or --subsampling auto/422/420.")
    print("pages w/TrimBox : %d" % len(pages))
    print("PDF/X version   : %s" % pdfx_version)
    print("ICC profile     : %s (%s bytes)" % (icc_path, format(len(icc), ",")))
    print("dates pinned to : %s" % date_d)
    if jpegtran is None and not args.no_jpeg_opt:
        print("note            : jpegtran not found — install libjpeg-turbo for lossless image savings")
    ok, _ = run_checks(os.path.abspath(args.output), quiet=True)
    print("self-check      : %s" % ("PASS" if ok else "FAIL (run `verify` for details)"))
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------- verify

def run_checks(path, quiet=False):
    with open(path, "rb") as f:
        data = f.read()
    fails, warns = [], []

    def emit(msg):
        if not quiet:
            print(msg)

    def check(name, cond, detail="", warn=False):
        status = "PASS" if cond else ("WARN" if warn else "FAIL")
        emit("[%s] %-42s%s" % (status, name, (" — " + detail) if detail else ""))
        if not cond:
            (warns if warn else fails).append(name)

    check("header is PDF 1.3/1.4",
          data.startswith(b"%PDF-1.3") or data.startswith(b"%PDF-1.4"),
          data[:8].decode("latin1"))

    try:
        objs = parse_pdf(data)
    except Exception as ex:
        check("parse", False, str(ex))
        return False, fails
    check("parse", True, "%d objects" % len(objs))

    root_ref, info_ref = get_trailer_refs(data)
    cat = objs[root_ref].dict_bytes()

    oi = re.search(rb"/OutputIntents\s*\[\s*(\d+)\s+\d+\s+R", cat)
    check("catalog.OutputIntents present", bool(oi))
    if oi:
        intent = objs[int(oi.group(1))].dict_bytes()
        check("intent.S == /GTS_PDFX", b"/S /GTS_PDFX" in intent)
        dest = dict_get_ref(intent, b"/DestOutputProfile")
        check("intent.DestOutputProfile present", dest is not None)
        if dest:
            st = objs[dest]
            check("ICC N==4 (CMYK)", b"/N 4" in st.dict_bytes())
            check("ICC acsp magic",
                  st.stream is not None and len(st.stream) > 40 and st.stream[36:40] == b"acsp",
                  "%s bytes" % format(len(st.stream or b""), ","))
        check("intent.OutputConditionIdentifier",
              bool(re.search(rb"/OutputConditionIdentifier\s*\([^)]+\)", intent)))

    md = dict_get_ref(cat, b"/Metadata")
    xmp_txt = objs[md].stream if md else b""
    check("catalog.Metadata (XMP) present", bool(md))

    info = objs[info_ref].dict_bytes() if info_ref else b""
    v = re.search(rb"/GTS_PDFXVersion\s*\(([^)]*)\)", info)
    check("info.GTS_PDFXVersion is PDF/X-1a",
          bool(v) and v.group(1).startswith(b"PDF/X-1a:"),
          v.group(1).decode() if v else "missing")
    xv = v.group(1).decode() if v else ""
    check("info.Trapped == False", b"/Trapped /False" in info)
    check("info.CreationDate", b"/CreationDate (D:" in info)
    check("info.ModDate", b"/ModDate (D:" in info)
    if xmp_txt and xv:
        flat = xmp_txt.replace(b"<![CDATA[", b"").replace(b"]]>", b"")
        check("xmp GTS_PDFXVersion matches Info",
              ("<pdfxid:GTS_PDFXVersion>%s<" % xv).encode() in flat)

    pages = find_pages(objs, root_ref)
    bad_trim = []
    for pn in pages:
        h = objs[pn].dict_bytes()
        mb = re.search(rb"/MediaBox\s*\[([^\]]*)\]", h)
        tb = re.search(rb"/TrimBox\s*\[([^\]]*)\]", h)
        if not mb or not tb or mb.group(1).split() != tb.group(1).split():
            bad_trim.append(pn)
    check("TrimBox==MediaBox on all pages", not bad_trim,
          "%d pages" % len(pages) if not bad_trim else "bad: %s" % bad_trim[:5])

    unembedded = []
    for o in objs.values():
        h = o.dict_bytes()
        if not re.search(rb"/Type\s*/Font\b", h):
            continue
        fd = None
        if dict_get_ref(h, b"/FontDescriptor"):
            fd = objs[dict_get_ref(h, b"/FontDescriptor")]
        elif dict_get_ref(h, b"/DescendantFonts"):
            df = re.search(rb"/DescendantFonts\s*\[\s*(\d+)\s+\d+\s+R", h)
            if df:
                dfh = objs[int(df.group(1))].dict_bytes()
                fdr = dict_get_ref(dfh, b"/FontDescriptor")
                if fdr:
                    fd = objs[fdr]
        if fd is None:
            continue
        fh = fd.dict_bytes()
        if not any(k in fh for k in (b"/FontFile", b"/FontFile2", b"/FontFile3")):
            bf = re.search(rb"/BaseFont\s*/([\w+-]+)", h)
            unembedded.append(bf.group(1).decode() if bf else "?")
    check("all fonts embedded", not unembedded, ", ".join(sorted(set(unembedded))[:5]))

    rgb_ops = 0
    transp = []
    non_cmyk_images = []
    for o in sorted(objs.values(), key=lambda x: x.num):
        h = o.dict_bytes()
        if re.search(rb"/Type\s*/ExtGState", h):
            for key in (b"/CA", b"/ca"):
                vals = re.findall(key + rb"\s+([\d.]+)", h)
                if any(float(x) < 1.0 for x in vals):
                    transp.append("ExtGState %s<1" % key.decode())
            bm = re.search(rb"/BM\s*/(\w+)", h)
            if bm and bm.group(1) != b"Normal":
                transp.append("BM/%s" % bm.group(1).decode())
        if re.search(rb"/Subtype\s*/Image", h) and o.stream is not None:
            if b"/SMask" in h:
                transp.append("image SMask obj %d" % o.num)
            if re.search(rb"/DeviceRGB|/CalRGB", h):
                non_cmyk_images.append("obj %d RGB" % o.num)
        if o.stream is None or b"/Length" not in h:
            continue
        if b"FlateDecode" not in h:
            continue
        try:
            body = zlib.decompress(o.stream)
        except Exception:
            continue
        if b"Tj" in body or b"TJ" in body or b"Do " in body:
            for op in (b"rg", b"RG"):
                rgb_ops += len(re.findall(rb"(?:^|[\s])" + re.escape(op) + rb"[\s]", body))
    check("no RGB colour operators in content", rgb_ops == 0, "%d found" % rgb_ops if rgb_ops else "")
    check("no transparency (SMask/blend/alpha)", not transp, "; ".join(transp[:4]))
    check("no RGB images", not non_cmyk_images, "; ".join(non_cmyk_images[:4]))

    # external extras --------------------------------------------------------
    if shutil.which("qpdf"):
        p = subprocess.run(["qpdf", "--check", path], capture_output=True)
        # qpdf exit codes: 0 = clean, 3 = warnings only (common with Apple-produced
        # JPEGs whose entropy segments carry padding; tolerant decoders handle them)
        if p.returncode == 0:
            check("qpdf --check", True)
        elif p.returncode == 3:
            first = next((l for l in (p.stdout + p.stderr).decode(errors="replace").splitlines()
                          if "WARNING" in l or "error" in l.lower()), "")
            check("qpdf --check", False, first[:70], warn=True)
        else:
            check("qpdf --check", False,
                  (p.stdout + p.stderr).decode(errors="replace").splitlines()[0][:70])
    else:
        emit("[skip] qpdf not installed")
    if shutil.which("pdfimages"):
        p = subprocess.run(["pdfimages", "-list", path], capture_output=True)
        rows = p.stdout.decode().splitlines()[2:]
        big_cmyk = small_other = 0
        ppis = []
        cmyk_ok = True
        for r in rows:
            cols = r.split()
            if len(cols) < 15:
                continue
            w, color, xp = int(cols[3]), cols[5], float(cols[12])
            if w >= 1000:
                big_cmyk += 1
                if color != "cmyk":
                    cmyk_ok = False
                    emit("[FAIL] large image colour=%s on page %s" % (color, cols[0]))
                ppis.append(xp)
            else:
                small_other += 1
        check("large images are CMYK", cmyk_ok, "%d images" % big_cmyk)
        if ppis:
            lo, hi = min(ppis), max(ppis)
            check("300 dpi preserved (>=295)", lo >= 295,
                  "min %.0f / max %.0f ppi" % (lo, hi))
    else:
        emit("[skip] poppler pdfimages not installed")

    total = len(fails)
    emit("")
    emit("RESULT: %s — %d failure(s), %d warning(s)"
         % ("CONFORMING (by structural checks)" if total == 0 else "NON-CONFORMING",
            total, len(warns)))
    return total == 0, fails


def cmd_verify(args):
    ok, _ = run_checks(os.path.abspath(args.file))
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------- main

def parse_quality(s):
    """'p90'/'P90'/90 -> 90"""
    m = re.fullmatch(r"[pP]?(\d{2})", str(s))
    if not m or not (10 <= int(m.group(1)) <= 95):
        raise argparse.ArgumentTypeError(
            "quality must be p10..p95 (e.g. p90, p75) or an integer 10..95")
    return int(m.group(1))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_icc = os.path.join(here, "vendor", "default_cmyk.icc")

    ap = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Lossless-pixel PDF/X-1a finaliser & configurable JPEG "
                    "recompressor (reproducible).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="finalise to reproducible PDF/X-1a")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--quality", type=parse_quality, metavar="pNN|NN",
                   help="re-encode JPEGs at this quality: p95 p90 p85 p80 p75 p70 "
                        "... or an integer 10..95 (implies lossy mode; requires Pillow)")
    c.add_argument("--subsampling", choices=["none", "422", "420", "auto"], default="none",
                   help="chroma subsampling of the C channel in lossy mode "
                        "(M/Y/K stay full-res). 'auto' keeps whichever is smaller. "
                        "default: none (max colour fidelity)")
    c.add_argument("--preset", choices=["lossless", "recompress"], default=None,
                   help=argparse.SUPPRESS)  # legacy alias for --quality
    c.add_argument("--jpeg-quality", type=int, default=80, help=argparse.SUPPRESS)
    c.add_argument("--icc", default=default_icc,
                   help="CMYK ICC profile for the OutputIntent (default: vendored default_cmyk.icc)")
    c.add_argument("--condition-id", default="CGATS TR001",
                   help="OutputConditionIdentifier string (default 'CGATS TR001')")
    c.add_argument("--pdfx-version", choices=["PDF/X-1a:2001", "PDF/X-1a:2003"],
                   default="PDF/X-1a:2001")
    c.add_argument("--date", help='pin Creation/ModDate, e.g. D:20260823120000Z '
                                  '(default: SOURCE_DATE_EPOCH > source date > fixed)')
    c.add_argument("--title", help="override document Title")
    c.add_argument("--creator", help="override Creator string")
    c.add_argument("--no-convert", action="store_true",
                   help="skip automatic Ghostscript RGB->CMYK conversion even "
                        "when the audit says it is needed")
    c.add_argument("--no-jpeg-opt", action="store_true",
                   help="skip jpegtran lossless optimisation")
    c.set_defaults(func=cmd_compress)

    v = sub.add_parser("verify", help="run PDF/X-1a structural checks")
    v.add_argument("file")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
