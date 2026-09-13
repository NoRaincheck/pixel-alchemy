#!/usr/bin/env python3
"""
pdfx4_embed.py — embed fonts and finalise to PDF/X-4 without flattening.

Unlike pdfx1a-compress (which converts everything to DeviceCMYK and flattens
transparency), this keeps RGB, live transparency and layers intact. The only
byte-level changes are embedded font subsets plus PDF/X-4 conformance metadata
(TrimBox, OutputIntent, Info, XMP, deterministic IDs).

  ./pdfx4_embed.py embed INPUT.pdf OUTPUT.pdf [--icc X.icc] [--condition-id ID]
                                              [--date D:...] [--title T]

Requires: ghostscript, qpdf + the sibling pdfx1a-compress skill.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

TOOL_NAME = "pdfx4-embed"
HERE = os.path.dirname(os.path.abspath(__file__))
FINALISER = os.path.join(HERE, "..", "pdfx1a-compress", "pdfx1a_compress.py")


def gs_embed(src, dst):
    gs = shutil.which("gs")
    if not gs:
        sys.exit("error: Ghostscript is required (brew install ghostscript)")
    cmd = [
        gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
        "-sDEVICE=pdfwrite",
        "-o", dst,
        "-dCompatibilityLevel=1.6",  # X-4 base version; keeps OCG layers + transparency
        "-sColorConversionStrategy=LeaveColorUnchanged",
        "-dEmbedAllFonts=true", "-dSubsetFonts=true", "-dCompressFonts=true",
        "-dAutoRotatePages=/None",
        "-dDownsampleColorImages=false", "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dPassThroughJPEGImages=true", "-dPassThroughJPXImages=true",
        "-f", src,
    ]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not os.path.isfile(dst):
        err = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()
        sys.exit("error: Ghostscript font-embedding failed:\n  " + "\n  ".join(err[-6:]))


def qpdf_normalize(src, dst):
    """Unpack object streams to classic objects (RGB/transparency/layers and
    all stream bytes preserved) so the deterministic finaliser can parse."""
    qpdf = shutil.which("qpdf")
    if not qpdf:
        sys.exit("error: qpdf is required (brew install qpdf)")
    p = subprocess.run([qpdf, "--object-streams=disable", src, dst],
                       capture_output=True)
    if p.returncode != 0 or not os.path.isfile(dst):
        err = (p.stdout + p.stderr).decode(errors="replace").strip().splitlines()
        sys.exit("error: qpdf normalization failed:\n  " + "\n  ".join(err[-6:]))


def cmd_embed(args):
    src = os.path.abspath(args.input)
    out = os.path.abspath(args.output)
    if not os.path.isfile(FINALISER):
        sys.exit("error: sibling finaliser not found: %s" % FINALISER)
    workdir = tempfile.mkdtemp(prefix="pdfx4-")
    try:
        stem = os.path.splitext(os.path.basename(src))[0]
        stage_gs = os.path.join(workdir, stem + ".stage1.pdf")
        stage1 = os.path.join(workdir, stem + ".stage2.pdf")
        print("stage 1         : embed fonts (RGB/transparency/layers preserved)")
        gs_embed(src, stage_gs)
        print("stage 2         : unpack object streams (qpdf, bytes preserved)")
        qpdf_normalize(stage_gs, stage1)
        cmd = [sys.executable, FINALISER, "compress", stage1, out,
               "--pdfx-version", "PDF/X-4", "--no-jpeg-opt",
               "--id-source", src,
               "--condition-id", args.condition_id]
        if args.icc:
            cmd += ["--icc", os.path.abspath(args.icc)]
        for flag, val in (("--date", args.date), ("--title", args.title),
                          ("--creator", args.creator)):
            if val:
                cmd += [flag, val]
        print("stage 3         : PDF/X-4 conformance via pdfx1a-compress")
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            ib, ob = os.path.getsize(src), os.path.getsize(out)
            print("true delta      : %+d bytes (%+.2f%%) vs original input"
                  % (ob - ib, 100.0 * (ob - ib) / ib))
        sys.exit(rc)
    finally:
        try:
            shutil.rmtree(workdir)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Embed fonts and finalise to PDF/X-4 without flattening "
                    "RGB/transparency/layers.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="embed fonts + finalise to PDF/X-4")
    e.add_argument("input")
    e.add_argument("output")
    e.add_argument("--icc", default=None,
                   help="CMYK ICC profile for the OutputIntent "
                        "(default: pdfx1a-compress vendored profile)")
    e.add_argument("--condition-id", default="CGATS TR001",
                   help="OutputConditionIdentifier string")
    e.add_argument("--date", help="pin Creation/ModDate, e.g. D:20260823120000Z")
    e.add_argument("--title", help="override document Title")
    e.add_argument("--creator", help="override Creator string")
    e.set_defaults(func=cmd_embed)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
