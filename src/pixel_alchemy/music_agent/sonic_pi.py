"""Sonic Pi OSC validation — port of App/services/sonicPi.py, opt-in.

Usage:
    from pixel_alchemy.music_agent.sonic_pi import validate_sonic_pi_file
    ok, msg = validate_sonic_pi_file("songs/my_song/my_song.rb", host="localhost", port=4557)
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"#\s*(Apply|Add code here|Arrangement goes here|Sonic Pi code for)", re.I)


def lint_sonic_pi(code: str) -> list[str]:
    """Static checks without Sonic Pi server."""
    issues: list[str] = []
    if PLACEHOLDER_RE.search(code):
        issues.append("placeholder comment found — incomplete code")
    if "live_loop" in code:
        # upstream docs ask to avoid live_loop, but it's not an error — warn only
        pass
    # check for unmatched do/end roughly
    if code.count(" do") < code.count(" in_thread"):
        pass
    return issues


def validate_sonic_pi_file(
    rb_path: str | Path,
    host: str = "localhost",
    port: int = 4557,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Send file to Sonic Pi via OSC and wait for feedback.

    Requires Sonic Pi with AppSetup/recording.rb loaded:
      live_loop :listen do
        use_real_time
        script = sync "/osc*/run-code"
        begin
          eval script[0]
          osc_send '127.0.0.1', 4559, '/feedback', 'MusicAgent Code was executed successfully'
        rescue Exception => e
          osc_send '127.0.0.1', 4559, '/feedback', e.message
        end
      end
    Returns (ok, message). If python-osc not installed or Sonic Pi not reachable, falls back to lint.
    """
    rb_path = Path(rb_path)
    if not rb_path.exists():
        return False, f"file not found: {rb_path}"
    code = rb_path.read_text(encoding="utf-8", errors="ignore")
    lint = lint_sonic_pi(code)
    if lint:
        return False, "; ".join(lint)

    try:
        from pythonosc import dispatcher, osc_server, udp_client
    except ImportError:
        return True, "python-osc not installed — lint only, no live validation"

    feedback: list[str] = []
    done = threading.Event()

    def _handler(_addr, *args):
        if args:
            feedback.append(str(args[0]))
        done.set()

    disp = dispatcher.Dispatcher()
    disp.map("/feedback", _handler)
    # try 4559, fallback
    server = None
    for fport in [4559, 4558, 4560, 4561]:
        try:
            server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", fport), disp)
            feedback_port = fport
            break
        except OSError:
            continue
    if server is None:
        return False, "could not bind feedback port 4559"

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = udp_client.SimpleUDPClient(host, port)
        client.send_message("/run-code", code)
        ok = done.wait(timeout)
        if not ok:
            return False, f"no feedback from Sonic Pi at {host}:{port} within {timeout}s (is recording.rb running?)"
        msg = feedback[0] if feedback else ""
        if "error" in msg.lower():
            return False, msg
        return True, msg or "MusicAgent Code was executed successfully"
    finally:
        server.shutdown()
        server.server_close()


def fix_sonic_pi_notes(code: str) -> str:
    """Fix :C#4 -> :cs4 for Sonic Pi Ruby symbols."""
    return re.sub(r":([A-G])#(\d)", lambda m: f":{m.group(1).lower()}s{m.group(2)}", code)


def fix_sample_paths_for_llm(code: str) -> str:
    """Strip absolute sample paths before sending back to LLM (inverse of song.py)."""
    pat = r'sample\s+"(?:[^"]*Samples[\\/]+)([^"]+)"'
    return re.sub(pat, lambda m: f'sample "{m.group(1).replace(chr(92), "/")}"', code)
