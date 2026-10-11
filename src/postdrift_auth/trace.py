"""Canonical trace: TShark-dissected PCAP -> client->broker MQTT events (CONNECT, SUBSCRIBE, PUBLISH).

Fields used (tshark -T json, MQTT dissector): frame.time_epoch, tcp.stream, tcp.seq, tcp.len, tcp.dstport,
mqtt.hdrflags_tree/mqtt.msgtype, mqtt.clientid (CONNECT), mqtt.topic (PUBLISH topic / SUBSCRIBE filters), mqtt.msg (payload hex),
mqtt.dupflag (diagnostic only, not a feature).

TCP handling (Stage 2C finding): with tshark's default TCP sequence analysis, a segment that fills a hole after later segments
arrived (typical under netem loss/reordering) is labelled "retransmission" and its MQTT content is NOT dissected, so a PUBLISH the
broker did receive vanishes. We therefore run tshark with `tcp.analyze_sequence_numbers:FALSE` (every segment is dissected) and
drop only TRUE duplicates ourselves: a segment with the same (tcp.stream, tcp.seq, tcp.len) as an earlier frame. An event's time is
the capture time of the first frame that carried it (segment arrival at the broker interface, not in-order delivery time).
Assumption (checked and reported): every MQTT PDU lies inside one TCP segment; a duplicate seq with a different length is counted
as an overlap conflict.
Identity: PUBLISH/SUBSCRIBE carry no client_id, so each TCP stream is attributed to the client_id of the CONNECT that opened
it. Streams whose CONNECT is not in the capture are counted and dropped, never guessed.
No labels, scenario names or ground-truth files are read here.
"""
from __future__ import annotations

import calendar
import csv
import hashlib
import json
import re
import subprocess
from decimal import Decimal

from .features import Event

BROKER_PORT = "1883"
TYPES = {"1": "CONNECT", "3": "PUBLISH", "8": "SUBSCRIBE"}
_TIME = re.compile(r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)\.(\d+)Z$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pairs_hook(pairs):
    d: dict = {}
    for k, v in pairs:
        if k in d:
            if k not in d.setdefault("__multi__", set()):
                d["__multi__"].add(k)
                d[k] = [d[k]]
            d[k].append(v)
        else:
            d[k] = v
    d.pop("__multi__", None)
    return d


def _as_list(x):
    return x if isinstance(x, list) else [x]


def epoch_decimal(s: str) -> Decimal:
    m = _TIME.match(s)
    if not m:
        raise ValueError(f"unrecognised frame.time_epoch {s!r}")
    y, mo, d, h, mi, se, frac = m.groups()
    return Decimal(calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(se)))) + Decimal("0." + frac)


def run_tshark(pcap: str, tshark: str = "tshark") -> list[dict]:
    p = subprocess.run([tshark, "-r", pcap, "-o", "tcp.analyze_sequence_numbers:FALSE", "-Y", "mqtt", "-T", "json"],
                       capture_output=True, check=True)
    return json.loads(p.stdout.decode() or "[]", object_pairs_hook=_pairs_hook)


def frames_to_events(frames: list[dict], t0: Decimal | str | float) -> tuple[list[Event], dict]:
    """Client->broker events with t in seconds since T0. Returns (events, diagnostics)."""
    t0 = Decimal(repr(t0)) if isinstance(t0, float) else Decimal(t0)
    stream_cid: dict[str, str] = {}
    pending: list[tuple[str, float, str, str | None, int | None, bool | None]] = []
    diag = {"frames": len(frames), "duplicate_segments_dropped": 0, "seq_overlap_conflicts": 0, "non_client_dropped": 0,
            "unattributed_dropped": 0, "dup_flag_publishes": 0}
    seen_seq: dict[tuple[str, str], str] = {}
    for fr in frames:
        layers = fr["_source"]["layers"]
        tcp = layers["tcp"]
        if tcp["tcp.dstport"] != BROKER_PORT:
            diag["non_client_dropped"] += 1
            continue
        key = (tcp["tcp.stream"], tcp["tcp.seq"])
        if key in seen_seq:
            diag["duplicate_segments_dropped" if seen_seq[key] == tcp["tcp.len"] else "seq_overlap_conflicts"] += 1
            continue
        seen_seq[key] = tcp["tcp.len"]
        t = float(epoch_decimal(layers["frame"]["frame.time_epoch"]) - t0)
        stream = tcp["tcp.stream"]
        for m in _as_list(layers["mqtt"]):
            kind = TYPES.get(m["mqtt.hdrflags_tree"]["mqtt.msgtype"])
            if kind is None:
                continue
            if kind == "CONNECT":
                stream_cid[stream] = m["mqtt.clientid"]
                pending.append((stream, t, kind, None, None, None))
            elif kind == "SUBSCRIBE":
                for f in _as_list(m["mqtt.topic"]):
                    pending.append((stream, t, kind, f, None, None))
            else:
                hexs = m.get("mqtt.msg", "")
                size = (len(hexs) + 1) // 3 if hexs else 0
                dup = m["mqtt.hdrflags_tree"]["mqtt.dupflag"] == "1"
                diag["dup_flag_publishes"] += dup
                pending.append((stream, t, kind, m["mqtt.topic"], size, dup))
    events = []
    for stream, t, kind, topic, size, dup in pending:
        cid = stream_cid.get(stream)
        if cid is None:
            diag["unattributed_dropped"] += 1
            continue
        events.append(Event(t, kind, cid, topic, size, dup))
    events.sort(key=lambda e: (e.t, e.kind))
    return events, diag


def run_tshark_tcp_rows(pcap: str, tshark: str = "tshark") -> list[tuple[str, str, str]]:
    """(tcp.stream, tcp.seq, tcp.len) of every client->broker segment with payload, whether or not MQTT was dissected."""
    p = subprocess.run([tshark, "-r", pcap, "-o", "tcp.analyze_sequence_numbers:FALSE", "-Y", f"tcp.dstport=={BROKER_PORT} && tcp.len>0",
                        "-T", "fields", "-e", "tcp.stream", "-e", "tcp.seq", "-e", "tcp.len"], capture_output=True, check=True)
    return [tuple(ln.split("\t")) for ln in p.stdout.decode().splitlines() if ln.strip()]


def _pdu_size(m: dict) -> int:
    rl = int(m["mqtt.len"])
    return 1 + (1 if rl < 128 else 2 if rl < 16384 else 3 if rl < 2097152 else 4) + rl


def audit_segments(frames: list[dict], tcp_rows: list[tuple[str, str, str]]) -> dict:
    """Check the one-MQTT-message-per-TCP-segment assumption.

    Per client->broker segment (after duplicate removal) the decoded MQTT PDU sizes must add up to tcp.len exactly, and per stream
    the total TCP payload bytes must equal the total decoded PDU bytes. A PDU split over segments, or a segment holding a partial PDU,
    breaks one of the two equalities.
    """
    seen: dict[tuple[str, str], str] = {}
    pdu_by_stream: dict[str, int] = {}
    frame_mismatch = 0
    for fr in frames:
        layers = fr["_source"]["layers"]
        tcp = layers["tcp"]
        if tcp["tcp.dstport"] != BROKER_PORT:
            continue
        key = (tcp["tcp.stream"], tcp["tcp.seq"])
        if key in seen:
            continue
        seen[key] = tcp["tcp.len"]
        size = sum(_pdu_size(m) for m in _as_list(layers["mqtt"]))
        pdu_by_stream[tcp["tcp.stream"]] = pdu_by_stream.get(tcp["tcp.stream"], 0) + size
        frame_mismatch += size != int(tcp["tcp.len"])
    seen_rows: set[tuple[str, str, str]] = set()
    tcp_by_stream: dict[str, int] = {}
    for st, seq, ln in tcp_rows:
        if (st, seq) in seen_rows:
            continue
        seen_rows.add((st, seq))
        tcp_by_stream[st] = tcp_by_stream.get(st, 0) + int(ln)
    bad = {st: (tcp_by_stream.get(st, 0), pdu_by_stream.get(st, 0)) for st in set(tcp_by_stream) | set(pdu_by_stream)
           if tcp_by_stream.get(st, 0) != pdu_by_stream.get(st, 0)}
    return {"streams": len(tcp_by_stream), "tcp_payload_bytes": sum(tcp_by_stream.values()), "pdu_bytes": sum(pdu_by_stream.values()),
            "streams_with_byte_mismatch": len(bad), "frames_with_pdu_len_mismatch": frame_mismatch, "mismatching_streams": bad}


def parse_pcap(pcap: str, t0, tshark: str = "tshark") -> tuple[list[Event], dict]:
    return frames_to_events(run_tshark(pcap, tshark), t0)


def parse_pcap_audited(pcap: str, t0, tshark: str = "tshark") -> tuple[list[Event], dict, dict]:
    frames = run_tshark(pcap, tshark)
    events, diag = frames_to_events(frames, t0)
    return events, diag, audit_segments(frames, run_tshark_tcp_rows(pcap, tshark))


def write_trace_csv(events, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "kind", "client_id", "topic", "size", "dup"])
        for e in events:
            w.writerow([repr(e.t), e.kind, e.client_id, e.topic or "", "" if e.size is None else e.size,
                        "" if e.dup is None else int(e.dup)])


def read_trace_csv(path: str) -> list[Event]:
    with open(path, newline="") as f:
        return [Event(float(r["t"]), r["kind"], r["client_id"], r["topic"] or None,
                      int(r["size"]) if r["size"] != "" else None, bool(int(r["dup"])) if r["dup"] != "" else None)
                for r in csv.DictReader(f)]
