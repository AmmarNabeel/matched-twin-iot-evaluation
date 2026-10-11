"""Tiny synthetic PCAP writer (Ethernet/IPv4/TCP carrying MQTT) for deterministic parser tests. Test-support only."""
import struct


def _vl(n: int) -> bytes:
    out = b""
    while True:
        d, n = n % 128, n // 128
        out += bytes([d | (128 if n else 0)])
        if not n:
            return out


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack(">H", len(b)) + b


def mqtt_connect(cid: str) -> bytes:
    body = _s("MQTT") + bytes([4, 2]) + struct.pack(">H", 60) + _s(cid)
    return bytes([0x10]) + _vl(len(body)) + body


def mqtt_subscribe(msgid: int, filters) -> bytes:
    body = struct.pack(">H", msgid) + b"".join(_s(f) + bytes([1]) for f in filters)
    return bytes([0x82]) + _vl(len(body)) + body


def mqtt_publish(topic: str, payload_len: int, dup: bool = False, msgid: int = 1) -> bytes:
    body = _s(topic) + struct.pack(">H", msgid) + b"x" * payload_len
    return bytes([0x30 | 0x02 | (0x08 if dup else 0)]) + _vl(len(body)) + body


class PcapWriter:
    def __init__(self):
        self.packets = []
        self.seq = {}

    def add(self, t: float, src_port: int, mqtt_bytes: bytes, seq=None, src_ip=(10, 0, 0, 2), dst_ip=(10, 0, 0, 1), dst_port=1883):
        key = (src_ip, src_port)
        s = self.seq.get(key, 1000) if seq is None else seq
        if seq is None:
            self.seq[key] = s + len(mqtt_bytes)
        tcp = struct.pack(">HHIIBBHHH", src_port, dst_port, s, 1, 5 << 4, 0x18, 65535, 0, 0) + mqtt_bytes
        ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0, bytes(src_ip), bytes(dst_ip)) + tcp
        eth = b"\x02\x00\x00\x00\x00\x01" + b"\x02\x00\x00\x00\x00\x02" + b"\x08\x00" + ip
        self.packets.append((t, eth))

    def write(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
            for t, p in self.packets:
                sec = int(t)
                f.write(struct.pack("<IIII", sec, int(round((t - sec) * 1e6)), len(p), len(p)) + p)
