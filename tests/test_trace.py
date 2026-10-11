import os
import shutil
import tempfile
import unittest

from postdrift_auth.trace import audit_segments, frames_to_events, parse_pcap, parse_pcap_audited, run_tshark, run_tshark_tcp_rows, read_trace_csv, write_trace_csv, sha256_file
from tests.pcap_synth import PcapWriter, mqtt_connect, mqtt_publish, mqtt_subscribe

T0 = 1_000_000.0


@unittest.skipUnless(shutil.which("tshark"), "tshark not installed")
class PcapParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.pcap = os.path.join(cls.dir, "syn.pcap")
        w = PcapWriter()
        w.add(T0 + 0.000000, 40001, mqtt_connect("A"))
        w.add(T0 + 0.100000, 40002, mqtt_connect("B"))
        w.add(T0 + 1.000000, 40001, mqtt_subscribe(1, ["f/1", "f/2"]))
        w.add(T0 + 2.123456, 40001, mqtt_publish("t/a", 10) + mqtt_publish("t/b", 20, dup=True, msgid=2))  # two PDUs in one frame
        w.add(T0 + 3.000000, 40001, mqtt_publish("t/c", 30, msgid=3))
        w.add(T0 + 3.500000, 40002, mqtt_publish("t/z", 0, msgid=1))
        w.add(T0 + 4.000000, 40003, mqtt_publish("t/orphan", 5))                                      # stream without CONNECT
        w.add(T0 + 4.500000, 1883, mqtt_publish("t/down", 5), src_ip=(10, 0, 0, 1), dst_ip=(10, 0, 0, 2), dst_port=40001)  # broker -> client
        w.write(cls.pcap)
        cls.events, cls.diag = parse_pcap(cls.pcap, repr(T0))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_required_fields_connect_subscribe_publish(self):
        e = [x for x in self.events if x.client_id == "A"]
        self.assertEqual([(x.kind, x.topic, x.size, x.dup) for x in e],
                         [("CONNECT", None, None, None), ("SUBSCRIBE", "f/1", None, None), ("SUBSCRIBE", "f/2", None, None),
                          ("PUBLISH", "t/a", 10, False), ("PUBLISH", "t/b", 20, True), ("PUBLISH", "t/c", 30, False)])

    def test_timestamps_relative_to_t0_with_microsecond_precision(self):
        pa = [x for x in self.events if x.client_id == "A" and x.kind == "PUBLISH"]
        self.assertAlmostEqual(pa[0].t, 2.123456, places=6)
        self.assertAlmostEqual(pa[2].t, 3.0, places=6)

    def test_identity_attribution_by_connect(self):
        self.assertEqual({x.client_id for x in self.events}, {"A", "B"})
        self.assertEqual([x.size for x in self.events if x.client_id == "B" and x.kind == "PUBLISH"], [0])

    def test_unattributed_and_downstream_frames_dropped_and_counted(self):
        self.assertNotIn("t/orphan", [x.topic for x in self.events])
        self.assertNotIn("t/down", [x.topic for x in self.events])
        self.assertEqual(self.diag["unattributed_dropped"], 1)
        self.assertEqual(self.diag["non_client_dropped"], 1)

    def test_csv_roundtrip_and_pcap_hash(self):
        p = os.path.join(self.dir, "t.csv")
        write_trace_csv(self.events, p)
        self.assertEqual(read_trace_csv(p), self.events)
        self.assertEqual(len(sha256_file(self.pcap)), 64)

    def test_parse_is_deterministic(self):
        self.assertEqual(parse_pcap(self.pcap, repr(T0))[0], self.events)

    def test_hole_filling_segment_is_kept_and_true_duplicate_is_dropped(self):
        """Stage 2C regression: the segment that fills a hole (seq lower than an already-seen segment) carries a real PUBLISH."""
        w = PcapWriter()
        w.add(T0 + 0.0, 40001, mqtt_connect("A"))
        p1, p2, p3 = mqtt_publish("t/1", 10, msgid=1), mqtt_publish("t/2", 11, msgid=2), mqtt_publish("t/3", 12, msgid=3)
        s1 = w.seq[((10, 0, 0, 2), 40001)]
        s2, s3 = s1 + len(p1), s1 + len(p1) + len(p2)
        w.add(T0 + 5.0, 40001, p2, seq=s2)          # p1 (seq s1) was lost on the way; p2 arrives first
        w.add(T0 + 5.1, 40001, p3, seq=s3)
        w.add(T0 + 5.6, 40001, p1, seq=s1)          # the "retransmission" of p1 is the FIRST copy the broker interface sees
        w.add(T0 + 5.9, 40001, p1, seq=s1)          # a true duplicate (e.g. spurious retransmit after a lost ACK)
        path = os.path.join(self.dir, "hole.pcap")
        w.write(path)
        ev, d = parse_pcap(path, repr(T0))
        pubs = [(round(e.t, 3), e.topic) for e in ev if e.kind == "PUBLISH"]
        self.assertEqual(pubs, [(5.0, "t/2"), (5.1, "t/3"), (5.6, "t/1")])
        self.assertEqual((d["duplicate_segments_dropped"], d["seq_overlap_conflicts"]), (1, 0))

    def test_overlap_conflict_is_counted_not_silently_merged(self):
        w = PcapWriter()
        w.add(T0, 40001, mqtt_connect("A"))
        s = w.seq[((10, 0, 0, 2), 40001)]
        w.add(T0 + 1, 40001, mqtt_publish("t/1", 10), seq=s)
        w.add(T0 + 2, 40001, mqtt_publish("t/1", 10) + mqtt_publish("t/2", 10, msgid=2), seq=s)   # same seq, longer segment
        path = os.path.join(self.dir, "overlap.pcap")
        w.write(path)
        _, d = parse_pcap(path, repr(T0))
        self.assertEqual(d["seq_overlap_conflicts"], 1)

    def test_dup_flag_counted_for_diagnostics(self):
        self.assertEqual(self.diag["dup_flag_publishes"], 1)

    def test_segment_audit_passes_when_each_pdu_fits_one_segment(self):
        _, _, audit = parse_pcap_audited(self.pcap, repr(T0))
        self.assertEqual((audit["streams_with_byte_mismatch"], audit["frames_with_pdu_len_mismatch"]), (0, 0))
        self.assertEqual(audit["tcp_payload_bytes"], audit["pdu_bytes"])

    def test_segment_audit_detects_a_pdu_split_over_two_segments(self):
        w = PcapWriter()
        w.add(T0, 40001, mqtt_connect("A"))
        pub = mqtt_publish("t/a", 40)
        w.add(T0 + 1, 40001, pub[:20])        # first part of one PUBLISH
        w.add(T0 + 1.2, 40001, pub[20:])      # remainder
        path = os.path.join(self.dir, "split.pcap")
        w.write(path)
        _, _, audit = parse_pcap_audited(path, repr(T0))
        self.assertTrue(audit["streams_with_byte_mismatch"] > 0 or audit["frames_with_pdu_len_mismatch"] > 0, msg=audit)

    def test_segment_audit_detects_partial_pdu_that_is_never_completed(self):
        w = PcapWriter()
        w.add(T0, 40001, mqtt_connect("A"))
        w.add(T0 + 1, 40001, mqtt_publish("t/a", 40)[:20])
        path = os.path.join(self.dir, "partial.pcap")
        w.write(path)
        _, _, audit = parse_pcap_audited(path, repr(T0))
        self.assertEqual(audit["streams_with_byte_mismatch"], 1)

    def test_tshark_exposes_fields_needed(self):
        l = run_tshark(self.pcap)[0]["_source"]["layers"]
        self.assertIn("frame.time_epoch", l["frame"])
        self.assertIn("tcp.stream", l["tcp"])
        self.assertIn("mqtt.clientid", l["mqtt"])


if __name__ == "__main__":
    unittest.main()
