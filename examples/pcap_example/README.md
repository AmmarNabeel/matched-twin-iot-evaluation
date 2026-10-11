One complete run (M1/M2 family 4100000, arm A) of the study: the PCAP, the trigger file (client id and trigger time) and the features that were extracted from it.
`python scripts/extract_features.py` parses the PCAP again with tshark and compares the result with `expected_features.csv`.
