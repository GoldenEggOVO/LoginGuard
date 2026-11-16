import struct
import zlib
import pytest
from scripts.ait_extract_ranges import extract_member


def test_extract_checks_exact_name_size_and_crc():
    data = b'one\ntwo\n'
    encoder = zlib.compressobj(wbits=-15)
    compressed = encoder.compress(data) + encoder.flush()
    name = b'gather/host/logs/auth.log'
    crc = zlib.crc32(data)
    header = struct.pack('<4s5H3L2H', b'PK\x03\x04', 20, 0, 8, 0, 0, crc, len(compressed), len(data), len(name), 0)
    block = header + name + compressed + b'trailing data'
    assert extract_member(block, name.decode(), len(compressed), len(data), crc) == data
    with pytest.raises(ValueError):
        extract_member(block, '../other', len(compressed), len(data), crc)
    with pytest.raises(ValueError):
        extract_member(block, name.decode(), len(compressed), len(data), crc ^ 1)
