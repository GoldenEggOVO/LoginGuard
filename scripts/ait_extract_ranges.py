from pathlib import Path
import hashlib
import struct
import zlib
from scripts.ait_zip_index import entries


def extract_member(block, expected_name, compressed_size, expected_size, expected_crc):
    if len(block) < 30:
        raise ValueError('Truncated ZIP local header')
    fields = struct.unpack_from('<4s5H3L2H', block)
    if fields[0] != b'PK\x03\x04' or fields[3] != 8 or expected_size > 10_000_000:
        raise ValueError('Unsupported ZIP member')
    name_len, extra_len = fields[9], fields[10]
    name = block[30:30 + name_len].decode('utf-8')
    if name != expected_name:
        raise ValueError('ZIP member name mismatch')
    start = 30 + name_len + extra_len
    payload = block[start:start + compressed_size]
    if len(payload) != compressed_size:
        raise ValueError('Truncated ZIP member')
    result = zlib.decompress(payload, -15)
    if len(result) != expected_size or zlib.crc32(result) != expected_crc:
        raise ValueError('ZIP member size or CRC mismatch')
    return result


def main():
    root = Path('data/public/ait_lds')
    index = {row[0]: row for row in entries(root / 'zip-directory.bin')}
    selected = [('gather/intranet_server/logs/auth.log', 'auth-member.bin', 'auth.log'),
                ('labels/intranet_server/logs/auth.log', 'auth-label-member.bin', 'auth.labels.jsonl'),
                ('dataset.yaml', 'dataset-member.bin', 'dataset.yaml')]
    out = root / 'selected'
    out.mkdir(exist_ok=True)
    for member, source, target in selected:
        _, method, crc, compressed, size, _ = index[member]
        if method != 8:
            raise ValueError('Expected deflate compression')
        data = extract_member((root / source).read_bytes(), member, compressed, size, crc)
        (out / target).write_bytes(data)
        print(f'{member}: bytes={len(data)} crc32={crc:08x} sha256={hashlib.sha256(data).hexdigest()}')


if __name__ == '__main__':
    main()
