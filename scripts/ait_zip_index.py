from pathlib import Path
import struct


def entries(path):
    data = Path(path).read_bytes()
    offset = 0
    while data[offset:offset + 4] == b'PK\x01\x02':
        fields = struct.unpack_from('<4s6H3L5H2L', data, offset)
        name_len, extra_len, comment_len = fields[10:13]
        name = data[offset + 46:offset + 46 + name_len].decode('utf-8')
        yield name, fields[4], fields[7], fields[8], fields[9], fields[16]
        offset += 46 + name_len + extra_len + comment_len


if __name__ == '__main__':
    for name, method, crc, compressed, size, local_offset in entries('data/public/ait_lds/zip-directory.bin'):
        if '/logs/auth.log' in name:
            print(f'{name}\tmethod={method}\tcrc={crc:08x}\tcompressed={compressed}\tsize={size}\toffset={local_offset}')
