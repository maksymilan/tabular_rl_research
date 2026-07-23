#!/usr/bin/env python3
"""Fetch one member from a remote ZIP using HTTP byte ranges.

This is useful for OmniSQL's 22 GB evaluation archive: the BIRD prompt file
can be reproduced without downloading unrelated training data and databases.
The URL must redirect to an object store that supports byte ranges.
"""

from __future__ import annotations

import argparse
import binascii
import os
import pathlib
import struct
import subprocess
import tempfile
import zlib


CLASSIC_EOCD = b"PK\x05\x06"
ZIP64_EOCD = b"PK\x06\x06"
ZIP64_LOCATOR = b"PK\x06\x07"
CENTRAL_HEADER = b"PK\x01\x02"
LOCAL_HEADER = b"PK\x03\x04"
ZIP64_EXTRA_ID = 0x0001


def run_curl(args: list[str]) -> bytes:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail",
        "--retry",
        "5",
        "--connect-timeout",
        "30",
        "--max-time",
        "600",
        *args,
    ]
    return subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout


def resolve_url(url: str) -> tuple[str, int]:
    headers = run_curl(["--head", url]).decode("latin-1")
    location = ""
    linked_size = 0
    for raw_line in headers.splitlines():
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        if key.lower() == "location":
            location = value.strip()
        elif key.lower() in {"x-linked-size", "content-length"}:
            try:
                linked_size = max(linked_size, int(value.strip()))
            except ValueError:
                pass
    if not location:
        location = url

    final_headers = run_curl(["--head", location]).decode("latin-1")
    total_size = 0
    accepts_ranges = False
    for raw_line in final_headers.splitlines():
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        if key.lower() == "content-length":
            total_size = int(value.strip())
        elif key.lower() == "accept-ranges" and value.strip().lower() == "bytes":
            accepts_ranges = True
    if total_size <= 0:
        total_size = linked_size
    if total_size <= 0:
        raise RuntimeError("remote object did not report a positive content length")
    if not accepts_ranges:
        raise RuntimeError("remote object did not advertise byte-range support")
    return location, total_size


def fetch_range(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"invalid byte range: {start}-{end}")
    expected = end - start + 1
    payload = run_curl(
        [
            "--range",
            f"{start}-{end}",
            "--max-filesize",
            str(expected + 1),
            url,
        ]
    )
    if len(payload) != expected:
        raise RuntimeError(
            f"range {start}-{end} returned {len(payload)} bytes; expected {expected}"
        )
    return payload


def parse_zip64_extra(
    extra: bytes,
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int,
    disk_start: int,
) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        field = extra[cursor : cursor + field_size]
        cursor += field_size
        if field_id != ZIP64_EXTRA_ID:
            continue
        field_cursor = 0

        def read_u64() -> int:
            nonlocal field_cursor
            value = struct.unpack_from("<Q", field, field_cursor)[0]
            field_cursor += 8
            return value

        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = read_u64()
        if compressed_size == 0xFFFFFFFF:
            compressed_size = read_u64()
        if local_offset == 0xFFFFFFFF:
            local_offset = read_u64()
        if disk_start == 0xFFFF:
            field_cursor += 4
        break
    return uncompressed_size, compressed_size, local_offset


def central_directory_location(url: str, total_size: int) -> tuple[int, int]:
    tail_size = min(total_size, 1 << 20)
    tail_offset = total_size - tail_size
    tail = fetch_range(url, tail_offset, total_size - 1)
    eocd_index = tail.rfind(CLASSIC_EOCD)
    if eocd_index < 0:
        raise RuntimeError("classic ZIP end-of-central-directory record not found")
    eocd = tail[eocd_index : eocd_index + 22]
    if len(eocd) < 22:
        raise RuntimeError("truncated classic ZIP end-of-central-directory record")
    central_size, central_offset = struct.unpack_from("<II", eocd, 12)
    if central_size != 0xFFFFFFFF and central_offset != 0xFFFFFFFF:
        return central_offset, central_size

    locator_index = tail.rfind(ZIP64_LOCATOR, 0, eocd_index)
    if locator_index < 0:
        raise RuntimeError("ZIP64 locator not found")
    zip64_eocd_offset = struct.unpack_from("<Q", tail, locator_index + 8)[0]
    zip64_eocd = fetch_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
    if not zip64_eocd.startswith(ZIP64_EOCD):
        raise RuntimeError("invalid ZIP64 end-of-central-directory signature")
    central_size = struct.unpack_from("<Q", zip64_eocd, 40)[0]
    central_offset = struct.unpack_from("<Q", zip64_eocd, 48)[0]
    return central_offset, central_size


def find_member(
    central_directory: bytes, requested_name: str
) -> tuple[int, int, int, int, int]:
    cursor = 0
    while cursor + 46 <= len(central_directory):
        if central_directory[cursor : cursor + 4] != CENTRAL_HEADER:
            raise RuntimeError(f"invalid central-directory header at byte {cursor}")
        compression_method = struct.unpack_from("<H", central_directory, cursor + 10)[0]
        expected_crc32 = struct.unpack_from("<I", central_directory, cursor + 16)[0]
        compressed_size = struct.unpack_from("<I", central_directory, cursor + 20)[0]
        uncompressed_size = struct.unpack_from("<I", central_directory, cursor + 24)[0]
        name_size, extra_size, comment_size = struct.unpack_from(
            "<HHH", central_directory, cursor + 28
        )
        disk_start = struct.unpack_from("<H", central_directory, cursor + 34)[0]
        local_offset = struct.unpack_from("<I", central_directory, cursor + 42)[0]
        name_start = cursor + 46
        name_end = name_start + name_size
        extra_end = name_end + extra_size
        filename = central_directory[name_start:name_end].decode("utf-8")
        extra = central_directory[name_end:extra_end]
        uncompressed_size, compressed_size, local_offset = parse_zip64_extra(
            extra,
            uncompressed_size,
            compressed_size,
            local_offset,
            disk_start,
        )
        if filename == requested_name:
            return (
                local_offset,
                compression_method,
                compressed_size,
                uncompressed_size,
                expected_crc32,
            )
        cursor = extra_end + comment_size
    raise KeyError(f"ZIP member not found: {requested_name}")


def extract_member(url: str, member: str, output: pathlib.Path) -> None:
    resolved_url, total_size = resolve_url(url)
    central_offset, central_size = central_directory_location(resolved_url, total_size)
    central_directory = fetch_range(
        resolved_url, central_offset, central_offset + central_size - 1
    )
    (
        local_offset,
        compression_method,
        compressed_size,
        uncompressed_size,
        expected_crc32,
    ) = find_member(central_directory, member)

    local_header = fetch_range(resolved_url, local_offset, local_offset + 29)
    if not local_header.startswith(LOCAL_HEADER):
        raise RuntimeError("invalid local-file header signature")
    name_size, extra_size = struct.unpack_from("<HH", local_header, 26)
    data_offset = local_offset + 30 + name_size + extra_size
    compressed = fetch_range(
        resolved_url, data_offset, data_offset + compressed_size - 1
    )
    if compression_method == 0:
        payload = compressed
    elif compression_method == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported ZIP compression method: {compression_method}")
    if len(payload) != uncompressed_size:
        raise RuntimeError(
            f"uncompressed size mismatch: got {len(payload)}, expected {uncompressed_size}"
        )
    actual_crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    if actual_crc32 != expected_crc32:
        raise RuntimeError(
            f"CRC32 mismatch: got {actual_crc32:08x}, expected {expected_crc32:08x}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary_path = pathlib.Path(temporary.name)
    os.replace(temporary_path, output)
    print(
        f"extracted {member} -> {output} "
        f"({uncompressed_size} bytes, crc32={actual_crc32:08x})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    extract_member(args.url, args.member, args.output)


if __name__ == "__main__":
    main()
