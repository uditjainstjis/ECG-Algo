#!/usr/bin/env python3
"""
Arduino ECG Serial Proxy — Robust Binary + ASCII Reader

The Arduino EXG device can send data in two formats:
  1. Binary: 10-byte records (2-byte int16 ECG + 8-byte int64 timestamp)
  2. ASCII:  One integer per line (e.g. "512\n")

This proxy auto-detects the format, converts to millivolts, and writes
a rolling 60-second numpy buffer to /tmp/ecg_buffer.npy every 200ms.
The Streamlit dashboard reads this file for live visualization.

Usage:
    python arduino_proxy.py                     # auto-detect port
    python arduino_proxy.py --port /dev/cu.usbmodem1101
"""

import time
import struct
import argparse
import numpy as np
from collections import deque
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ecg-proxy")

RECORD_SIZE = 10  # 2 bytes ECG + 8 bytes timestamp


def find_arduino_port():
    """Auto-detect the Arduino serial port."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            dev = p.device.lower()
            if "usbmodem" in dev or "usbserial" in dev or "arduino" in p.description.lower():
                return p.device
        # Fallback: pick any /dev/cu.usb*
        for p in ports:
            if "usb" in p.device.lower():
                return p.device
    except Exception:
        pass
    return None


def detect_format(ser, timeout=3.0):
    """Read a few bytes to determine if the Arduino sends binary or ASCII."""
    ser.timeout = timeout
    chunk = ser.read(40)  # Read enough for ~4 binary records or a few ASCII lines
    if not chunk:
        return "unknown", chunk

    # If we can decode it as clean ASCII with digits, it's text mode
    try:
        text = chunk.decode("ascii")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        nums = [float(l) for l in lines if l.replace(".", "").replace("-", "").isdigit()]
        if len(nums) >= 1:
            logger.info(f"Detected ASCII format (sample values: {nums[:3]})")
            return "ascii", chunk
    except (UnicodeDecodeError, ValueError):
        pass

    # Check if it looks like valid binary records
    if len(chunk) >= RECORD_SIZE:
        ecg_val = struct.unpack_from("<h", chunk, 0)[0]
        if -5000 < ecg_val < 5000:  # Reasonable ADC range
            logger.info(f"Detected BINARY format (first ECG sample: {ecg_val})")
            return "binary", chunk

    return "binary", chunk  # Default to binary


def parse_binary_chunk(data):
    """Parse binary data into ECG samples (millivolts)."""
    samples = []
    n_records = len(data) // RECORD_SIZE
    for i in range(n_records):
        offset = i * RECORD_SIZE
        ecg_raw = struct.unpack_from("<h", data, offset)[0]
        # Convert to millivolts: assume 10-bit ADC at 3.3V ref
        mv = (ecg_raw / 1023.0) * 3.3 - 1.65
        samples.append(mv)
    return samples, len(data) % RECORD_SIZE  # return leftover bytes count


def main():
    parser = argparse.ArgumentParser(description="Arduino ECG Serial Proxy (Binary + ASCII)")
    parser.add_argument("--port", type=str, default=None, help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--fs", type=int, default=250, help="Sampling rate (Hz)")
    parser.add_argument("--outfile", type=str, default="/tmp/ecg_buffer.npy", help="Buffer file")
    args = parser.parse_args()

    buffer_size = args.fs * 60  # 60 seconds rolling buffer
    buffer = deque(maxlen=buffer_size)
    sample_count = 0

    logger.info("=" * 50)
    logger.info("  HeartEngine Arduino ECG Proxy")
    logger.info("  Auto-reconnect enabled. Ctrl+C to stop.")
    logger.info("=" * 50)

    while True:
        # --- Find port ---
        port = args.port or find_arduino_port()
        if not port:
            logger.warning("No Arduino detected. Plug in your EXG device... (retrying in 2s)")
            # Write empty buffer so frontend knows we're offline
            np.save(args.outfile, np.array([], dtype=np.float64))
            time.sleep(2)
            continue

        # --- Connect ---
        try:
            import serial
            ser = serial.Serial(port, args.baud, timeout=1)
            time.sleep(2)  # Arduino reset delay
            ser.reset_input_buffer()
            logger.info(f"✅ Connected to {port} @ {args.baud} baud")
        except Exception as e:
            logger.error(f"Failed to open {port}: {e}. Retrying in 2s...")
            time.sleep(2)
            continue

        # --- Detect format ---
        fmt, leftover = detect_format(ser)
        logger.info(f"Format: {fmt} | Starting data capture...")
        ser.timeout = 0.1  # Fast non-blocking reads from now on

        binary_remainder = leftover if fmt == "binary" else b""
        last_flush = time.time()
        last_log = time.time()

        # Process leftover from detection
        if fmt == "binary" and len(leftover) >= RECORD_SIZE:
            samples, remainder_len = parse_binary_chunk(leftover)
            buffer.extend(samples)
            sample_count += len(samples)
            binary_remainder = leftover[-(remainder_len):] if remainder_len > 0 else b""

        # --- Main read loop ---
        try:
            while True:
                if fmt == "binary":
                    chunk = ser.read(RECORD_SIZE * 25)  # Read up to 25 samples at a time
                    if chunk:
                        data = binary_remainder + chunk
                        samples, remainder_len = parse_binary_chunk(data)
                        buffer.extend(samples)
                        sample_count += len(samples)
                        binary_remainder = data[-(remainder_len):] if remainder_len > 0 else b""
                else:
                    # ASCII mode
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        try:
                            val = float(line)
                            mv = (val / 1023.0) * 3.3 - 1.65
                            buffer.append(mv)
                            sample_count += 1
                        except ValueError:
                            pass

                # Flush to disk every 200ms
                now = time.time()
                if now - last_flush > 0.2 and len(buffer) > 0:
                    np.save(args.outfile, np.array(buffer, dtype=np.float64))
                    last_flush = now

                # Log status every 5 seconds
                if now - last_log > 5.0:
                    logger.info(f"📊 {sample_count} total samples | buffer: {len(buffer)} | ~{len(buffer)/args.fs:.1f}s")
                    last_log = now

        except serial.SerialException as e:
            logger.error(f"🔌 Device disconnected: {e}")
            logger.info("Waiting for reconnect...")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Stopping proxy. Goodbye!")
            try:
                ser.close()
            except Exception:
                pass
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(1)


if __name__ == "__main__":
    main()
