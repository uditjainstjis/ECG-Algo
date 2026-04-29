"""
Arduino ECG Serial Reader
===========================
Reads live ECG data from Arduino (AD8232 or similar) over USB serial.
Supports auto-detection of serial ports and configurable baud rates.
"""

import time
import numpy as np
import threading
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ArduinoECGReader:
    """Thread-safe circular buffer reader for Arduino ECG serial data."""

    def __init__(self, port: str = None, baud: int = 115200, buffer_sec: float = 30.0, fs: int = 250):
        self.port = port
        self.baud = baud
        self.fs = fs
        self.buffer_size = int(buffer_sec * fs)
        self.buffer = deque(maxlen=self.buffer_size)
        self._running = False
        self._thread = None
        self._serial = None
        self._connected = False

    @staticmethod
    def list_ports():
        """List available serial ports."""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            return [(p.device, p.description) for p in ports]
        except Exception:
            return []

    def connect(self) -> bool:
        """Open serial connection."""
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Arduino reset delay
            self._serial.flushInput()
            self._connected = True
            logger.info(f"Connected to {self.port} @ {self.baud} baud")
            return True
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            self._connected = False
            return False

    def start(self):
        """Start background reading thread."""
        if not self._connected:
            if not self.connect():
                return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop reading."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False

    def _read_loop(self):
        """Background serial reading loop."""
        while self._running and self._serial and self._serial.is_open:
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    try:
                        value = float(line)
                        # Normalize Arduino ADC (0-1023) to millivolts
                        mv = (value / 1023.0) * 3.3 - 1.65
                        self.buffer.append(mv)
                    except ValueError:
                        pass  # Skip non-numeric lines
            except Exception:
                time.sleep(0.01)

    def get_buffer(self) -> np.ndarray:
        """Get current buffer as numpy array."""
        return np.array(list(self.buffer), dtype=np.float64)

    @property
    def is_connected(self) -> bool:
        return self._connected and self._running

    @property
    def samples_collected(self) -> int:
        return len(self.buffer)


class SimulatedECGReader:
    """Simulated live ECG for demo when no hardware is connected."""

    def __init__(self, fs: int = 250, hr_bpm: float = 72):
        self.fs = fs
        self.hr_bpm = hr_bpm
        self.buffer = deque(maxlen=fs * 30)
        self._running = False
        self._thread = None
        self._start_time = 0

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False

    def _generate_loop(self):
        rr = 60.0 / self.hr_bpm
        sample_interval = 1.0 / self.fs
        while self._running:
            t = time.time() - self._start_time
            phase = (t % rr) / rr
            # Synthetic PQRST morphology
            val = 0.0
            if 0.10 < phase < 0.18:  # P wave
                val = 0.15 * np.sin((phase - 0.10) / 0.08 * np.pi)
            elif 0.22 < phase < 0.24:  # Q
                val = -0.1 * np.sin((phase - 0.22) / 0.02 * np.pi)
            elif 0.24 < phase < 0.28:  # R
                val = 1.0 * np.sin((phase - 0.24) / 0.04 * np.pi)
            elif 0.28 < phase < 0.32:  # S
                val = -0.2 * np.sin((phase - 0.28) / 0.04 * np.pi)
            elif 0.45 < phase < 0.60:  # T wave
                val = 0.3 * np.sin((phase - 0.45) / 0.15 * np.pi)
            val += np.random.normal(0, 0.02)
            self.buffer.append(val)
            time.sleep(sample_interval)

    def get_buffer(self) -> np.ndarray:
        return np.array(list(self.buffer), dtype=np.float64)

    @property
    def is_connected(self):
        return self._running

    @property
    def samples_collected(self):
        return len(self.buffer)
