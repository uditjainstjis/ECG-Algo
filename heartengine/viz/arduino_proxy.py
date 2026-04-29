import time
import argparse
import numpy as np
from collections import deque
import logging
import serial
import serial.tools.list_ports
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_port():
    try:
        ports = list(serial.tools.list_ports.comports())
        if not ports: return None
        for p in ports:
            if "usb" in p.device.lower() or "cu." in p.device.lower():
                return p.device
        return ports[0].device
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Arduino ECG Serial Proxy")
    parser.add_argument('--port', type=str, default=None, help="Serial port (e.g. /dev/cu.usbserial-...)")
    parser.add_argument('--baud', type=int, default=115200, help="Baud rate")
    parser.add_argument('--fs', type=int, default=250, help="Sampling rate")
    parser.add_argument('--outfile', type=str, default="/tmp/ecg_buffer.npy", help="Output file path")
    args = parser.parse_args()

    buffer_size = args.fs * 30
    buffer = deque(maxlen=buffer_size)
    last_write_time = time.time()
    
    logging.info("Starting robust Arduino Proxy. Will auto-reconnect if device is unplugged.")
    
    while True:
        port = args.port or find_port()
        if not port:
            logging.warning("No Arduino found. Retrying in 2 seconds...")
            time.sleep(2)
            continue
            
        try:
            ser = serial.Serial(port, args.baud, timeout=1)
            time.sleep(2) # wait for arduino to reset
            ser.flushInput()
            logging.info(f"Connected to Arduino on {port} at {args.baud} baud! Reading data...")
            
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    try:
                        val = float(line)
                        mv = (val / 1023.0) * 3.3 - 1.65
                        buffer.append(mv)
                    except ValueError:
                        pass
                        
                current_time = time.time()
                if current_time - last_write_time > 0.5:
                    if len(buffer) > 0:
                        np.save(args.outfile, np.array(list(buffer), dtype=np.float64))
                    last_write_time = current_time
                    
        except serial.SerialException as e:
            logging.error(f"Serial disconnected: {e}. Reconnecting in 2 seconds...")
            if 'ser' in locals() and ser.is_open:
                try:
                    ser.close()
                except:
                    pass
            time.sleep(2)
        except KeyboardInterrupt:
            logging.info("Stopping proxy...")
            break
        except Exception as e:
            logging.error(f"Read error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
