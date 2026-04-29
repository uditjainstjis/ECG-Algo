import time
import argparse
import numpy as np
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Arduino ECG Serial Proxy")
    parser.add_argument('--port', type=str, default=None, help="Serial port (e.g. /dev/cu.usbserial-...)")
    parser.add_argument('--baud', type=int, default=115200, help="Baud rate")
    parser.add_argument('--fs', type=int, default=250, help="Sampling rate")
    parser.add_argument('--outfile', type=str, default="/tmp/ecg_buffer.npy", help="Output file path")
    args = parser.parse_args()

    # Try to auto-detect port if not provided
    port = args.port
    if not port:
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if ports:
                # Prefer anything with "usb" or "tty" or "cu"
                port = ports[0].device
                for p in ports:
                    if "usb" in p.device.lower():
                        port = p.device
                        break
        except Exception:
            pass
            
    if not port:
        logging.error("No serial ports found. Please connect your Arduino.")
        return

    try:
        import serial
    except ImportError:
        logging.error("pyserial is not installed. Run: pip install pyserial")
        return

    logging.info(f"Connecting to Arduino on {port} at {args.baud} baud...")
    
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
        time.sleep(2) # wait for arduino to reset
        ser.flushInput()
        logging.info("Connected! Reading ECG data...")
    except Exception as e:
        logging.error(f"Failed to open port {port}: {e}")
        return

    # Keep a rolling buffer of 30 seconds
    buffer_size = args.fs * 30
    buffer = deque(maxlen=buffer_size)
    
    last_write_time = time.time()
    
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                try:
                    val = float(line)
                    # Convert Arduino 10-bit ADC to mV
                    mv = (val / 1023.0) * 3.3 - 1.65
                    buffer.append(mv)
                except ValueError:
                    pass
                    
            # Write to disk every 0.5 seconds
            current_time = time.time()
            if current_time - last_write_time > 0.5:
                if len(buffer) > 0:
                    np.save(args.outfile, np.array(list(buffer), dtype=np.float64))
                last_write_time = current_time
                
        except KeyboardInterrupt:
            logging.info("Stopping proxy...")
            break
        except Exception as e:
            logging.error(f"Read error: {e}")
            time.sleep(1)
            
    ser.close()

if __name__ == "__main__":
    main()
