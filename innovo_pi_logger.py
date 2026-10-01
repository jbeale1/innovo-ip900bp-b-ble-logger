#!/usr/bin/env python3
"""
Innovo iP900BP-B BLE Logger for Raspberry Pi
Uses Bleak library, with CSV file logging of 1 sps and 24 sps data streams
J.Beale 1-Oct-2026
"""

import asyncio
import csv
import os
import sys
import time
import argparse
from datetime import datetime
from bleak import BleakClient, BleakScanner

VERSION = "2.7"

DEVICE_NAME = "iP900BPB"

# Generate filename with start time: YYYYMMDD_HHMMSS_pulse.csv
def get_csv_filename():
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    return os.path.expanduser(f"~/{timestamp}_pulse.csv")


async def find_innovo_device(timeout=5.0, target_address=None):
    """
    Scan for Innovo iP900BPB devices and return the address with strongest signal.
    If target_address is specified, only return that specific address if found.
    If multiple devices found (and no target_address), returns the one with strongest RSSI.
    Returns the device address or None if not found.
    """
    devices_found = {}

    def detection_callback(device, advertisement_data):
        """Called when a device is discovered"""
        if device.name and DEVICE_NAME in device.name:
            devices_found[device.address] = advertisement_data.rssi

    try:
        async with BleakScanner(detection_callback) as scanner:
            await asyncio.sleep(timeout)
    except Exception as e:
        print(f"Error during device scan: {e}")
        return None

    if not devices_found:
        return None

    # If looking for a specific target address, return it if found
    if target_address:
        if target_address in devices_found:
            rssi = devices_found[target_address]
            print(f"Found {DEVICE_NAME} at {target_address} (RSSI: {rssi} dBm)")
            return target_address
        else:
            return None

    # Return address with strongest RSSI (highest value = least negative = strongest)
    strongest_address = max(devices_found, key=devices_found.get)
    strongest_rssi = devices_found[strongest_address]
    print(f"Found {DEVICE_NAME} at {strongest_address} (RSSI: {strongest_rssi} dBm)")
    return strongest_address


class OximeterLogger:
    """Logs oximeter readings to CSV and displays them"""

    def __init__(self, csv_file, waveform_csv_file, client=None):
        self.csv_file = csv_file
        self.waveform_csv_file = waveform_csv_file
        self.csv_writer = None
        self.csv_handle = None
        self.waveform_csv_writer = None
        self.waveform_csv_handle = None
        self.client = client
        self.spo2 = "--"
        self.pulse = "--"
        self.respiration = "--"
        self.perfusion_index = "--"
        self.packet_count = 0
        self.waveform_sample_count = 0
        self.missing_frames = 0
        self.signal_acquired = False  # Track if we've received first valid reading

        self.init_csv()

    def init_csv(self):
        """Initialize CSV files with headers"""
        # Summary CSV
        self.csv_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow(['timestamp', 'packet_num', 'spo2', 'pulse_bpm', 'respiration_per_min', 'perfusion_index'])
        self.csv_handle.flush()
        print(f"Summary CSV initialized: {self.csv_file}")

        # Waveform CSV
        self.waveform_csv_handle = open(self.waveform_csv_file, 'w', newline='')
        self.waveform_csv_writer = csv.writer(self.waveform_csv_handle)
        self.waveform_csv_writer.writerow(['sample_num', 'waveform_value'])
        self.waveform_csv_handle.flush()
        print(f"Waveform CSV initialized: {self.waveform_csv_file}")

    def update(self, data):
        """Parse and log measurement packet (summary or waveform)"""
        data_bytes = bytes(data)

        # 13-byte summary packet (spo2, pulse, respiration, perfusion index)
        if len(data_bytes) == 13 and data_bytes[0] == 0x3e:
            self.spo2 = data_bytes[1]
            self.pulse = data_bytes[3]
            self.respiration = data_bytes[5]
            self.perfusion_index = data_bytes[11] / 10.0
            self.packet_count += 1

            # Mark that we've received a valid signal (first non-zero reading)
            if not self.signal_acquired and (self.spo2 > 0 or self.pulse > 0 or self.respiration > 0 or self.perfusion_index > 0):
                self.signal_acquired = True

            # Check for all-zero frame (signal dropout or finger removed) - only after initial signal acquired
            if self.signal_acquired and self.spo2 == 0 and self.pulse == 0 and self.respiration == 0 and self.perfusion_index == 0:
                self.missing_frames += 1

            # Write summary to CSV with one decimal place of second precision
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + f'.{now.microsecond // 100000}'
            self.csv_writer.writerow([
                timestamp,
                self.packet_count,
                self.spo2,
                self.pulse,
                self.respiration,
                self.perfusion_index
            ])
            self.csv_handle.flush()

            self.display()
            return True

        # 2-byte waveform packet (24 Hz waveform data)
        elif len(data_bytes) == 2 and data_bytes[0] == 0x01:
            waveform_value = data_bytes[1]
            self.waveform_sample_count += 1

            # Write waveform to CSV
            self.waveform_csv_writer.writerow([
                self.waveform_sample_count,
                waveform_value
            ])
            self.waveform_csv_handle.flush()
            return True

        return False

    def display(self):
        """Show current readings"""
        sys.stdout.write(f"\r[{self.packet_count:4d}] SpO2: {self.spo2:3}% | "
                        f"Pulse: {self.pulse:3} BPM | "
                        f"Respiration: {self.respiration:2}/min | "
                        f"PI: {self.perfusion_index:4.1f}% | Bad: {self.missing_frames:3d}     ")
        sys.stdout.flush()

    def close(self):
        """Close CSV files"""
        if self.csv_handle:
            self.csv_handle.close()
            print(f"\n\nLogged {self.packet_count} summary measurements to {self.csv_file}")
            if self.missing_frames > 0:
                print(f"Bad frames (all zeros): {self.missing_frames}")

        if self.waveform_csv_handle:
            self.waveform_csv_handle.close()
            print(f"Logged {self.waveform_sample_count} waveform samples to {self.waveform_csv_file}")


class RSSILogger:
    """Logs 1-second aggregated RSSI (signal strength) of the Innovo device"""

    SIGNAL_LOSS_MARKER = -200  # Value to indicate no signal received
    MAX_LOSS_LOG_SECONDS = 10  # Log loss for up to 10 seconds

    def __init__(self, rssi_csv_file):
        self.rssi_csv_file = rssi_csv_file
        self.csv_writer = None
        self.csv_handle = None
        self.log_count = 0
        self.rssi_samples = []
        self.last_log_second = None
        self.last_avg_rssi = None
        self.last_min_rssi = None
        self.last_signal_second = None  # Tracks when signal was last received
        self.consecutive_loss_seconds = 0  # Counts consecutive seconds without signal

        # Tracking for overall minimums and loss count (only after initial acquisition)
        self.signal_acquired = False  # Tracks if we've gotten any real signal
        self.min_avg_rssi = None  # Overall minimum avg_rssi (excluding loss markers)
        self.min_min_rssi = None  # Overall minimum min_rssi (excluding loss markers)
        self.total_loss_seconds = 0  # Total count of loss-of-signal seconds

        self.init_csv()

    def init_csv(self):
        """Initialize CSV file with headers"""
        self.csv_handle = open(self.rssi_csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow(['timestamp', 'avg_rssi', 'min_rssi'])
        self.csv_handle.flush()
        print(f"RSSI CSV initialized: {self.rssi_csv_file}")

    def add_rssi_sample(self, rssi):
        """Collect an RSSI sample"""
        now = datetime.now()
        current_second = int(now.timestamp())

        # Initialize on first sample
        if self.last_log_second is None:
            self.last_log_second = current_second

        # Add sample to current window
        self.rssi_samples.append(rssi)

        # Mark that we received a signal
        self.last_signal_second = current_second
        self.consecutive_loss_seconds = 0

    def tick(self):
        """Called every second to log aggregated data and handle signal loss"""
        now = datetime.now()
        current_second = int(now.timestamp())

        # Initialize on first tick
        if self.last_log_second is None:
            self.last_log_second = current_second
            return

        # Check if we've crossed into a new second
        if current_second > self.last_log_second:
            # Determine what to log
            if self.rssi_samples:
                # We have real samples
                avg_rssi = sum(self.rssi_samples) / len(self.rssi_samples)
                min_rssi = min(self.rssi_samples)
                self.consecutive_loss_seconds = 0

                # Mark that we've acquired signal and update overall minimums
                if not self.signal_acquired:
                    self.signal_acquired = True

                if self.min_avg_rssi is None or avg_rssi < self.min_avg_rssi:
                    self.min_avg_rssi = avg_rssi
                if self.min_min_rssi is None or min_rssi < self.min_min_rssi:
                    self.min_min_rssi = min_rssi
            else:
                # No samples this second - check if we should log signal loss
                if self.last_signal_second is not None:
                    seconds_since_signal = current_second - self.last_signal_second
                    if seconds_since_signal <= self.MAX_LOSS_LOG_SECONDS:
                        avg_rssi = self.SIGNAL_LOSS_MARKER
                        min_rssi = self.SIGNAL_LOSS_MARKER
                        self.consecutive_loss_seconds += 1

                        # Count loss seconds only after initial acquisition
                        if self.signal_acquired:
                            self.total_loss_seconds += 1
                    else:
                        # Too long without signal, don't log
                        self.rssi_samples = []
                        self.last_log_second = current_second
                        return
                else:
                    # Haven't received any signal yet
                    self.rssi_samples = []
                    self.last_log_second = current_second
                    return

            # Log with timestamp of the previous second
            log_time = datetime.fromtimestamp(self.last_log_second)
            timestamp = log_time.strftime('%Y-%m-%dT%H:%M:%S.0')

            self.csv_writer.writerow([
                timestamp,
                f"{avg_rssi:.1f}" if avg_rssi != self.SIGNAL_LOSS_MARKER else self.SIGNAL_LOSS_MARKER,
                min_rssi
            ])
            self.csv_handle.flush()

            self.last_avg_rssi = avg_rssi
            self.last_min_rssi = min_rssi
            self.log_count += 1
            self.display()

            # Reset for new second
            self.rssi_samples = []
            self.last_log_second = current_second

    def display(self):
        """Show compact RSSI status: [count] current_avg / current_min  [min_avg/min_min] [loss_count]"""
        if self.last_avg_rssi is not None and self.signal_acquired:
            if self.last_avg_rssi == self.SIGNAL_LOSS_MARKER:
                # Show loss but keep overall stats
                avg_str = "LOSS"
                min_str = "LOSS"
            else:
                avg_str = f"{self.last_avg_rssi:.1f}"
                min_str = f"{self.last_min_rssi}"

            # Format overall minimums (only show if we have them)
            if self.min_avg_rssi is not None:
                min_avg_str = f"{self.min_avg_rssi:.1f}"
                min_min_str = f"{self.min_min_rssi:.1f}"
            else:
                min_avg_str = "--"
                min_min_str = "--"

            status = f"[ {self.log_count:4d}] {avg_str} / {min_str}  [{min_avg_str}/{min_min_str}] [{self.total_loss_seconds}]"
            sys.stdout.write(f"\r{status}     ")
            sys.stdout.flush()

    def close(self):
        """Close CSV file and log any remaining samples"""
        # Log any remaining samples in the current window
        if self.rssi_samples:
            avg_rssi = sum(self.rssi_samples) / len(self.rssi_samples)
            min_rssi = min(self.rssi_samples)

            # Update overall minimums for remaining samples
            if self.min_avg_rssi is None or avg_rssi < self.min_avg_rssi:
                self.min_avg_rssi = avg_rssi
            if self.min_min_rssi is None or min_rssi < self.min_min_rssi:
                self.min_min_rssi = min_rssi

            log_time = datetime.fromtimestamp(self.last_log_second)
            timestamp = log_time.strftime('%Y-%m-%dT%H:%M:%S.0')

            self.csv_writer.writerow([
                timestamp,
                f"{avg_rssi:.1f}",
                min_rssi
            ])
            self.csv_handle.flush()
            self.log_count += 1

        if self.csv_handle:
            self.csv_handle.close()
            print(f"\n\nLogged {self.log_count} seconds of RSSI data to {self.rssi_csv_file}")


async def scan_rssi_only():
    """Scan for RSSI without connecting to the device"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rssi_csv_filename = os.path.expanduser(f"~/{timestamp}_rssi.csv")

    logger = RSSILogger(rssi_csv_filename)

    def detection_callback(device, advertisement_data):
        """Called when a BLE advertisement is detected"""
        if device.name and DEVICE_NAME in device.name:
            rssi = advertisement_data.rssi
            logger.add_rssi_sample(rssi)

    try:
        print("\n" + "="*80)
        print("Scanning for RSSI signals (press Ctrl+C to stop)")
        print("="*80 + "\n")

        async with BleakScanner(detection_callback) as scanner:
            # Keep scanning until interrupted
            while True:
                try:
                    await asyncio.sleep(1)
                    logger.tick()  # Log aggregated data every second
                except KeyboardInterrupt:
                    print("\n\nStopping scan...")
                    break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        logger.close()

    return 0


async def main():
    """Connect and stream live measurements with auto-reconnect on BLE loss"""

    original_device_address = None
    scanning_printed = False  # Track if we've printed the scanning message for this reconnection
    exit_requested = False  # Track if user pressed Ctrl+C

    while True:  # Outer reconnection loop
        # Exit if user pressed Ctrl+C
        if exit_requested:
            break

        try:
            # Find the Innovo device
            if original_device_address:
                if not scanning_printed:
                    print(f"\nScanning for original device {original_device_address}...")
                    scanning_printed = True
                device_address = await find_innovo_device(timeout=5.0, target_address=original_device_address)
                if not device_address:
                    await asyncio.sleep(2.0)
                    continue
            else:
                print("Scanning for Innovo devices...")
                device_address = await find_innovo_device(timeout=5.0)
                if not device_address:
                    print(f"Error: Could not find {DEVICE_NAME} device. Make sure it's powered on and in range.")
                    return 1

            # Remember the original address for reconnection
            if not original_device_address:
                original_device_address = device_address

            # Create new CSV files with current timestamp
            csv_filename = get_csv_filename()
            # Generate waveform CSV filename (same timestamp as summary)
            timestamp = csv_filename.split('/')[-1].split('_pulse')[0]
            waveform_csv_filename = os.path.expanduser(f"~/{timestamp}_pulse_waveform.csv")

            logger = None
            last_data_time = time.time()
            notify_chars = []

            try:
                print(f"Connecting to {device_address}...")
                async with BleakClient(device_address, timeout=10.0) as client:
                    logger = OximeterLogger(csv_filename, waveform_csv_filename, client=client)
                    print(f"Connected!")
                    print("Discovering characteristics...")

                    # Find all notify/indicate characteristics
                    for service in client.services:
                        for char in service.characteristics:
                            if "notify" in char.properties or "indicate" in char.properties:
                                notify_chars.append(str(char.uuid))

                    print(f"Found {len(notify_chars)} notify characteristic(s)")

                    # Create notification handler with timeout tracking
                    def notification_handler(sender, data):
                        nonlocal last_data_time
                        last_data_time = time.time()  # Update last data timestamp
                        logger.update(bytes(data))

                    # Subscribe to all notify characteristics
                    for uuid in notify_chars:
                        try:
                            await client.start_notify(uuid, notification_handler)
                        except Exception as e:
                            pass

                    print("\n" + "="*80)
                    print("Live Measurements (put finger on oximeter, press Ctrl+C to stop)")
                    print("="*80 + "\n")

                    # Monitor for data and connection loss
                    try:
                        while True:
                            await asyncio.sleep(1)  # Check every second for data loss

                            # Check if no data received for 10 seconds
                            if time.time() - last_data_time > 10.0:
                                print(f"\n\nNo data received for 10 seconds - BLE connection lost")
                                break
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        print("\n\nStopping...")
                        exit_requested = True
                        # Don't re-raise - let context manager exit cleanly

                    # Cleanup notifications (always runs)
                    for uuid in notify_chars:
                        try:
                            await client.stop_notify(uuid)
                        except:
                            pass

            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl+C during connection setup
                exit_requested = True
            except Exception as e:
                print(f"Connection error: {e}")
                import traceback
                traceback.print_exc()

            finally:
                if logger:
                    logger.close()

            # Wait before attempting reconnection (silent)
            await asyncio.sleep(2.0)
            # Reset flag so scanning message prints again if needed
            scanning_printed = False

        except (KeyboardInterrupt, asyncio.CancelledError):
            # Ctrl+C during reconnection scan
            exit_requested = True
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(2.0)

    print("\n\nProgram terminated.")
    return 0


if __name__ == "__main__":
    print("="*80)
    print(f"Innovo iP900BP-B BLE Logger (Raspberry Pi) v{VERSION}")
    print("="*80)
    print()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Innovo iP900BP-B BLE Logger")
    parser.add_argument('--rssi', action='store_true', help='Scan for RSSI signal strength without connecting')
    args = parser.parse_args()

    try:
        if args.rssi:
            exit_code = asyncio.run(scan_rssi_only())
        else:
            exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 0

    sys.exit(exit_code)
