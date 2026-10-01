#!/usr/bin/env python3
"""
Innovo iP900BP-B BLE Pulse Waveform Viewer
Real-time display of pulse waveform data with PyQtGraph
"""

import asyncio
import csv
import sys
from datetime import datetime
from collections import deque
from bleak import BleakClient
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont

DEVICE_ADDRESS = "D3:67:B7:93:27:00"

class BLEThread(QThread):
    """Runs BLE connection in a separate thread"""
    waveform_data = pyqtSignal(int)
    summary_data = pyqtSignal(dict)
    connection_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        """Run the BLE event loop"""
        asyncio.run(self.connect_and_stream())

    async def connect_and_stream(self):
        """Connect to device and stream data with retry logic"""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.connection_status.emit(f"Connecting (attempt {attempt + 1}/{max_retries})...")
                print(f"Connection attempt {attempt + 1}/{max_retries}...")

                async with BleakClient(DEVICE_ADDRESS, timeout=10.0) as client:
                    self.connection_status.emit("Connected")
                    print(f"Connected to {DEVICE_ADDRESS}")

                    # Find notify characteristics
                    notify_chars = []
                    for service in client.services:
                        for char in service.characteristics:
                            if "notify" in char.properties or "indicate" in char.properties:
                                notify_chars.append(char)

                    print(f"Found {len(notify_chars)} notify characteristics")

                    def make_handler(uuid):
                        def handler(sender, data):
                            data_bytes = bytes(data)

                            # 13-byte summary packet (starts with 0x3e)
                            if len(data_bytes) == 13 and data_bytes[0] == 0x3e:
                                summary = {
                                    'spo2': data_bytes[1],
                                    'pulse': data_bytes[3],
                                    'respiration': data_bytes[5],
                                    'perfusion_index': data_bytes[11] / 10.0,
                                    'timestamp': datetime.now()
                                }
                                self.summary_data.emit(summary)

                            # 2-byte waveform packet
                            elif len(data_bytes) == 2 and data_bytes[0] == 0x01:
                                waveform_value = data_bytes[1]
                                self.waveform_data.emit(waveform_value)

                        return handler

                    # Subscribe to all notify characteristics
                    for char in notify_chars:
                        await client.start_notify(char.uuid, make_handler(char.uuid))

                    print("Listening for waveform data...")

                    # Keep running until interrupted
                    while self.running:
                        await asyncio.sleep(0.1)

                    # Cleanup
                    for char in notify_chars:
                        try:
                            await client.stop_notify(char.uuid)
                        except:
                            pass

                self.connection_status.emit("Disconnected")
                return  # Connection successful, exit retry loop

            except Exception as e:
                error_msg = f"Connection failed (attempt {attempt + 1}/{max_retries}): {e}"
                self.connection_status.emit(error_msg)
                print(error_msg)

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("Failed to connect after all retries")
                    import traceback
                    traceback.print_exc()

    def stop(self):
        """Stop the BLE thread"""
        self.running = False


class OximeterViewer(QMainWindow):
    """PyQtGraph-based pulse waveform viewer"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Innovo Pulse Waveform Viewer")
        self.setGeometry(100, 100, 1200, 650)

        # Data storage
        self.max_points = 600  # Will adjust once sample rate is known
        self.waveform_data = deque(maxlen=self.max_points)
        self.sample_count = 0
        self.sample_rate = None
        self.packets_since_summary = 0

        # Shutdown flag to prevent writing after file is closed
        self.shutting_down = False

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        self.init_csv()

        # Current summary data
        self.current_spo2 = "--"
        self.current_pulse = "--"
        self.current_respiration = "--"
        self.current_pi = "--"

        # Setup UI
        self.setup_ui()

        # Start BLE thread
        self.ble_thread = BLEThread()
        self.ble_thread.waveform_data.connect(self.on_waveform_data)
        self.ble_thread.summary_data.connect(self.on_summary_data)
        self.ble_thread.connection_status.connect(self.on_connection_status)
        self.ble_thread.start()

        # Update plot timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(50)  # Update 20 times per second

    def init_csv(self):
        """Initialize CSV file for waveform logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f"{timestamp}_pulse_waveform.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'sample_num', 'waveform_value'])
        self.csv_file.flush()
        print(f"Waveform CSV initialized: {self.csv_filename}")

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Create plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Waveform Value', units='')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setTitle('Pulse Waveform (Last 10 seconds)')
        self.plot_widget.setYRange(0, 255)
        self.plot_widget.showGrid(True, True, alpha=0.3)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('cyan', width=2))

        layout.addWidget(self.plot_widget, 4)

        # Info panel
        self.info_label = QLabel()
        self.info_label.setFont(QFont('Courier', 11))
        layout.addWidget(self.info_label, 1)

        central_widget.setLayout(layout)
        self.update_info_label()

    def on_waveform_data(self, value):
        """Handle incoming waveform data"""
        if self.shutting_down:
            return

        self.sample_count += 1
        self.packets_since_summary += 1

        # Add to circular buffer
        self.waveform_data.append(value)

        # Log to CSV (check file is still open)
        if self.csv_file and not self.csv_file.closed:
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            self.csv_writer.writerow([timestamp, self.sample_count, value])

    def on_summary_data(self, data):
        """Handle incoming summary data (13-byte packet)"""
        self.current_spo2 = data['spo2']
        self.current_pulse = data['pulse']
        self.current_respiration = data['respiration']
        self.current_pi = data['perfusion_index']

        # Infer sample rate from packet count between summaries
        if self.sample_rate is None and self.packets_since_summary > 0:
            self.sample_rate = self.packets_since_summary
            print(f"Inferred sample rate: {self.sample_rate} Hz")
            # Adjust circular buffer size to hold ~10 seconds
            self.max_points = max(300, self.sample_rate * 10)
            # Create new deque with adjusted size
            old_data = list(self.waveform_data)
            self.waveform_data = deque(maxlen=self.max_points)
            self.waveform_data.extend(old_data)

        self.packets_since_summary = 0
        self.update_info_label()

    def on_connection_status(self, status):
        """Handle connection status changes"""
        self.setWindowTitle(f"Innovo Pulse Waveform Viewer - {status}")

    def update_plot(self):
        """Update the plot display"""
        if len(self.waveform_data) > 1 and self.sample_rate:
            # Create time axis (relative to current, in seconds)
            times = [(i - len(self.waveform_data)) / self.sample_rate for i in range(len(self.waveform_data))]
            self.curve.setData(times, list(self.waveform_data))

    def update_info_label(self):
        """Update the info display"""
        if self.sample_rate:
            rate_str = f"{self.sample_rate} Hz"
        else:
            rate_str = "-- Hz (waiting...)"

        # Format PI value - handle both string and float
        pi_str = f"{self.current_pi:.1f}%" if isinstance(self.current_pi, (int, float)) else self.current_pi

        info_text = (
            f"SpO2: {self.current_spo2}%  |  "
            f"Pulse: {self.current_pulse} BPM  |  "
            f"Respiration: {self.current_respiration}/min  |  "
            f"PI: {pi_str}  |  "
            f"Samples: {self.sample_count:6d}  |  "
            f"Rate: {rate_str}  |  "
            f"File: {self.csv_filename}"
        )
        self.info_label.setText(info_text)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts: ESC or Q to exit"""
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Cleanup on close"""
        # Set shutdown flag to prevent further writes
        self.shutting_down = True

        # Stop timers and threads
        self.update_timer.stop()
        self.ble_thread.stop()
        self.ble_thread.wait()

        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            print(f"Logged {self.sample_count} waveform samples to {self.csv_filename}")

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = OximeterViewer()
    viewer.show()
    sys.exit(app.exec_())
