#!/usr/bin/env python3
"""
BLE Device Scanner - reports all discoverable BLE devices and their descriptive data
"""

import asyncio
from bleak import BleakScanner

async def scan_devices():
    """Scan for all BLE devices and report detailed information"""

    print("="*80)
    print("Scanning for BLE devices (10 seconds)...")
    print("="*80 + "\n")

    # Use callback-based scanner to capture RSSI and advertisement data
    devices_dict = {}

    def detection_callback(device, advertisement_data):
        """Called when a device is discovered"""
        if device.address not in devices_dict:
            devices_dict[device.address] = {
                'device': device,
                'advertisement_data': advertisement_data,
                'rssi': advertisement_data.rssi
            }
        else:
            # Update with newer RSSI if available
            devices_dict[device.address]['rssi'] = advertisement_data.rssi

    async with BleakScanner(detection_callback) as scanner:
        await asyncio.sleep(10.0)

    if not devices_dict:
        print("No BLE devices found.")
        return

    print(f"Found {len(devices_dict)} device(s):\n")

    for i, (address, info) in enumerate(sorted(devices_dict.items()), 1):
        device = info['device']
        advertisement_data = info['advertisement_data']
        rssi = info['rssi']

        print(f"{i}. Device: {device.name if device.name else '(no name)'}")
        print(f"   Address:  {device.address}")
        print(f"   RSSI:     {rssi} dBm")

        # Service UUIDs from advertisement
        if advertisement_data.service_uuids:
            print(f"   Service UUIDs: {advertisement_data.service_uuids}")

        # Local name
        if advertisement_data.local_name:
            print(f"   Local Name: {advertisement_data.local_name}")

        # Manufacturer data
        if advertisement_data.manufacturer_data:
            print(f"   Manufacturer Data: {advertisement_data.manufacturer_data}")

        # Service data
        if advertisement_data.service_data:
            print(f"   Service Data: {advertisement_data.service_data}")

        # TX Power
        if advertisement_data.tx_power is not None:
            print(f"   TX Power: {advertisement_data.tx_power} dBm")

        print()

async def main():
    try:
        await scan_devices()
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
