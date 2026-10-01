This **innovo_pi_logger.py** connects to my INNOVO iP900BP-B "finger pulse oximeter" over Bluetooth (BLE protocol = Bluetooth Low Energy) 
and records the data stream it generates. It was tested on a Raspberry Pi 3, and I would guess should work on 
any other machine as well that has Bluetooth capability and the python libraries.

It records SpO2, Pulse, Respiration Rate, and Perfusion Index (PI%) once per second. 
Note the respiration rate (estimated from HR timing variation) is quite unreliable. 
The SpO2 and Pulse values seem reasonable though. The Bluetooth data also includes the raw plethysmograph signal, 
although it's only 8 bit resolution at 24 samples per second. 
I'm not sure how accurate this signal is, but it looks plausible and sometimes even shows a little bump that seems 
to be the dicrotic notch signal from the aortic valve closure.

The **pulse_waveform_viewer.py** was tested on a Windows laptop.
These programs search for a device named "iP900BPB" and if multiple exist, uses the one with the best signal strength.
This is an unofficial program and I have no connection with the INNOVO company other than having bought a few of their units.
Below output is from a Windows laptop running pulse_waveform_viewer.
![Pulse Graph](InnovoPulseGraph.PNG)

Below output is from a headless Pi logging the data. You can also run this program with the **--rssi** option 
to just show and log the signal strength of the device without actually getting the pulse data from it. It will record the current and minimum-observed signal so you can walk around your house to see how far away you can get and still get reception. In my case, it still works two rooms away.
```
pi@rp4:~/Documents/sleep $ ./innovo_pi_logger.py
================================================================================
Innovo iP900BP-B BLE Logger (Raspberry Pi) v2.4
================================================================================
Scanning for Innovo devices...
Found iP900BPB at ED:9B:47:2E:0F:68 (RSSI: -65 dBm)
Connecting to ED:9B:47:2E:0F:68...
Summary CSV initialized: /home/pi/20260930_230330_pulse.csv
Waveform CSV initialized: /home/pi/20260930_230330_pulse_waveform.csv
Connected!
Discovering characteristics...
Found 4 notify characteristic(s)
================================================================================
Live Measurements (put finger on oximeter, press Ctrl+C to stop)
================================================================================
[  28] SpO2:  97% | Pulse:  72 BPM | Respiration:  9/min | PI:  8.0% | Bad:   3     ^C
Stopping...
Logged 28 summary measurements to /home/pi/20260930_230330_pulse.csv
Bad frames (all zeros): 3
Logged 684 waveform samples to /home/pi/20260930_230330_pulse_waveform.csv
```
