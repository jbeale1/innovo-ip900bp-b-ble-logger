This **innovo_pi_logger.py** connects to my INNOVO iP900BP-B "finger pulse oximeter" over Bluetooth (BLE protocol = Bluetooth Low Energy) 
and records the data stream it generates. It was tested on a Raspberry Pi 3, and I would guess should work on 
any other machine as well that has Bluetooth capability and the python libraries.

It records SpO2, Pulse, Respiration Rate, and Perfusion Index (PI%) once per second. 
Note the respiration rate (estimated from HR timing variation) is quite unreliable. 
The SpO2 and Pulse values seem reasonable though. The Bluetooth data also includes the raw plethysmograph signal, 
although it's only 8 bit resolution at 24 samples per second. 
I'm not sure how accurate this signal is, but it looks plausible and sometimes even shows a little bump that seems 
to be the dicrotic notch signal from the aortic valve closure.

To use it with your own device you would need to change the DEVICE_ADDRESS to match your particular unit.
This is an unofficial program and I have no connection with the INNOVO company other than having bought a few of their units.

