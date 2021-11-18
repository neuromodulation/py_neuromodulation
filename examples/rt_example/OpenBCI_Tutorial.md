
Tutorial OpenBCI LSL integration 
1. Install mne real-time environment 

2. Install OpenBCI and install necessary FTDI driver 

3. Install openBCI labstreamlayer: https://docs.openbci.com/docs/06Software/02-CompatibleThirdPartySoftware/LSL 

4. run in a terminal from the OpenBCI_LSL repo 

> $ python openbci_lsl.py --stream
> 2021-03-19 14:24:27.599 (   6.241s) [                ]             common.cpp:50    INFO| git:e762b9e10ad0d77651923c0afa7f435af38e8a9b/branch:refs/tags/v1.14.0/build:Release/compiler:MSVC-19.0.24245.0/link:shared
> 2021-03-19 14:24:27.607 (   6.249s) [                ] stream_outlet_impl.cpp:86    WARN| Couldn't create multicast responder for 224.0.0.1 (set_option: A socket operation was attempted to an unreachable host)
> 2021-03-19 14:24:27.612 (   6.255s) [                ] stream_outlet_impl.cpp:86    WARN| Couldn't create multicast responder for 224.0.0.1 (set_option: A socket operation was attempted to an unreachable host)
> 
> -------INSTANTIATING BOARD-------
> Connecting to V3 at port COM3
> Serial established...
> OpenBCI V3 8-16 channel
> On Board ADS1299 Device ID: 0x3E
> LIS3DH Device ID: 0x33
> Firmware: v3.1.2
> $$$
> --------------------------------------
> LSL Configuration:
>   Stream 1:
>       Name: openbci_eeg
>       Type: EEG
>       Channel Count: 8
>       Sampling Rate: 250.0
>       Channel Format: float32
>       Source Id: openbci_eeg_id127
>   Stream 2:
>       Name: openbci_aux
>       Type: AUX
>       Channel Count: 3
>       Sampling Rate: 250.0
>       Channel Format: float32
>       Source Id: openbci_aux_id127
> 
> Electrode Location Montage:
> ['Fp1', 'Fp2', 'C3', 'C4', 'T5', 'T6', 'O1', 'O2']
> ---------------------------------------
> 
> --------------INFO---------------
> Commands:
>     Type "/start" to stream to LSL
>     Type "/stop" to stop stream.
>     Type "/exit" to disconnect the board.
> Advanced command map available at http://docs.openbci.com
> 
> -------------BEGIN---------------

5. start streaming with /run 
6. open examples/openBCI_test.py and edit the host_name as the only from the pylsl “Source Id” (in the upper example openbci_eeg_id127 for Stream 1) 

