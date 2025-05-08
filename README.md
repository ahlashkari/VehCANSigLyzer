![](https://github.com/ahlashkari/CANSigLyzer/blob/main/bccc.jpg)

# Vehicle Controller Area Network (CAN) Signal Analyzer (VehCANSigLyzer)

Fabrication attacks involve injecting fake CAN frames onto the CAN bus, disrupting normal message timing and causing ECUs (Electronic Control Units) to misinterpret signal data. This analyzer extracts two categories of raw CAN traffic features to detect such behavior: timing-based and signal-level.

##Timing-Based Features
This analyzer derives two useful timing-related features from the raw timestamp column:

###time_interval – The time difference between each CAN frame and the previous frame (regardless of AID).

###aid_time_interval – The time difference between a frame and the previous frame with the same arbitration ID (AID).

These timing features capture the disruptions introduced by injected messages during an attack.

##Signal Extraction from CAN Frames
Signals encoded in each frame's data_field were decoded using the cantools Python library. This analyzer used a DBC file—specifically, the hyundai_kia_generic.dbc from the OpenDBC project. Although the exact vehicle model in the CAN-MIRGU dataset is unknown, this DBC file matches both the AIDs and their associated functions as described in the original paper.

We extracted 545 distinct signals from the decoded frames, each associated with a specific AID. Signal column names are prefixed to avoid collisions with their corresponding AID (e.g., 289_engine_speed).

##Final Feature Set
Our final feature matrix includes:
### arbitration_id (converted to decimal),
### The two timing-based features,
### 500+ decoded signal features.



# Project Team members 

* [**Arash Habibi Lashkari:**](http://ahlashkari.com/index.asp) Founder and supervisor

* [**Shaila Sharmin:**](https://github.com/ohoaha) Graduate student, researcher, and developer - York University (6 months, 2024 - 2025)




# Acknowledgment

This project was made possible through funding from the Mitacs Globalink Research Award (GRA) to Shaila Sharmin, who is under the supervision of Prof. Arash Habibi Lashkari at York University in Canada.
