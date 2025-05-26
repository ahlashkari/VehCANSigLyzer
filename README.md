![](https://github.com/ahlashkari/CANSigLyzer/blob/main/bccc.jpg)

# Vehicle Controller Area Network (CAN) Signal Analyzer (VehCANSigLyzer)

Fabrication attacks involve the injection of fake CAN frames onto the CAN bus, which disrupts the normal timing of legitimate frames and provides Electronic Control Units (ECUs) with incorrect signal data. This analyzer extracts two categories of raw CAN traffic features to detect such behavior: timing-based and signal-level features.

While this analyzer has been used with the [CAN-MIRGU dataset](https://www.ndss-symposium.org/ndss-paper/auto-draft-482/), it may be used to extract timing-based and signal-level features from any dataset that is similarly formatted and for which a CAN database file (.DBC) is available. DBCs for many vehicle models are available at [`opendbc`](https://github.com/commaai/opendbc), which is an open-source repository of reverse-engineered DBC files.  

## Timing-Based Features
This analyzer derives two useful timing-related features from the raw timestamp column:

* `time_interval` - The time difference between each CAN frame and the previous frame (regardless of arbitration identifier (AID))
* `aid_time_interval` - The time difference between a frame and the previous frame with the same arbitration ID (AID)

These timing features capture the disruptions introduced by injected messages during a fabrication attack.

## Signal-Based Features
Signals encoded in each frame's `data_field` were decoded using the `cantools` Python library. Extracting signal features requires the correct DBC for the source vehicle. 

For the CAN-MIRGU dataset, the `hyundai_kia_generic.dbc` file was used from the `opendbc` project. Although the exact vehicle model in the CAN-MIRGU dataset is unknown, this DBC file matches both the AIDs and their associated functions as described in the original paper.

We extracted 545 distinct signals from the decoded frames, each associated with a specific AID. Signal column names are prefixed with the corresponding AID to avoid collisions with other similarly named signals (e.g., `2B0.SAS_Speed`).

## Final Feature Set

Our final feature matrix includes:
* `arbitration_id` (converted to decimal),
* Two timing-based features, `time_interval` and `aid_time_interval`
* 500+ decoded signal features


# Project Team members 

* [**Arash Habibi Lashkari:**](http://ahlashkari.com/index.asp) Founder and supervisor

* [**Shaila Sharmin:**](https://github.com/ohoaha) Graduate student, researcher, and developer - York University (6 months, 2024 - 2025)


# Acknowledgment

This project was made possible through funding from the Mitacs Globalink Research Award (GRA) to Shaila Sharmin, who is under the supervision of Prof. Arash Habibi Lashkari at York University in Canada.
