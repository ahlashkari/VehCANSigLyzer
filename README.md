![](https://github.com/ahlashkari/CANSigLyzer/blob/main/bccc.jpg)

# Vehicular Controller Area Network (CAN) Signal Analyzer (VehCANSigLyzer)

Fabrication attacks involve the injection of fake CAN frames onto the CAN bus, which disrupts the normal timing of legitimate frames and provides Electronic Control Units (ECUs) with incorrect signal data. This analyzer extracts two categories of raw CAN traffic features to detect such behavior: timing-based and signal-level features.

While this analyzer has been used with the [HCRL Attack & Defense Challenge dataset](https://www.ndss-symposium.org/wp-content/uploads/autosec2021_23035_paper.pdf) (HCRL A&D), it may be used to extract timing-based and signal-level features from any dataset that is similarly formatted and for which a CAN database file (.DBC) is available. DBCs for many vehicle models are available at [`opendbc`](https://github.com/commaai/opendbc), which is an open-source repository of reverse-engineered DBC files.  

### Timing-Based Features
This analyzer derives two useful timing-related features from the raw timestamp column:

* `time_interval` - The time difference between each CAN frame and the previous frame (regardless of arbitration identifier (AID))
* `aid_time_interval` - The time difference between a frame and the previous frame with the same arbitration ID (AID)

These timing features capture the disruptions introduced by injected messages during a fabrication attack.

### Signal-Based Features
Signals encoded in each frame's `data_field` were decoded using the `cantools` Python library. Extracting signal features requires the correct DBC for the source vehicle. 

For the HCRL A&D dataset, the `hyundai_kia_generic.dbc` file was used from the `opendbc` project.

We extracted 660 distinct signals from the decoded frames, each associated with a specific AID. Signal column names are prefixed with the corresponding AID to avoid collisions with other similarly named signals (e.g., `386.WHL_SPD_RR`).

### Final Feature Set

Our final feature matrix includes:
* `arbitration_id` (converted to decimal),
* Two timing-based features, `time_interval` and `aid_time_interval`
* 600+ decoded signal features

## Usage 

Before using VehCANSigLyzer, install the required packages listed in `requirements.txt` using 
```bash
pip install -r requirements.txt
``` 
VehCANSigLyzer was developed and tested using Python 3.10.0, but other versions may be compatible as well. 

To use it with the HCRL A&D dataset, download the dataset to a folder named `hcrl` in the root folder. 

## Copyright (c) 2025

For citation in your works and also understanding VehCANSigLyzer completely, you can find below published papers:

- CAN-BiGRUBERT: Unveiling Automotive Vehicle Intruders by Profiling and Characterizing Anomalies in Controller Area Network Shaila Sharmin, Arash Habibi Lashkari, Hafizah Mansor and Andi Fitriah Abdul Kadir, Computer Networks, Vol. 276, 2025 

## Project Team members 

* [**Arash Habibi Lashkari:**](http://ahlashkari.com/index.asp) Founder and supervisor

* [**Shaila Sharmin:**](https://github.com/ohoaha) Graduate student, researcher, and developer - York University (6 months, 2024 - 2025)


## Acknowledgment

This project was made possible through funding from the Mitacs Globalink Research Award (GRA) to Shaila Sharmin, who is under the supervision of Prof. Arash Habibi Lashkari at York University in Canada.
