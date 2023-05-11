# ACTC
****
### ACTC: ACtive Threshold Calibration

[![py\_versions](https://img.shields.io/badge/python-3.7%2B-blue)](https://pypi.org/pypi/cleanlab/)

This repository contains code used in our paper: </br>
"ACTC: Active Threshold Calibration for Cold-Start Knowledge Graph Completion"
to be published at ACL 2023 🎉 </br>
by Anastasiia Sedova and Benjamin Roth.

For any questions please [get in touch](mailto:anastasiia.sedova@univie.ac.at).

---

### What is ACTC?

ACTC is a method for estimation the relation threshold for a cold-start knowledge graph completion.
ACTC leverages a limited set of labeled and a large set of unlabeled data in order to calculate per-relation thresholds.
Basing on these thresholds and plausibility scores calculated by a knowledge graph embedding model, one 
can make a decision about whether a new triple should be included to the knowledge graph or not.
Mostly important, it helps to find thresholds in a setting where there is only a limited set of available manual 
annotations.

---

### Usage

`python main.py --path_to_models directory/with/KGE/model/predictions/are/stored 
--output_dir path/to/output/directory --path_to_config path/to/config/file`

An example of a config file: `scripts/configs/config.json`

---
### Citation 

When using our work please cite our ArXiV preprint: 

```
```

---
### Acknowledgements 💎

This research was funded by the WWTF though the project “Knowledge-infused Deep Learning for Natural Language 
Processing” (WWTF Vienna Research Group VRG19-008).






