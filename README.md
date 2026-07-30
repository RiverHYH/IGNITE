# INST0062 Dissertation Project — IGNITE
## Project Overview
This repository contains the source code for the **`INST0062 Dissertation Project`**, supporting full reproducibility of results and demonstrating the intellectual contribution of the work.

**IGNITE — Integrated Geospatial Navigation and Inference for Thermal Events —** is a **zero‑shot**, **post‑hoc** **augmentation** framework designed for **CNN‑based computer‑vision fire detectors**. The system enriches conventional object‑level detection with **affordance‑driven semantic reasoning**, mapping the functional **relationships** between **scene objects** and **fire** into **logic predicates**. By comparing these predicates against stored human‑lingual knowledge, IGNITE **suppresses** alarms for **benign**, **contextually appropriate fires** while **preserving** sensitivity to **hazardous ones**.

------
**Disclaimer**
The theory, design, and implementation of this project were developed independently by **UCL student SN22086919**, under the supervision of **Dr. Daniel Onah**. Although the framework demonstrates promising potential for intelligent fire‑safety applications, it is currently an academic research prototype.
It **must not** be used in real‑world safety‑critical deployments.


## Project Structure
```text
IGNITE
├── Dataset
│   ├── HFD                     # Original Home Fire Dataset (Kaggle)
│   ├── HFDobj                  # Self‑annotated repurposed HFD
│   ├── KAD                     # Source of HFDobj annotations (Roboflow)
│   └── evaluation_slices       # Human Safety Oracle
│
├── IO_Module                   # Front‑end input/output modules
│   ├── boundingBoxDrawer       # Renders bounding boxes
│   ├── cameraList              # Lists FFmpeg‑supported cameras
│   ├── cameraSetting           # Gets/Sets camera settings
│   ├── logger                  # Generates live logs and final reports
│   └── videoCapture            # Captures video and streams frames to back‑end
│
├── README
│
├── Service
│   ├── ObjectDetector
│   │   ├── fire.pt             # Fine‑tuned fire detection weights
│   │   ├── obj1.pt             # Fine‑tuned object detection weights
│   │   └── yolo                # Wrapper around YOLO26
│   │
│   ├── Predicate
│   │   ├── affordanceEmbedder  # Embeds geospatial relationships into predicates
│   │   └── triplet2natural     # Maps systematic predicates to natural language
│   │
│   └── Semantic
│       ├── common_knowledge    # Common knowledge base
│       └── continuousLatentInferencer
│                               # Wrapper around BAAI/BGE for vector search
│
├── app                         # Main application integrating IO_Module + Service
│
├── logs                        # Auto‑generated logs (optional)
│
├── reports                     # Auto‑generated reports (optional)
│
├── requirement.txt             # Python dependencies
│
└── test                        # Testing scripts for reproduction
```

## Prerequisites
### Hardware
The project is implemented and tested on a **Windows 11** machine with **Python 3.11**. The Hardware used is as follows:
- **CPU**: Intel Core i7-12700H
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop
- **VRAM**: 4GB
- **RAM**: 16GB DDR5

Therefore, in order to reproduce testing results, the hardware should be equivalent or above the specifications.

### Software and Packages
The following packages are required to run the project:
-  **FFMPEG**. FFMPEG is needed to enable the `Streaming` Regime. Please make sure your device is equipped with FFMPEG or install it from [here](https://www.ffmpeg.org/download.html).
-  All Python Packages listed in `requirement.txt`. Please install them using `pip install -r requirement.txt`.

## Usage(Demo)
1. Execute `app.py` to run the demo. 
   - To run under the `Streaming` regime, set `streaming=True`.
   - To run under the `Static` regime, set `streaming=False`, pass the path to **ONE** JPG file via `img_path=<path to image>`, and provide the path.
2. Follow the instructions on the console.
3. To exit the `Streaming` Regime, press `ESC`. `Static` Regime will automatically exit after Inferencing is done.


## Usage(Reproduction of Experiment)
1. Execute `test.py` to run the test. 
2. The test will be conducted automatically.
3. Test results will be saved in **reports** and **logs** folder as the date of experiment.
