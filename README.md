# INST0062 Dissertation Project — IGNITE
## Project Overview
This repository contains the source code for the **`INST0062 Dissertation Project`**, supporting full reproducibility of results and demonstrating the intellectual contribution of the work.

**IGNITE — Integrated Geospatial Navigation and Inference for Thermal Events —** is a **zero‑shot**, **post‑hoc** **augmentation** framework designed for **CNN‑based computer‑vision fire detectors**. The system enriches conventional object‑level detection with **affordance‑driven semantic reasoning**, mapping the functional **relationships** between **scene objects** and **fire** into **logic predicates**. By comparing these predicates against stored human‑lingual knowledge, IGNITE **suppresses** alarms for **benign**, **contextually appropriate fires** while **preserving** sensitivity to **hazardous ones**.

------
**Disclaimer**
The theory, design, and implementation of this project were developed independently by **UCL student SN22086919**, under the supervision of **Dr. Daniel Onah**. Although the framework demonstrates promising potential for intelligent fire‑safety applications, it is currently an academic research prototype.
It **must not** be used in real‑world safety‑critical deployments.


## Project Structure
**IGNITE**
├── **Dataset**
│   ├── **HFD**:Original [Home Fire Dataset(HFD)](https://www.kaggle.com/datasets/pengbo00/home-fire-dataset)
│   ├── **HFDobj**: Self-annotated Repurposed HFD
│   ├── **KAD**: Source of HFDobj Annotations[(KAD)](https://universe.roboflow.com/cdd-workspace-oo6dk/kitchen-rtzfe)
│   └── **evaluation_slices**: Human Safety Oracle
├── **IO_Module**: Front-end Modules that handles the input and output of the system
│   ├── `boundingBoxDrawer`: Renders Bounding Boxes 
│   ├── `cameraList`: Lists All FFMPEG supported Camera 
│   ├── `cameraSetting`: Gets/Sets Camera Settings
│   ├── `logger`: Generates live Logs and Final Reports
│   └── `videoCapture`: Captures Video from Camera, and funnels frames to back-end
├── **README**
├── **Service**
│   ├── **ObjectDetector**
│   │   ├── *`fire.pt`*: Fine-Tuned Fire Detection Pytorch Weightings
│   │   ├── *`obj1.pt`*: Fine-Tuned Object Detection Pytorch Weightings
│   │   └── `yolo`: Wraps around [YOLO26](https://github.com/ultralytics/yolo26), Handles Related Functionalities.
│   ├── **Predicate**
│   │   ├── `affordanceEmbedder`: Embeds Geospatial Relationships into Affordance Predicates. Uses global variable from `triplet2natural` to further map it to natural language.
│   │   └── `triplet2natural`: Contains the mapping of Systematic Predicates to Natural Language.
│   └── **Semantic**
│       ├── `common_knowledge`: Contains Common Knowledge Base for the system
│       └── `continuousLatentInferencer`:Wraps around [BAAI/BGE](https://huggingface.co/BAAI/bge-small-en-v1.5). Handles Vector Space Search and other Functionalities.
├── `app`: Main Application File. Integrates Modules from `IO_Module`, `Service`.
├── **logs**: **(Optional)**, Contains Logs from the System. Auto Generated in the first run.
├── **reports**: **(Optional)**, Contains Reports from the System. Auto Generated in the first run.
├── **requirement.txt**: Contains all the required packages for the system.
└── `test`: Contains Testing Scripts. Use for Reproduction.


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
   - To run under `Streaming` regime, set `streaming` to `True`.
   - To run under `Static` regime, set `streaming` to `False`, pass the path to **ONE** JPG file in via `img_path=<path to image>` and provide the path.
2. Follow the instructions on the console.
3. To Exit the `Streaming` Regime, press `ESC`. `Static` Regime will automatically exit after Inferencing is done.


## Usage(Reproduction of Experiment)
1. Execute `test.py` to run the test. 
2. Test will be conducted automatically.
3. Test results will be saved in **reports** and **logs** folder as the date of experiment.