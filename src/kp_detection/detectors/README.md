# detectors

## Overview

Detector classes, one per algorithm, each extending `KeyPointDetector`.

## Components

| Component | Description |
| --------- | ----------- |
| [`standard.py`](./standard.py) | `StandardKPDetector` for ORB, SIFT, BRISK, AKAZE and KAZE |
| [`agast.py`](./agast.py) | AGAST detector |
| [`fast.py`](./fast.py) | FAST detector |
| [`mser.py`](./mser.py) | MSER blob detector |
| [`simple_blob.py`](./simple_blob.py) | `cv2.SimpleBlobDetector` wrapper |
| [`harris/`](./harris/README.md) | Harris corner detector and its parameters |
| [`shi_tomashi/`](./shi_tomashi/README.md) | Shi-Tomasi corner detector (`goodFeaturesToTrack`) and its parameters |
