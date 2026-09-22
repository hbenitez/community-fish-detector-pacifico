# Community Fish Detector (CFD)

This repository provides pretrained object detection models for identifying one class: “fish”.

The model was trained on the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset), a collaboratively built, large-scale dataset that unifies >1.9 million images and >935,000 fish bounding boxes from 17 open datasets spanning freshwater, marine, and lab environments.

With this project, our goal is to detect any fish, anywhere. 

These models represent an initial training effort. They perform reasonably well across a variety of environments but can certainly be improved. If you’d like to contribute improvements or new experiments, [please get in touch](mailto:fppvrn@gmail.com)!


## Table of Contents

1. [Models](#models)  
2. [Quick start](#quick-start)  
3. [Eastern Pacific fine-tuning study](#eastern-pacific-fine-tuning-study)  
4. [Contributors](#contributors)
5. [Example predictions](#example-predictions)  
6. [Also see](#also-see)  


## Models

| Model | Architecture | Input image size | Target classes | Dataset | Inference code license |
|--|--|--|--|--|--|
| [community-fish-detector-2026.02.02-rf-detr-nano-640.pth](https://github.com/filippovarini/community-fish-detector/releases/download/cfd-2026.02.02-rf-detr-nano/community-fish-detector-2026.02.02-rf-detr-nano-640.pth) | [RF-DETR Nano](https://rfdetr.roboflow.com/reference/nano/) | 640 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | Apache |
| [cfd-yolov12x-1.00.pt](https://github.com/WildHackers/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt) | [YOLOv12x](https://docs.ultralytics.com/models/yolo12/) | 1024 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | AGPL |

This table only describes license information for the training and inference code.  The [training data](https://lila.science/datasets/community-fish-detection-dataset) is a composite of multiple datasets with a variety of licenses.

## Quick start

These instructions describe the process for running the RF-DETR version of the model.  The YOLOv12x version is deprecated, but you can see instructions for running it in an [older version of this README](https://github.com/filippovarini/community-fish-detector/tree/35564151a9f0f9c639ec5d0eb758fe35a64fa687?tab=readme-ov-file#quick-start).

### Clone the repo

```bash
git clone https://github.com/WildHackers/community-fish-detector.git
cd community-fish-detector
```

### Download the model weights

- You download weights from the [Releases page]([url](https://github.com/WildHackers/community-fish-detector/releases)).

### Install dependencies

```bash
pip install -r requirements.txt 
```

### Run inference

#### Command-line batch inference 

`rf_detr_batch_inference.py` runs the model recursively on a folder of images, and writes results in [MegaDetector output format](http://lila.science/megadetector-output-format).

```bash
python rf_detr_batch_inference.py "/path/to/your/model.pth" "/path/to/your/image/folder" "/path/to/your/output/file.json"
```

#### Programmatic inference

```python
from rfdetr import RFDETRNano
import supervision as sv
from PIL import Image

# Load model
model = RFDETRNano(pretrain_weights="/path/to/your/model.pth", resolution=640)

# Run on an image
image = Image.open("/path/to/your/image.jpg")
detections = model.predict(image, threshold=0.3)

# Annotate and save the result
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [f"fish {conf:.2f}" for conf in detections.confidence]

annotated = box_annotator.annotate(image.copy(), detections)
annotated = label_annotator.annotate(annotated, detections, labels=labels)
annotated.save("/path/to/your/output.jpg")
```

Be sure to specify `resolution=640`; the RF-DETR Nano architecture defaults to 384, but our detector was fine-tuned at 640, and we expect that you will get better results at 640.  This is not necessary for the batch inference script, which automatically detects the training size.

## Eastern Pacific fine-tuning study

This fork adds a study fine-tuning a detector on two reef video transects from
Utría, Colombian Pacific, and comparing it against the general-purpose CFD
baseline. Full write-up, including a data audit of the original pipeline:
[`docs/data-audit-and-remediation.md`](docs/data-audit-and-remediation.md).

The annotations cover 790 frames and 1,771 boxes across 13 species, but 60% of
boxes are an unidentified `not_defined` catch-all and four species have one or
two examples, so species detection is not supportable. The study collapses to a
single `fish` class — an unidentified fish is still a fish — which also puts it
on the same task as the CFD baseline.

### Scripts

| Script | Purpose |
|--|--|
| [`scripts/build_splits.py`](scripts/build_splits.py) | VIA annotations → YOLO dataset with leakage-free train/val/test splits |
| [`scripts/train_yolo.py`](scripts/train_yolo.py) | Fine-tune a YOLO detector from a dataset config |
| [`scripts/evaluate.py`](scripts/evaluate.py) | Score any number of models on a split with identical metrics |
| [`scripts/confusion_matrix.py`](scripts/confusion_matrix.py) | Confusion matrix and per-species recall |
| [`scripts/detection_metrics.py`](scripts/detection_metrics.py) | Shared metrics (mAP via pycocotools, IoU matching) |
| [`scripts/detectors.py`](scripts/detectors.py) | Shared model loading and cached inference |

### Avoiding data leakage

Frames are sampled one per three seconds from continuous transects, so
neighbouring frames are near-duplicates and a random split leaks them across
the boundary. `build_splits.py` cuts each video into contiguous temporal
blocks, assigns whole blocks, and drops a guard band either side of every
boundary where the assignment changes. No frame in one split is within 15
seconds of real time of a frame in another.

```bash
python3 scripts/build_splits.py \
  --via-json annotations/reef1_via_clean.json \
  --images-root data/images_jpeg_reef_01 data/images_jpeg_reef_02 \
  --dataset-dir dataset/pacifico-fish \
  --splits-dir configs/splits \
  --yaml-out configs/pacifico-fish-1class.yaml \
  --pattern train val train train test train train val train train test train
```

| Split | Images | Boxes | Background frames |
|---|---:|---:|---:|
| train | 495 | 981 | 148 |
| val | 116 | 232 | 38 |
| test | 115 | 369 | 24 |

### Training

```bash
python3 scripts/train_yolo.py --data configs/pacifico-fish-1class.yaml --device mps
```

yolov8n, 50 epochs, imgsz 640, batch 8. Validation mAP50 peaks at **0.583**
(epoch 32); validation loss bottoms at epoch 33 and drifts up after, so the
50-epoch budget overshoots.

### Evaluation

```bash
python3 scripts/evaluate.py \
  --dataset-dir dataset/pacifico-fish --split test \
  --model "CFD baseline" rfdetr models/community-fish-detector-2026.02.02-rf-detr-nano-640.pth \
  --model "Fine-tuned" yolo runs/detect/pacifico_fish_1class/weights/best.pt \
  --out results/test_comparison.json
```

On the held-out test split (115 images, 369 boxes):

| Model | mAP50 | mAP50-95 | P@0.25 | R@0.25 | F1 |
|---|---:|---:|---:|---:|---:|
| CFD baseline (RF-DETR nano, 640) | 0.172 | 0.054 | 0.193 | 0.287 | 0.231 |
| Fine-tuned (yolov8n, 640) | **0.733** | **0.323** | 0.655 | 0.715 | 0.684 |

**That 4.3× is not an improvement in detection.** Sweeping the IoU threshold
shows the baseline finds *more* fish than the fine-tuned model when boxes only
have to overlap loosely:

| IoU | Baseline recall | Baseline TP | Fine-tuned recall | Fine-tuned TP |
|---:|---:|---:|---:|---:|
| 0.10 | **0.821** | **303** | 0.762 | 281 |
| 0.30 | 0.648 | 239 | 0.759 | 280 |
| 0.50 | 0.287 | 106 | 0.715 | 264 |

The baseline draws systematically tighter boxes than this project's annotator,
so correct detections are rejected on IoU: it loses 65% of its finds between
IoU 0.10 and 0.50, against 6% for the fine-tuned model. What fine-tuning bought
is agreement with the annotation convention and roughly half the false
positives — not better detection.

### Confusion matrix and per-species recall

```bash
python3 scripts/confusion_matrix.py \
  --dataset-dir dataset/pacifico-fish --via-json annotations/reef1_via_clean.json \
  --split test \
  --model "CFD baseline" rfdetr models/community-fish-detector-2026.02.02-rf-detr-nano-640.pth \
  --model "Fine-tuned" yolo runs/detect/pacifico_fish_1class/weights/best.pt \
  --out results/test_confusion.json
```

At conf 0.25, IoU 0.50. Both models are single-class, so there is no class
confusion to report and the matrix is detection against background; the
background/background cell is undefined for detection.

| | Baseline: actual fish | Baseline: actual bg | | Fine-tuned: actual fish | Fine-tuned: actual bg |
|---|---:|---:|---|---:|---:|
| **predicted fish** | 106 | 442 | | 264 | 139 |
| **predicted background** | 263 | n/a | | 105 | n/a |

Recall by the species the annotator assigned, even though neither model
predicts species:

| Species | Boxes | Baseline recall | Fine-tuned recall |
|---|---:|---:|---:|
| `not_defined` | 209 | 0.167 | 0.617 |
| *Thalassoma lucasanum* | 86 | 0.314 | **0.942** |
| *Azurina atrilobata* | 49 | 0.653 | 0.694 |
| *Stegastes acapulcoensis* | 10 | 0.000 | **1.000** |
| *Bodianus diplotaenia* | 7 | 0.714 | 0.857 |
| *Scarus ghobban* | 5 | **1.000** | 0.400 |
| *Diodon holocanthus* | 3 | 0.667 | 0.667 |

Only the first three rows carry enough boxes to be meaningful; the rest are
reported for completeness and should not be read as trends.

`not_defined` is the hardest class for both models, which is consistent with
those being the small, distant or blurred fish the annotator could not identify
— and is an argument for keeping them as positives rather than discarding them.


### Five-class species variant

`build_splits.py --task species` gives its own class to every species with at
least `--min-boxes` annotations, pools the rest as `other_fish`, and keeps
`not_defined` as an explicit `unidentified` class rather than deleting it —
dropped boxes become background, which teaches the model that fish are
background.

```bash
python3 scripts/build_splits.py \
  --via-json annotations/reef1_via_clean.json \
  --images-root data/images_jpeg_reef_01 data/images_jpeg_reef_02 \
  --dataset-dir dataset/pacifico-species \
  --splits-dir configs/splits-species \
  --yaml-out configs/pacifico-species-5class.yaml \
  --task species --min-boxes 60 \
  --pattern train val train train test train train val train train test train

python3 scripts/train_yolo.py --data configs/pacifico-species-5class.yaml \
  --device mps --name pacifico_species_5class
```

Test results, and training box counts for contrast:

| Class | train boxes | test boxes | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | 981 | 368 | **0.588** | 0.296 |
| *Stegastes acapulcoensis* | 42 | 10 | **0.852** | 0.466 |
| *Thalassoma lucasanum* | 192 | 86 | 0.679 | 0.330 |
| `other_fish` | 39 | 15 | 0.621 | 0.328 |
| `unidentified` | 651 | 209 | 0.432 | 0.170 |
| *Azurina atrilobata* | 57 | 48 | 0.355 | 0.186 |

**The ranking is almost inverted against training data volume.** *Stegastes*
has 42 training boxes and the best score; `unidentified` has fifteen times as
many and is second worst. What predicts performance here is the number of
*independent* observations — *Azurina*'s 57 boxes come from 8 contiguous
episodes of a schooling fish, while *Stegastes*' 42 are spread over 19 — and
intrinsic difficulty: `unidentified` is by definition the fish the annotator
could not make out.

Species labels cost detection performance. Collapsing every prediction to
`fish` and scoring against the same ground truth:

| Model | mAP50 | mAP50-95 | P@0.25 | R@0.25 | F1 |
|---|---:|---:|---:|---:|---:|
| CFD baseline (RF-DETR nano) | 0.172 | 0.054 | 0.193 | 0.287 | 0.231 |
| Fine-tuned, 1 class | **0.733** | **0.323** | 0.655 | 0.715 | 0.684 |
| Fine-tuned, 5 classes, collapsed | 0.663 | 0.298 | 0.679 | 0.659 | 0.669 |

About 10% of detection performance buys the species labels. Note that
*Stegastes* rests on 10 test boxes and `other_fish` on 15, so those figures
have wide intervals and should not be quoted as point estimates.

## Contributors

This model was created by a collective effort of the following folks: <a href="https://www.linkedin.com/in/filippo-varini/">Filippo Varini</a>, <a href="https://dmorris.net">Dan Morris</a>, <a href="https://www.linkedin.com/in/sonny-burniston/">Sonny Burniston</a>, <a href="https://www.oceaneboulais.net/">Oceane Boulais</a>, <a href="https://www.mbari.org/person/kevin-barnard/">Kevin Barnard</a>, <a href="https://www.mbari.org/person/laura-chrobak/">Laura Chrobak</a>, <a href="https://alexvmt.github.io/">Alexander Merdian-Tarko</a>, <a href="https://www.linkedin.com/in/kameswari-devi-ayyagari-031820b7/">Devi Ayyagari</a>, <a href="https://www.linkedin.com/in/mona-dhiflaoui/">Mona Dhiflaoui</a>, <a href="https://www.linkedin.com/in/jiashu-chen-w/">Joshua Chen</a>, and many others.

If you contributed, but you don't see your name here, please [email us](mailto:fppvrn@gmail.com).

We welcome further contributions; if you have a dataset that could expand coverage, or want to contribute to improving the model, please [reach out](mailto:fppvrn@gmail.com)!

## Example Predictions

Below we provide some visual examples that overlay the ground truth with the model detections, to give you a qualitative sense of the model's training domain.

<img src="./assets/example7.png" />
<img src="./assets/example1.png" />
<img src="./assets/example2.png" />
<img src="./assets/example3.png" />
<img src="./assets/example4.png" />
<img src="./assets/example5.png" />
<img src="./assets/example6.png" />

## Also see

* [Hugging Face Space](https://huggingface.co/spaces/FathomNet/community-fish-detector) for this model set up by the [FathomNet](https://www.fathomnet.org/) community.
