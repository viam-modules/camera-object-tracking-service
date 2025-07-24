# CAMERA-OBJECT-TRACKER-SERVICE

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for tracking object using feature-based matching.


## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/modular-resources/#configuration) and select the `viam:vision:camera-object-tracking-service` model from the [`camera-object-tracking-service` module](https://app.viam.com/module/viam/camera-object-tracking-service).
This module implements the following methods of the [vision service API](https://docs.viam.com/services/vision/#api):

- `GetDetections()`: returns the bounding boxes with the unique id as label and the matching confidence.
- `CaptureAllFromCamera()`: returns the image and detections all together, given a camera name.

The tracking problem consists of two parts: detection and visual matching. The `camera-object-tracking-service` module provides the necessary logic to perform object tracking, given a detector and a visual matcher (i.e., an embedder). If you don't configure an embedder (ML model service) or detector (vision service), defaults will be used. Defaults are chosen to be general purpose.

## Configure your `camera-object-tracking-service` vision service

> [!NOTE]
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add vision / camera-object-tracking-service to your machine](https://docs.viam.com/configure/#components).

### Attributes description

| Name                  | Type   | Inclusion | Default                | Description                                                                                                                                                                                                 |
|-----------------------|--------|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `camera_name`         | string | Required  | -                      | The name of the (configured) camera to be used by the tracking model.                                                                                                |
| `detector_name`       | string | Optional  | Pre-trained Faster R-CNN| The name of the underlying detector (configured vision service) to be used by the tracking model. The default uses a pre-trained Faster R-CNN model within the module.                                      |
| `max_frequency_hz`    | float  | Optional  | 10                     | Frequency at which the tracking steps are performed and logs updated. May be slower if processor can’t keep up, but it’s always the max frequency.                                                          |
| `embedder_name`       | string | Optional  | (Pre-trained ResNet) Not implemented yet     | The name of the (configured) mlmodel service that will make a visual feature vector from an input image. The default uses a pre-trained ResNet model within the module.                                     |
| `embedder_distance`   | string | Optional  | -                      | Either 'cosine' or 'euclidean' for the similarity distance between an entity in two different frames. Only required when `embedder_name` is given.                                                          |
| `embedder_threshold`  | float  | Optional  | -                      | Any two tracks with an embedding distance above this number will be considered two different entities. Only required when `embedder_name` is given.                                                         |
| `classifier_name`     | string | Optional  | -                   | The name of the classifier (configured vision service) that will classify each detection. When present, the classification of each track will be included in the output track logs.                          |
| `lambda`              | float [0-1] | Optional  | TBD                    | When matching across frames, how much to trust the motion vs. the visual feature matching. `lambda = 0.05` leans towards motion, `lambda = 0.95` towards visual feature matching.                           |
| `zones`               | map of string-ListOfListOfOrderedPair | Optional  | None                   | ** Not implemented yet **  |
| `chosen_labels`       | map of string-float | Optional  | Empty                  | A list of class names (string) and confidence scores (float[0-1]) such that only detections with a class_name in the list and a confidence above the corresponding score are included. If empty, no filter.  |
| `crop_region`             | dict   | Optional  | -        | Defines a region of the image to crop for processing. Must include four float values between 0 and 1: `x1_rel`, `y1_rel`, `x2_rel`, `y2_rel` representing the relative coordinates of the crop region. |

## Makefile targets for arm-jetson JP6 machines only

This project includes a `Makefile` script to automate the PyInstaller build process for Jetson machines. Building and deploying the module for other platforms should be done through CI.
PyInstaller is used to create standalone executables from the Python module scripts.

####  1. `make setup-jp6`

1. installs system dependencies (cuDNN and cuSPARSELt)
2. creates venv environment (under `./build/.venv`)
3. gets/builds python packages wheel files - Torch, Torchvision (built from source)

Cleaned with `make clean` (this also deletes pyinstaller build directory)

#### 2.  `make module.tar.gz`
This command builds the module executable using PyInstaller and creates a `.tar` file with the files needed to run the module.
The PyInstaller executable is created under `./build/pyinstaller_dist`.

#### 3. Upload to the registry with the right tag

```bash
viam login
viam module upload --version 0.0.0-rc0 --platform linux/arm64 --tags 'jetpack:6' archive.tar.gz
```

Cleaned with `make clean`
