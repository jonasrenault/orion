# Models

Orion provides [YOLO12](https://docs.ultralytics.com/models/yolo12/) models fine-tuned on a [custom dataset](datasets.md) of military vehicles with [4 classes](classes.md).

| Model                                                                                  | size<br><sup>(pixels) | params<br><sup>(M)   |
| ------------------------------------------------------------------------------------   | --------------------- | -------------------- |
| [orion12n](https://github.com/jonasrenault/orion/releases/download/v2.0.0/orion12n.pt) | 640                   | 2.6                  |
| [orion12s](https://github.com/jonasrenault/orion/releases/download/v2.0.0/orion12s.pt) | 640                   | 9.3                  |
| [orion12m](https://github.com/jonasrenault/orion/releases/download/v2.0.0/orion12m.pt) | 640                   | 20.2                 |
| [orion12l](https://github.com/jonasrenault/orion/releases/download/v2.0.0/orion12l.pt) | 640                   | 26.4                 |

## Training results

### Orion12n

![Orion12n training results](trains/orion12n/results.png)

### Orion12s

![Orion12s training results](trains/orion12s/results.png)

### Orion12m

![Orion12m training results](trains/orion12m/results.png)

### Orion12l

![Orion12l training results](trains/orion12l/results.png)

## Usage

To use one of Orion's models, download it from the [links above](#models) and then use one of Orion's CLI commands:

### Detect military vehicles in images

The `predict` command will use the model to detect military vehicles in images.

```bash
orion predict --help

 Usage: orion predict [OPTIONS] MODEL_PATH DATA

 Run predictions on a set of images using the given model.

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_path      PATH  model path. [required]                                                                                   │
│ *    data            PATH  data to make predictions on. [required]                                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --save       -s                               save annotated images.                                                                │
│ --save-txt       --no-save-txt                save detection results in a txt file. [default: save-txt]                             │
│ --save-conf      --no-save-conf               save confidence score for each detection. [default: save-conf]                        │
│ --output     -o                    DIRECTORY  save directory. [default: Path.cwd() /runs/predict]                                   │
│ --help                                        Show this message and exit.                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

For example, if you downloaded the [orion12m.pt](#models) model file into your current directory, and want to use it to detect military vehicles in an image, run

```bash
orion predict ./orion12m.pt resources/test/afvs.jpg -s
```

!!! tip
    The `predict` command with the `-s` option will save the annotated image.

![Annotated AFVs](imgs/afvs.jpg)

### Track military vehicles in videos

The `track` command will use the model to track military vehicles in videos.

```bash
orion track --help

 Usage: orion track [OPTIONS] MODEL_PATH DATA

 Track tanks in a video using a YOLO model and specified tracker.

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_path      PATH  model path. [required]                                                                                   │
│ *    data            PATH  input video. [required]                                                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --conf     -c      FLOAT      confidence threshold for detections. [default: 0.5]                                                   │
│ --tracker  -t      TEXT       tracker configuration file. [default: botsort.yaml]                                                   │
│ --output   -o      DIRECTORY  save directory. [default: Path.cwd() /runs/track]                                                     │
│ --help                        Show this message and exit.                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

For example, if you downloaded the [orion12m.pt](#models) model file into your current directory, and want to use it to track military vehicles in a video, run

```bash
orion track ./orion12m.pt resources/test/mev1.mp4
```

![MEV tracking](imgs/mev_tracking.gif)
