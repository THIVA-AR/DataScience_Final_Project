# DataScience_Final_Project


### Problem and objective
The goal is to automatically detect soldiers, weapons, vehicles, and trenches in images and videos, and then classify each scene as high threat, medium threat, or non‑threat. High‑threat situations are usually those where weapons are visible; medium threat occurs when only soldiers are detected; low or non‑threat scenes contain only vehicles/background or nothing relevant.

### Dataset preparation and filtering
The work starts by collecting a dataset of military/civilian scenes in YOLO format (images plus .txt label files per image). Each label file contains class IDs and bounding boxes in the normalized YOLO format.
​
Because the original dataset contained many classes, a filtering step creates a cleaner detection problem:

Raw classes (for example: multiple soldier categories, several vehicle types, weapon variants, trenches).

A Python filtering script scans each label file, keeps only desired classes, and remaps them into four final categories:

0 → soldier (merging different soldier IDs)

1 → weapon

2 → vehicle (merging multiple vehicle‑type IDs)

Only images that contain at least one of these classes are copied into a new dataset, with corresponding labels. This produces Military_dataset_filtered/ with standard YOLO folder structure: images/train, images/val, labels/train, labels/val, and a dataset YAML that describes paths, number of classes, and class names.
​

In addition, a second filtered dataset is built containing only weapons. For this, the script keeps only annotations whose original class is “weapon”, remaps them to a single class 0, and copies only those images. This becomes weapon_filtered/ with its own weapon_only.yaml. The purpose is to train a highly specialized weapon detector on images where weapons are clear and more balanced relative to the other classes.

Model training strategy
Training is done using YOLOv8 from Ultralytics, which provides a high‑level Python API for detection models.
​

Multi‑class model (soldier, weapon, vehicle, trench)
A pre‑trained YOLOv8 backbone (for example, yolov8n.pt or yolov8s.pt) is fine‑tuned on the filtered four‑class dataset. The training configuration includes:

Input size 640×640.

Around 60 epochs with early stopping based on validation performance.

Batch size tuned to available GPU memory.

Standard augmentations (flip, color jitter, mosaic) handled by the YOLO training pipeline.
​

During training, YOLOv8 automatically tracks metrics such as precision, recall, mAP@0.5, and mAP@0.5:0.95 for each class and overall, based on the validation set.
​
In this project, the multi‑class model generally learns soldier and vehicle very well but struggles more with weapon detection because weapons are often smaller, partially occluded, and less frequent in the original distribution – a common pattern in imbalanced detection tasks.
​

Weapon‑only specialist model
To strengthen weapon detection, a second YOLOv8 model is trained on the weapon‑only dataset. This model has:

Only one class (weapon).

A smaller but more focused dataset where weapons occupy a larger portion of each image.

Because the task is simpler and the class is better balanced, this model usually achieves much higher mAP and recall for weapons than the multi‑class model. The project then uses this specialist for all weapon predictions, while relying on the multi‑class model for soldier/vehicle/trench.

Combined inference and threat logic
At inference time (for an image, video frame, or webcam frame), the system runs both models on the same frame:

The multi‑class YOLO model predicts soldiers, vehicles, and trenches.

The weapon specialist model predicts weapons.

Detections from both models are merged in code. Overlapping boxes are reduced via YOLO’s built‑in non‑maximum suppression (NMS) and confidence thresholds, so only the strongest bounding boxes remain.
​

A simple threat classification layer is added on top:

If one or more weapons are detected → “HIGH THREAT: N weapon(s)”.

Else if one or more soldiers are detected → “MEDIUM: N soldier(s)”.

Else → “LOW / NON‑THREAT” (only vehicles, trenches, or background).

This logic can be extended further (for example, by considering distance between soldiers and weapons, or distinguishing friendly vs. enemy soldiers), but the current implementation gives a clear, interpretable decision for each frame.

Performance evaluation
To evaluate the trained models, the project uses YOLOv8’s validation mode on the held‑out val split defined in the YAML. This automatically computes:
​

Precision (P) – proportion of predicted boxes that are correct.

Recall (R) – proportion of ground‑truth objects that the model finds.

mAP@0.5 – mean average precision at IoU 0.5, which is the standard YOLO quality metric.

mAP@0.5:0.95 – mean AP averaged over IoU thresholds from 0.5 to 0.95, as in COCO.
​

Metrics are reported overall and per class (soldier, weapon, vehicle, trench), so it becomes clear that weapon performance improves significantly in the weapon‑only model compared to the original multi‑class model.

Streamlit web application
To make the system user‑friendly, the project includes a Streamlit web app as the front end.
​

### Key behaviors:

Model loading: On startup, the app loads both YOLO models and keeps them in memory using a cached resource so they are not reloaded on every request, which speeds up interaction.

### Image mode:

The user uploads an image.

The app decodes it, runs the combined detection + threat logic, and displays:

The image with bounding boxes for soldiers, weapons, vehicles, and trenches.

The threat classification text (for example, “HIGH THREAT: 2 weapon(s)”) just above the image for clarity.

A processed image can be downloaded for further analysis or reporting.

### Video mode:

The user uploads an MP4 or similar video file.

The app processes the video frame by frame, performing the same combined detection and threat logic.

A preview is shown in the browser as frames update, and an annotated video file is written in the background.

After processing, the user can download the annotated video and see the final threat status summarizing the clip.

Streamlit is used because it allows rapid creation of interactive apps in pure Python, with built‑in widgets for file upload, display, and download, which is ideal for packaging a computer‑vision model for non‑technical users.
​

### Summary of workflow
Putting it all together, the project workflow is:

Collect & label data for soldiers, weapons, vehicles, trenches (YOLO format).

Filter and remap classes to a consistent four‑class dataset, plus a weapon‑only subset.

### Train:

Multi‑class YOLOv8 detector on the filtered dataset.

Weapon‑only YOLOv8 detector on the focused dataset.

Evaluate both using precision, recall, and mAP to confirm that the weapon specialist significantly improves weapon detection quality.
​

### Deploy via:

A Python inference script for real-time detection (image/video/webcam).

A Streamlit web app that lets users upload media, view detections and threat labels, and download results.
