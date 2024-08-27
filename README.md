# Coral Segmentation Project with YOLOv8

This project focuses on coral segmentation using the YOLOv8 model and machine learning techniques.

## Notebooks

1. **`yolov8_train.ipynb`:** This notebook is the core of the project, handling the training, validation, and inference of the YOLOv8 model. It is essential for accurate coral detection and segmentation in images.

2. **`cross-validation.ipynb`:** Implements K-Fold cross-validation to robustly evaluate the performance of the trained YOLOv8 model and ensure its generalization to new data.

3. **`mask_%.ipynb`:** This notebook analyzes segmentation masks generated by YOLOv8. It calculates the percentage of area occupied by each detected object in the images and saves the results in a CSV file.

4. **`area_img.ipynb`:** Contains a script that utilizes telemetry data and image areas to estimate the area covered by an image based on the height of the ROV (Remotely Operated Vehicle). While the results may not be perfect, it offers an initial approximation to the problem.

5. **`auto_annotate.ipynb`:** Implements an automatic annotation function that combines YOLOv8 and the Segment Anything Model (SAM) to streamline the data labeling process and facilitate the training of segmentation models.

6. **`non-common.ipynb`:** This notebook handles mismatched files (images without corresponding annotations) during the auto-annotation process.

7. **`seg_coco_json_to_yolo.py`:** This Python script converts annotations in COCO JSON format to the YOLOv8 PyTorch format, ensuring data compatibility with the training model.

8. **`CWC_GIS.ipynb`:** Joins the data from telemetry and the .csv from **'mask_%.ipynb'** to create a .csv for QGIS.

9. **`analysis_CWC.ipynb`:** Estadistical analysis of the data and figure extraction.

10. **`images`:** Val images

11. **`model_weights`:** https://drive.google.com/file/d/1d6u6H1Dd1CSFnlcBHoR08GrjV3yhhfoJ/view?usp=sharing


## Usage Instructions

1. **Data Preparation:** Organize your images and annotations in the folder structure required by YOLOv8.
2. **Training:** Run the `yolov8_train.ipynb` notebook to train the model on your data.
3. **Automatic Annotations (Optional):** If you need to label more images, use `auto_annotate.ipynb` to generate annotations automatically.
4.  **Annotation Conversion:** If your annotations are in COCO JSON format, use `seg_coco_json_to_yolo.py` to convert them to the YOLOv8 PyTorch format.
5. **Evaluation:** Use `cross-validation.ipynb` to evaluate the performance of the trained model.
6. **Mask Analysis:** Run `mask_%.ipynb` to analyze the segmentation masks and get information about the area covered by objects.
7. **Area Estimation (Optional):** If you have telemetry data, use `area_img.ipynb` to estimate the area covered by the images.
8. **QGIS (Optional):** Creates a GIS importable `CWC_GIS.ipynb`.
9. **Estadistical analysis:** `analysis_CWC.ipynb`

## Requirements

Make sure you have the following Python libraries installed:

- `ultralytics`
- `torch`
- `pandas`
- `numpy`
- `matplotlib`
- `Pillow (PIL)`
- `glob`
- `scipy`
- `seaborn`
- `matplotlib.pyplot`
- `scikit_posthocs`
- `sklearn`


If you have any questions or issues, feel free to open an issue in this repository!
