from ultralytics import YOLO

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rc('font', family='NanumGothic')
mpl.rcParams['axes.unicode_minus'] = False

# Load a model
model = YOLO("runs/detect/outputs_origin/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(data="validate_lowlight_gamma.yaml")  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # list of mAP50-95 for each category