# TTA for object detection
It's a module to make Test Time Augmentation with OD
Used augmnetations are constant

Also using NMS module to make predictions more clear for searching faces of objects

## Using
1. Install keras-retinanet https://github.com/fizyr/keras-retinanet
2. Just pass your image and trained model to TTA function and get ready-made boxes, scores and labels, calculated with TTA and NMS

## Testing
Use file test_TTA to try (replace image path)
