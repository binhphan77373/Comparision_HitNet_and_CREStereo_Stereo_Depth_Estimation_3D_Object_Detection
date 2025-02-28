# from yolov4.tf import YOLOv4
from ultralytics import YOLO
import tensorflow as tf
import time
import cv2
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# class ObjectDetectorAPI:
#     def __init__(self):       
#         self.model = YOLO("Yolov8/yolov8x.pt")
    
#     def predict(self, image):
#         start_time = time.time()
        
#         # YOLOv8 doesn't require manual color conversion or resizing
#         results = self.model(image,classes = [2,7])
        
#         # Process results
#         pred_bboxes = []
#         for r in results:
#             boxes = r.boxes.xyxy.cpu().numpy()
#             confs = r.boxes.conf.cpu().numpy()
#             cls = r.boxes.cls.cpu().numpy()
            
#             for box, conf, cl in zip(boxes, confs, cls):
#                 x1, y1, x2, y2 = box
#                 # YOLOv8 returns xyxy format, so we need to convert to xywh
#                 w = x2 - x1
#                 h = y2 - y1
#                 pred_bboxes.append([x1, y1, w, h, cl, conf])
        
#         pred_bboxes = np.array(pred_bboxes)
        
#         # Filter boxes based on score threshold
#         score_threshold = 0.5
#         pred_bboxes = pred_bboxes[pred_bboxes[:, 5] > score_threshold]
        
#         # Draw bounding boxes
#         result = image.copy()
#         for bbox in pred_bboxes:
#             x1, y1, w, h, class_id, conf = bbox
#             x2, y2 = x1 + w, y1 + h
#             cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            

#         exec_time = time.time() - start_time
#         print(f"Inference time: {exec_time * 1000:.2f} ms")

#         return result, pred_bboxes    
    
   
# class ObjectDetectorAPI:
#     def __init__(self):       
#         self.model = YOLO("Yolov8/yolov8x.pt")
    
#     def predict(self, image):
#         start_time = time.time()
        
#         # YOLOv8 doesn't require manual color conversion or resizing
#         results = self.model(image,classes = [2,7])
        
#         # Process results
#         pred_bboxes = []
#         for r in results:
#             boxes = r.boxes.xyxy.cpu().numpy()
#             confs = r.boxes.conf.cpu().numpy()
#             cls = r.boxes.cls.cpu().numpy()
            
#             for box, conf, cl in zip(boxes, confs, cls):
#                 x1, y1, x2, y2 = box
#                 # YOLOv8 returns xyxy format, so we need to convert to xywh
#                 w = x2 - x1
#                 h = y2 - y1
#                 pred_bboxes.append([x1, y1, w, h, cl, conf])
        
#         pred_bboxes = np.array(pred_bboxes)
        
#         # Filter boxes based on score threshold
#         score_threshold = 0.5
#         pred_bboxes = pred_bboxes[pred_bboxes[:, 5] > score_threshold]
        
#         # Draw bounding boxes
#         result = image.copy()
#         for bbox in pred_bboxes:
#             x1, y1, w, h, class_id, conf = bbox
#             x2, y2 = x1 + w, y1 + h
#             cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            

#         exec_time = time.time() - start_time
#         print(f"Inference time: {exec_time * 1000:.2f} ms")

#         return result, pred_bboxes

class ObjectDetectorAPI:
    def __init__(self):       
        self.model = YOLO("Yolov8/yolov8x.pt")
    
    def predict(self, image):
        start_time = time.time()
        
        # YOLOv8 doesn't require manual color conversion or resizing
        results = self.model(image, classes=[2, 7])
        
        # Process results
        pred_bboxes = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            
            for box, conf, cl in zip(boxes, confs, cls):
                x1, y1, x2, y2 = box
                # YOLOv8 returns xyxy format, so we need to convert to xywh
                w = x2 - x1
                h = y2 - y1
                pred_bboxes.append([x1, y1, w, h, cl, conf])
        
        # Convert to numpy array safely
        if not pred_bboxes:  # If empty list
            print("No objects detected")
            exec_time = time.time() - start_time
            print(f"Inference time: {exec_time * 1000:.2f} ms")
            return image.copy(), np.array([])
        
        pred_bboxes = np.array(pred_bboxes)
        
        # Check if the array has the right shape before filtering
        if len(pred_bboxes.shape) != 2 or pred_bboxes.shape[1] <= 5:
            print(f"Warning: pred_bboxes has unexpected shape: {pred_bboxes.shape}")
            exec_time = time.time() - start_time
            print(f"Inference time: {exec_time * 1000:.2f} ms")
            return image.copy(), pred_bboxes
        
        # Apply confidence filtering safely
        score_threshold = 0.5
        try:
            filtered_bboxes = pred_bboxes[pred_bboxes[:, 5] > score_threshold]
            pred_bboxes = filtered_bboxes
        except IndexError as e:
            print(f"Error during filtering: {e}")
            print(f"pred_bboxes shape: {pred_bboxes.shape}, type: {type(pred_bboxes)}")
            # Continue with unfiltered boxes
        
        # Handle case where all boxes were filtered out
        if len(pred_bboxes) == 0:
            print("No objects above confidence threshold")
            exec_time = time.time() - start_time
            print(f"Inference time: {exec_time * 1000:.2f} ms")
            return image.copy(), np.array([])
        
        # Draw bounding boxes
        result = image.copy()
        for bbox in pred_bboxes:
            # Make sure we have enough elements in the bbox
            if len(bbox) >= 6:
                x1, y1, w, h, class_id, conf = bbox
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Optional: add label
                label = f"Class {int(class_id)}: {conf:.2f}"
                cv2.putText(result, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        exec_time = time.time() - start_time
        print(f"Inference time: {exec_time * 1000:.2f} ms")

        return result, pred_bboxes