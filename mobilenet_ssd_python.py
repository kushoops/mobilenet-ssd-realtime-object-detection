import numpy as np
import argparse
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                        help='Path to text network file: MobileNetSSD_deploy.prototxt for Caffe model or')
    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                        help='Path to weights: MobileNetSSD_deploy.caffemodel for Caffe model or')
    parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
    return parser.parse_args()

def load_model(prototxt, weights):
    return cv2.dnn.readNetFromCaffe(prototxt, weights)

def get_video_capture(video_path):
    return cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)

def preprocess_frame(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    return blob, frame_resized

def non_maximum_suppression(detections, confidence_threshold, nms_threshold=0.3):
    boxes = []
    confidences = []
    class_ids = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            xLeftBottom = int(detections[0, 0, i, 3] * 300)
            yLeftBottom = int(detections[0, 0, i, 4] * 300)
            xRightTop = int(detections[0, 0, i, 5] * 300)
            yRightTop = int(detections[0, 0, i, 6] * 300)
            
            boxes.append([xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return [(boxes[i], confidences[i], class_ids[i]) for i in indices]
    else:
        return []

def draw_detections(frame, detections, confidence_threshold, class_names):
    heightFactor = frame.shape[0] / 300.0
    widthFactor = frame.shape[1] / 300.0

    filtered_detections = non_maximum_suppression(detections, confidence_threshold)
    for (box, confidence, class_id) in filtered_detections:
        xLeftBottom, yLeftBottom, box_width, box_height = box
        xRightTop = xLeftBottom + box_width
        yRightTop = yLeftBottom + box_height

        xLeftBottom = int(widthFactor * xLeftBottom)
        yLeftBottom = int(heightFactor * yLeftBottom)
        xRightTop = int(widthFactor * xRightTop)
        yRightTop = int(heightFactor * yRightTop)

        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

        if class_id in class_names:
            label = f"{class_names[class_id]}: {confidence:.2f}"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]), 
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine), 
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            print(label)

def main():
    args = parse_arguments()

    class_names = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                   5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
                   11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
                   16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

    net = load_model(args.prototxt, args.weights)
    cap = get_video_capture(args.video)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        blob, frame_resized = preprocess_frame(frame)
        net.setInput(blob)
        detections = net.forward()
        
        draw_detections(frame, detections, args.thr, class_names)
        
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) >= 0:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
