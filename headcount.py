import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 on Webcam")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera ID (usually 0 for built-in webcam)")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    camera_id = args.camera_id

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Set the frame width and height based on the camera's properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (
        ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon,
                          frame_resolution_wh=(frame_width, frame_height))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    people_count = 0  # Variable to keep track of the number of people

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        # Count the number of people detected
        people_count = sum(1 for label in labels if "person" in label.lower())

        # Annotate the frame with the count
        frame = cv2.putText(frame, f"People Count: {people_count}", (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
