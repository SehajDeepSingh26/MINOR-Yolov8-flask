from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Define the Alert model in the main.py file


class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    screenshot_paths = db.Column(db.String(200))


# Directory to save screenshots
SCREENSHOTS_DIR = "static/screenshots"

# Create the screenshots directory if it doesn't exist
if not os.path.exists(SCREENSHOTS_DIR):
    os.makedirs(SCREENSHOTS_DIR)

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

# Placeholder for parse_arguments function


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

# Placeholder for display_alert function


def display_alert(item_name, current_datetime):
    alert_message = f"Alert: Objectionable item ({item_name}) detected at {current_datetime}"
    print(alert_message)
    return alert_message

# ... Rest of your code ...


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (
        ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon,
                          frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    alert_log = []  # List to store alert messages

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        for label in labels:
            if "gun" in label.lower() or "knife" in label.lower():
                # Save the screenshot with a timestamp
                current_datetime = time.strftime("%Y%m%d%H%M%S")
                screenshot_path = os.path.join(
                    SCREENSHOTS_DIR, f"screenshot_{current_datetime}.jpg")
                screenshot_path = screenshot_path.replace("\\", "/")
                cv2.imwrite(screenshot_path, frame)
                item_name = label.split()[0]

                # Create an Alert object and add it to the database
                alert = Alert(item_name=item_name, screenshot_paths=screenshot_path)
                db.session.add(alert)
                db.session.commit()

                alert_message = display_alert(item_name, current_datetime)
                alert_log.append(alert_message)

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)

        key = cv2.waitKey(30)
        if key == 27:  # ESC key to stop the program
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the alert log when the program exits
    print("\nAlert Log:")
    for alert_message in alert_log:
        print(alert_message)


if __name__ == "__main__":
    db.create_all()
    main()
