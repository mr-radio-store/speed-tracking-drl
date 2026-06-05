from ultralytics import YOLO
import cv2
import time
import numpy as np


def main():

    # =========================
    # Load YOLO
    # =========================
    model = YOLO("yolov8n.pt")
    print("✅ YOLO loaded")

    # =========================
    # Open Camera (FIXED)
    # =========================
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    print("✅ Camera ready")

    # =========================
    # FPS
    # =========================
    prev_time = time.time()

    # =========================
    # Main Loop
    # =========================
    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ Frame dropped")
            continue

        # Prevent YOLO on black frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < 2:
            cv2.putText(frame, "BLACK FRAME DETECTED",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)
            cv2.imshow("YOLOv8", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # =========================
        # YOLO inference
        # =========================
        results = model.predict(frame, conf=0.25, verbose=False)

        output = frame.copy()

        if len(results[0].boxes) > 0:

            w = frame.shape[1]

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cx = (x1 + x2) // 2

                if cx < w // 3:
                    pos = "Left"
                elif cx < 2 * w // 3:
                    pos = "Center"
                else:
                    pos = "Right"

                text = f"{label} {conf:.2f} {pos}"

                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(output, text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

        # =========================
        # FPS (SAFE)
        # =========================
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(output,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        # =========================
        # Show
        # =========================
        cv2.imshow("YOLOv8 Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Closed cleanly")


if __name__ == "__main__":
    main()