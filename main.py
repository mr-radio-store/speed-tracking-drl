import cv2
import time
import numpy as np
from vision_engine import VisionEngine
from spatial_analyzer import SpatialAnalyzer


# =========================================================
# DRL STATE BUILDER
# =========================================================
class DRLStateBuilder:
    """Convert YOLO detections into RL state vector."""

    def build_state(self, detections, confidences, frame_width):

        if len(detections) == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        best_idx = int(np.argmax(confidences))
        x1, y1, x2, y2 = detections[best_idx]
        conf = confidences[best_idx]

        cx = (x1 + x2) / 2.0
        cx_norm = cx / frame_width

        return np.array([cx_norm, float(conf), 1.0], dtype=np.float32)


# =========================================================
# SIMPLE DRL POLICY (RULE-BASED PLACEHOLDER)
# =========================================================
class SimpleDRLAgent:

    def act(self, state):

        if state[2] == 0.0:
            return 1  # default FORWARD

        cx_norm = state[0]

        if cx_norm < 0.33:
            return 0  # LEFT
        elif cx_norm < 0.66:
            return 1  # FORWARD
        else:
            return 2  # RIGHT


# =========================================================
# MAIN SYSTEM
# =========================================================
class ObjectDetectionDRLApp:

    def __init__(self, camera_index=0, width=640, height=480):

        self.engine = VisionEngine()
        self.spatial = SpatialAnalyzer(width)

        self.state_builder = DRLStateBuilder()
        self.agent = SimpleDRLAgent()

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.prev_time = time.time()

        # Speed tracking
        self.pixel_to_meter_ratio = 0.0025
        self.tracking_history = {}  # label -> (cx, cy, t, speed)

    # -----------------------------------------------------
    def run(self):

        if not self.cap.isOpened():
            print("❌ Camera not opened")
            return

        print("✅ DRL Vision System Running")

        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            now = time.time()
            frame_width = frame.shape[1]
            output = frame.copy()

            # =================================================
            # YOLO INFERENCE
            # =================================================
            results = self.engine.process_frame(frame)

            detections = []
            confidences = []
            labels = []
            speeds = []

            # -------------------------------------------------
            # Parse detections
            # -------------------------------------------------
            for r in results:

                if len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, clss):

                    label = self.engine.model.names[cls_id]

                    cx = (box[0] + box[2]) // 2
                    cy = (box[1] + box[3]) // 2

                    # -------------------------------------------------
                    # SPEED ESTIMATION
                    # -------------------------------------------------
                    speed = 0.0

                    if label in self.tracking_history:

                        px, py, pt, pspeed = self.tracking_history[label]
                        dt = now - pt

                        if dt > 0.001:

                            dist_px = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                            dist_m = dist_px * self.pixel_to_meter_ratio
                            raw_speed = dist_m / dt

                            speed = (pspeed * 0.8) + (raw_speed * 0.2)

                        else:
                            speed = pspeed

                    self.tracking_history[label] = (cx, cy, now, speed)

                    # store outputs
                    detections.append(box)
                    confidences.append(float(conf))
                    labels.append(label)
                    speeds.append(speed)

                    # draw object
                    cv2.rectangle(output, (box[0], box[1]),
                                  (box[2], box[3]), (0, 255, 255), 2)

                    cv2.putText(output,
                                f"{label} {conf*100:.0f}% | {speed:.2f} m/s",
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2)

            # cleanup stale objects
            active = set(labels)
            self.tracking_history = {
                k: v for k, v in self.tracking_history.items()
                if k in active
            }

            # =================================================
            # DRL STATE + ACTION
            # =================================================
            state = self.state_builder.build_state(
                detections,
                confidences,
                frame_width
            )

            action = self.agent.act(state)

            # =================================================
            # ACTION DISPLAY
            # =================================================
            if action == 0:
                text, color = "ACTION: TURN LEFT", (0, 0, 255)
            elif action == 1:
                text, color = "ACTION: FORWARD", (0, 255, 0)
            else:
                text, color = "ACTION: TURN RIGHT", (255, 0, 0)

            cv2.putText(output, text,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            # =================================================
            # TARGET DISPLAY (BEST OBJECT)
            # =================================================
            if len(detections) > 0:

                best = int(np.argmax(confidences))

                bx1, by1, bx2, by2 = detections[best]
                blabel = labels[best]
                bconf = confidences[best]
                bspeed = speeds[best]

                cv2.rectangle(output, (bx1, by1),
                              (bx2, by2), color, 3)

                target_text = f"TARGET: {blabel.upper()} ({bconf*100:.0f}%)"
                speed_text = f"SPEED: {bspeed:.2f} m/s"

                cv2.putText(output, target_text,
                            (bx1, by1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

                # shifted right for clarity
                cv2.putText(output, speed_text,
                            (bx1 + 150, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2)

            # =================================================
            # FPS
            # =================================================
            fps = 1 / max(now - self.prev_time, 1e-6)
            self.prev_time = now

            cv2.putText(output, f"FPS: {fps:.1f}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)

            cv2.imshow("DRL YOLO System", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =========================================================
if __name__ == "__main__":
    app = ObjectDetectionDRLApp(camera_index=0)
    app.run()
