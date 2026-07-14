import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

CATEGORIES = [
    "fall",
    "grab",
    "gun",
    "hit",
    "kick",
    "lying_down",
    "run",
    "sit",
    "sneak",
    "stand",
    "struggle",
    "throw",
    "walk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi gerakan CCTV real-time dari kamera laptop."
    )
    parser.add_argument("--camera", type=int, default=0, help="Index kamera.")
    parser.add_argument(
        "--min-area",
        type=int,
        default=5000,
        help="Luas contour minimum agar dianggap gerakan.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=25,
        help="Threshold biner untuk membedakan frame saat ini dengan background.",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=21,
        help="Ukuran kernel Gaussian blur. Gunakan angka ganjil.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Jeda minimum antar penyimpanan snapshot gerakan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures"),
        help="Folder untuk menyimpan snapshot saat gerakan terdeteksi.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("activity_model.keras"),
        help="Path model TensorFlow (.keras/.h5) jika ingin klasifikasi aktivitas.",
    )
    parser.add_argument(
        "--prediction-window",
        type=int,
        default=5,
        help="Jumlah prediksi terakhir untuk dirata-ratakan.",
    )
    return parser.parse_args()


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def preprocess_for_model(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def load_activity_model(model_path: Path):
    if model_path is None:
        return None
    if tf is None:
        raise ImportError(
            "TensorFlow belum terpasang. Install tensorflow jika ingin memakai model."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_activity(model, frame: np.ndarray, history: deque) -> tuple[str, float] | tuple[None, None]:
    if model is None:
        return None, None

    input_tensor = preprocess_for_model(frame)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    history.append(probabilities)
    averaged_probabilities = np.mean(np.array(history), axis=0)
    best_index = int(np.argmax(averaged_probabilities))
    return CATEGORIES[best_index], float(averaged_probabilities[best_index])


def draw_status(
    frame: np.ndarray,
    motion_detected: bool,
    activity_label: str | None,
    activity_score: float | None,
) -> None:
    status_text = "STATUS: MOTION DETECTED" if motion_detected else "STATUS: IDLE"
    status_color = (0, 255, 0) if motion_detected else (0, 200, 255)
    cv2.putText(
        frame,
        status_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    if activity_label is not None and activity_score is not None:
        cv2.putText(
            frame,
            f"ACTIVITY: {activity_label} ({activity_score:.2f})",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )


def main() -> None:
    args = parse_args()
    blur_size = ensure_odd(args.blur)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_activity_model(args.model_path)
    prediction_history: deque = deque(maxlen=max(1, args.prediction_window))

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Kamera dengan index {args.camera} tidak bisa dibuka.")

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    baseline_frame = None
    last_capture_time = 0.0

    print("Tekan 'q' untuk keluar.")
    print("Tekan 'r' untuk reset background.")

    while True:
        detected_input, frame = camera.read()
        if not detected_input:
            print("Frame dari kamera tidak terbaca.")
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (blur_size, blur_size), 0)

        if baseline_frame is None:
            baseline_frame = grayscale
            cv2.putText(
                frame,
                "Mengambil background awal...",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("CCTV Motion Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        frame_delta = cv2.absdiff(baseline_frame, grayscale)
        _, threshold_frame = cv2.threshold(
            frame_delta, args.threshold, 255, cv2.THRESH_BINARY
        )
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

        contours, _ = cv2.findContours(
            threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < args.min_area:
                continue

            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        activity_label, activity_score = predict_activity(model, frame, prediction_history)
        draw_status(frame, motion_detected, activity_label, activity_score)

        if motion_detected and time.time() - last_capture_time >= args.cooldown:
            filename = args.output_dir / f"motion_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), frame)
            last_capture_time = time.time()

        cv2.imshow("CCTV Motion Detector", frame)
        cv2.imshow("Motion Mask", threshold_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            baseline_frame = grayscale
            prediction_history.clear()
            print("Background berhasil di-reset.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
