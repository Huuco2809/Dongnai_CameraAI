from __future__ import annotations

import os
import time
import cv2
import pygame
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO


# -------------------------------------------------
# Tối ưu bắt RTSP qua FFMPEG + TCP, rút timeout treo
# -------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|buffer_size;102400"


# ========================
# Tiện ích chung
# ========================

def load_cfg() -> Dict:
    """Tìm và đọc realtime_infer/config.yaml (hoặc ./config.yaml nếu chạy trong realtime_infer)."""
    THIS = Path(__file__).resolve()
    REALTIME = THIS.parent
    candidates = [REALTIME / "config.yaml", Path("./config.yaml")]
    for p in candidates:
        if p.exists():
            print(f"[INFO] Using config: {p}", flush=True)
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Không tìm thấy config.yaml trong realtime_infer.")


def ensure_dir(d: Path | str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def norm_label_for_audio(label: str) -> str:
    """Chuyển label sang dạng tên file âm thanh: 'motor bike' -> 'motor_bike'."""
    return label.strip().lower().replace(" ", "_")


def play_sound(label: str, sound_root: Path) -> None:
    """
    Phát âm thanh theo tên label (am_thanh/default/<label>.mp3).
    Test từ RTSP không phân camera, nên dùng 'default'.
    """
    try:
        mp3 = sound_root / "default" / f"{norm_label_for_audio(label)}.mp3"
        if not mp3.exists():
            print(f"[WARN] Không tìm thấy âm thanh: {mp3}", flush=True)
            return
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            print("[INFO] pygame mixer initialized", flush=True)
        pygame.mixer.music.load(str(mp3))
        pygame.mixer.music.play()
        print(f"[SOUND] {label} -> {mp3.name}", flush=True)
    except Exception as e:
        print(f"[WARN] play_sound error: {e}", flush=True)


def maybe_resize_for_window(img, max_w: int):
    """Resize ảnh để hiển thị nếu vượt quá max_w (0 = giữ nguyên)."""
    if max_w and max_w > 0 and img.shape[1] > max_w:
        scale = max_w / float(img.shape[1])
        new_h = int(img.shape[0] * scale)
        return cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# ========================
# Helper xử lý lớp / ngưỡng
# ========================

def build_class_maps(model) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Tạo 2 map: id->name và name->id từ model.names (Ultralytics)."""
    names_obj = model.names
    if isinstance(names_obj, dict):
        id2name = {int(k): str(v) for k, v in names_obj.items()}
    elif isinstance(names_obj, list):
        id2name = {i: str(n) for i, n in enumerate(names_obj)}
    else:
        id2name = {int(k): str(v) for k, v in dict(names_obj).items()}
    name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id


def get_effective_threshold(cls_id: int, label: str, cfg: Dict, default_conf: float) -> float:
    """
    Ngưỡng ưu tiên: class_thresholds[label] -> class_thresholds[str(cls_id)] -> default_conf.
    """
    cth: Dict = cfg.get("class_thresholds", {}) or {}
    if label in cth:
        try:
            return float(cth[label])
        except Exception:
            pass
    sid = str(cls_id)
    if sid in cth:
        try:
            return float(cth[sid])
        except Exception:
            pass
    return float(default_conf)


def is_enabled(cls_id: int, enabled: Optional[List[int]]) -> bool:
    """Nếu enabled=None hoặc rỗng => bật tất cả."""
    if not enabled:
        return True
    return cls_id in enabled


# ========================
# Xử lý một camera RTSP
# ========================

def process_camera_stream(
    cam_cfg: Dict,
    global_cfg: Dict,
    model: YOLO,
    id2name: Dict[int, str],
    out_dir: Path,
    default_show_window: bool,
    default_draw_boxes: bool,
    default_window_max_width: int,
    show_wait_ms: int,
    sound_root: Path,
    global_conf: float,
) -> bool:
    """
    Trả về True nếu người dùng nhấn 'q' (thoát toàn bộ). 'n' để chuyển camera kế.
    """
    name = cam_cfg.get("name", "camera")
    src = cam_cfg.get("src")
    if not src:
        print(f"[WARN] Bỏ qua camera '{name}' vì thiếu 'src'", flush=True)
        return False

    cam_conf = float(cam_cfg.get("confidence", global_conf))
    cam_cooldown = float(cam_cfg.get("cooldown_sec", global_cfg.get("cooldown_sec", 3.0)))
    enabled_classes = cam_cfg.get("enabled_classes", None)
    if isinstance(enabled_classes, list):
        enabled_classes = [int(x) for x in enabled_classes]

    cam_show_window = bool(cam_cfg.get("show_window", default_show_window))
    cam_draw_boxes = bool(cam_cfg.get("draw_boxes", default_draw_boxes))
    cam_window_max_width = int(cam_cfg.get("window_max_width", default_window_max_width))

    print(f"\n[INFO] === MỞ CAMERA: {name} ===", flush=True)
    print(f"[INFO] src={src}", flush=True)
    print(f"[INFO] confidence={cam_conf}, cooldown_sec={cam_cooldown}, enabled={enabled_classes if enabled_classes else 'ALL'}", flush=True)
    print(f"[INFO] show_window={cam_show_window}, draw_boxes={cam_draw_boxes}, max_w={cam_window_max_width}", flush=True)

    if cam_show_window:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được RTSP cho camera '{name}'", flush=True)
        if cam_show_window:
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
        return False

    last_alert_ts = 0.0  # cooldown phát âm thanh
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # tránh busy loop khi rớt khung hình
                cv2.waitKey(100)
                continue

            frame_idx += 1
            raw = frame.copy()   # ẢNH GỐC (không vẽ gì) -> để lưu làm dữ liệu train
            vis = frame.copy()   # Ảnh để vẽ bbox nhằm hiển thị

            # Nhận diện sơ bộ theo cam_conf; sau đó lọc theo class_thresholds
            results = model(vis, conf=cam_conf)
            dets = results[0].boxes

            had_valid = False
            best_label_for_sound: Optional[str] = None  # phát 1 lần/khung

            for box in dets:
                cls_id = int(box.cls[0])
                if not is_enabled(cls_id, enabled_classes):
                    continue

                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                label = id2name.get(cls_id, str(cls_id))

                eff_thres = get_effective_threshold(cls_id, label, global_cfg, cam_conf)
                # Log ngắn gọn để debug
                print(f"[{name}] {label} conf={conf:.2f} thr={eff_thres:.2f}", flush=True)

                if conf >= eff_thres:
                    had_valid = True
                    if cam_draw_boxes:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            vis, f"{label} {conf:.2f}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                        )
                    if best_label_for_sound is None:
                        best_label_for_sound = label

            # Lưu ảnh nếu có phát hiện hợp lệ
            if had_valid:
                ts = int(time.time())

                # Lưu ẢNH GỐC (không bbox/nhãn) để train
                raw_dir = out_dir / "raw"
                ensure_dir(raw_dir)
                raw_path = raw_dir / f"{name}_{ts}_{frame_idx}.jpg"
                cv2.imwrite(str(raw_path), raw)
                print(f"[INFO] Lưu ảnh gốc (train): {raw_path}", flush=True)

                # (Tùy chọn) nếu muốn lưu thêm ảnh đã vẽ bbox để tham khảo, bỏ comment 3 dòng dưới:
                # vis_dir = out_dir / "vis"
                # ensure_dir(vis_dir)
                # cv2.imwrite(str(vis_dir / f"{name}_{ts}_{frame_idx}.jpg"), vis)

                # Cooldown phát âm thanh
                now = time.time()
                if best_label_for_sound and (now - last_alert_ts) >= cam_cooldown:
                    play_sound(best_label_for_sound, sound_root)
                    last_alert_ts = now

            # Hiển thị
            if cam_show_window:
                show_img = maybe_resize_for_window(vis, cam_window_max_width)
                cv2.imshow(name, show_img)
                key = cv2.waitKey(show_wait_ms) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit by user (q).", flush=True)
                    return True  # thoát toàn bộ
                if key == ord('n'):
                    print("[INFO] Next camera (n).", flush=True)
                    break       # sang camera kế

    finally:
        cap.release()
        if cam_show_window:
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass

    return False


# ========================
# Main
# ========================

def main():
    # 1) Đường dẫn chính
    THIS = Path(__file__).resolve()           # realtime_infer/test_images.py
    REALTIME = THIS.parent                    # realtime_infer

    # 2) Load config
    cfg = load_cfg()

    # 3) Load YOLO
    model_path = Path(cfg.get("model_path", "./models/best.pt"))
    if not model_path.is_absolute():
        model_path = REALTIME / model_path
    print(f"[INFO] Loading model: {model_path}", flush=True)
    model = YOLO(str(model_path))

    # 4) Bản đồ class id <-> name
    id2name, _ = build_class_maps(model)
    print("[INFO] Classes mapping:", id2name, flush=True)

    # 5) Thư mục lưu ảnh kết quả
    out_dir = REALTIME / "runs_rtsp"
    ensure_dir(out_dir)

    # 6) Tham số global
    global_conf = float(cfg.get("confidence", 0.5))
    default_show_window = bool(cfg.get("show_window", True))
    default_draw_boxes = bool(cfg.get("draw_boxes", True))
    default_window_max_width = int(cfg.get("window_max_width", 960))
    show_wait_ms = int(cfg.get("show_wait_ms", 10))
    sound_root = REALTIME / cfg.get("sound_root", "am_thanh")

    print(f"[INFO] Global confidence = {global_conf}", flush=True)
    print(f"[INFO] Sound root: {sound_root}", flush=True)
    print(f"[INFO] Default show_window={default_show_window}, max_w={default_window_max_width}, wait_ms={show_wait_ms}", flush=True)

    # 7) Danh sách camera
    cameras = cfg.get("cameras", [])
    if not cameras:
        raise ValueError("Không có 'cameras' trong config.yaml.")

    # 8) Lần lượt chạy từng camera
    for cam_cfg in cameras:
        quit_all = process_camera_stream(
            cam_cfg=cam_cfg,
            global_cfg=cfg,
            model=model,
            id2name=id2name,
            out_dir=out_dir,
            default_show_window=default_show_window,
            default_draw_boxes=default_draw_boxes,
            default_window_max_width=default_window_max_width,
            show_wait_ms=show_wait_ms,
            sound_root=sound_root,
            global_conf=global_conf,
        )
        if quit_all:
            break

    # 9) Dọn dẹp
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
