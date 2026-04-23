import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

pipeline.start(config)

# Fetch IR intrinsics and stereo baseline for SGBM depth conversion
profile = pipeline.get_active_profile()
ir1 = profile.get_stream(rs.stream.infrared, 1)
ir2 = profile.get_stream(rs.stream.infrared, 2)
ir_intrinsics = ir1.as_video_stream_profile().get_intrinsics()
fx = ir_intrinsics.fx
baseline = abs(ir1.get_extrinsics_to(ir2).translation[0])  # metres

face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="blaze_face_full_range.tflite"),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5,
)


def put_text_with_bg(image, text, pos, font_scale, text_color, bg_color=(0, 0, 0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(image, (x, y - th - baseline), (x + tw, y + baseline), bg_color, cv2.FILLED)
    cv2.putText(image, text, pos, font, font_scale, text_color, thickness)


def detect_faces(image, detector):
    """Run face detection and draw boxes on image. Returns list of (x,y,w,h,score)."""
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )
    results = detector.detect(mp_image)
    detections = []
    for detection in results.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        score = detection.categories[0].score
        detections.append((x, y, w, h, score))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # put_text_with_bg(image, f"Face {score:.2f}", (x, y - 4), 0.6, (0, 255, 0), thickness=2)
        put_text_with_bg(image, f"Face", (x, y - 4), 0.6, (0, 255, 0), thickness=2)
    return detections


def draw_detections_with_depth(image, detections, get_depth_fn, unit, color=(0, 255, 0), x_offset_factor=0.2):
    """Overlay face bounding boxes and depth measurements onto a depth frame."""
    for (x, y, w, h, _) in detections:
        x = x + int(x_offset_factor * w)
        cx = min(x + w // 2, image.shape[1] - 1)
        cy = min(y + h // 2, image.shape[0] - 1)
        depth = get_depth_fn(cx, cy)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if depth > 0:
            put_text_with_bg(image, f"{depth:.2f}{unit}", (x, y - 4), 0.6, color, thickness=2)


def add_label(image, text):
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


def compute_sad_depth(ir_left, ir_right):
    """Disparity via Sum of Absolute Differences (SAD) block matching, implemented in numpy."""
    block_size = 15
    num_disparities = 128
    half_b = block_size // 2

    h, w = ir_left.shape
    left = ir_left.astype(np.float32)
    right = ir_right.astype(np.float32)

    min_sad = np.full((h, w), np.inf, dtype=np.float32)
    disparity = np.zeros((h, w), dtype=np.float32)

    for d in range(num_disparities):
        # Shift right image rightward by d pixels: pixel (y, x) in left matches (y, x-d) in right
        shifted_right = np.zeros_like(right)
        shifted_right[:, d:] = right[:, :w - d]

        abs_diff = np.abs(left - shifted_right)

        # Sum abs_diff over block_size x block_size blocks using the cumulative sum trick
        row_cs = np.cumsum(abs_diff, axis=0)
        row_sums = row_cs[block_size - 1:] - np.concatenate(
            [np.zeros((1, w), dtype=np.float32), row_cs[:-block_size]], axis=0
        )  # shape: (h - block_size + 1, w)

        col_cs = np.cumsum(row_sums, axis=1)
        block_sad = col_cs[:, block_size - 1:] - np.concatenate(
            [np.zeros((h - block_size + 1, 1), dtype=np.float32), col_cs[:, :-block_size]], axis=1
        )  # shape: (h - block_size + 1, w - block_size + 1)
        # block_sad[i, j] = SAD for block whose top-left is (i, j), center at (i+half_b, j+half_b)

        sad_map = np.full((h, w), np.inf, dtype=np.float32)
        sad_map[half_b:h - half_b, half_b:w - half_b] = block_sad

        update = sad_map < min_sad
        min_sad[update] = sad_map[update]
        disparity[update] = float(d)

    colormap = cv2.applyColorMap(
        255 - cv2.convertScaleAbs(disparity, alpha=255.0 / num_disparities), cv2.COLORMAP_JET
    )
    return colormap, disparity


def compute_stereo_depth(ir_left, ir_right):
    """Returns (colormap, raw_disparity) where disparity is in pixels (float32)."""
    block_size = 11
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(ir_left, ir_right).astype(np.float32) / 16.0
    colormap = cv2.applyColorMap(
        255 - cv2.convertScaleAbs(disparity, alpha=255.0 / 128.0), cv2.COLORMAP_JET
    )
    return colormap, disparity


cv2.namedWindow("Stream Grid", cv2.WINDOW_AUTOSIZE)

with FaceDetector.create_from_options(face_detector_options) as detector:
    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)

            if not color_frame or not depth_frame or not ir_left_frame or not ir_right_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data()).copy()

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            ir_left = np.asanyarray(ir_left_frame.get_data())
            ir_right = np.asanyarray(ir_right_frame.get_data())
            ir_left_bgr = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2BGR)
            ir_right_bgr = cv2.cvtColor(ir_right, cv2.COLOR_GRAY2BGR)

            stereo_colormap, stereo_disparity = compute_stereo_depth(ir_left, ir_right)
            sad_colormap, sad_disparity = compute_sad_depth(ir_left, ir_right)

            face_detections = detect_faces(color_image, detector)

            draw_detections_with_depth(
                depth_colormap,
                face_detections,
                lambda cx, cy: depth_frame.get_distance(cx, cy),
                "m",
            )

            draw_detections_with_depth(
                stereo_colormap,
                face_detections,
                lambda cx, cy: (fx * baseline / stereo_disparity[cy, cx])
                if stereo_disparity[cy, cx] > 0 else 0,
                "m",
            )

            draw_detections_with_depth(
                sad_colormap,
                face_detections,
                lambda cx, cy: (fx * baseline / sad_disparity[cy, cx])
                if sad_disparity[cy, cx] > 0 else 0,
                "m",
            )

            add_label(color_image, "RGB")
            add_label(depth_colormap, "Intel Realsense Depth")
            add_label(stereo_colormap, "Stereo SGBM Depth")
            add_label(ir_left_bgr, "IR Left")
            add_label(ir_right_bgr, "IR Right")
            add_label(sad_colormap, "SAD Depth")

            top_row = np.hstack((color_image, depth_colormap, stereo_colormap))
            bottom_row = np.hstack((ir_left_bgr, ir_right_bgr, sad_colormap))
            combined = np.vstack((top_row, bottom_row))

            cv2.imshow("Stream Grid", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("Stream Grid", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
