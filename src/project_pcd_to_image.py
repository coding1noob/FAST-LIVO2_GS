#!/usr/bin/env python3
"""Project timestamp-matched PCD points onto camera images."""

import argparse
import csv
import math
import re
import sys
import time
from bisect import bisect_left
from pathlib import Path

try:
    import cv2
    import numpy as np
    import yaml
except ImportError as exc:
    print("Missing dependency: {}".format(exc), file=sys.stderr)
    print("Install OpenCV, NumPy and PyYAML before running.", file=sys.stderr)
    sys.exit(2)

TIMESTAMP_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
METADATA_NAME = "projection_metadata.yaml"


def timestamp(path):
    match = TIMESTAMP_RE.search(path.stem)
    if not match:
        raise ValueError("No numeric timestamp in {}".format(path.name))
    return float(match.group(1))


def vector(values, size, name):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError("{} must contain {} values".format(name, size))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("{} contains non-finite values".format(name))
    return result


def transform_config(data, key):
    values = data.get(key)
    if not isinstance(values, dict):
        raise ValueError("Missing {} in projection configuration".format(key))
    rotation = vector(values.get("rotation"), 9, key + ".rotation").reshape(3, 3)
    translation = vector(values.get("translation"), 3, key + ".translation")
    determinant = np.linalg.det(rotation)
    if abs(determinant - 1.0) > 0.05:
        print("Warning: {} rotation determinant is not close to +1".format(key), file=sys.stderr)
    return rotation, translation


def from_old_config(config):
    camera = config.get("camera", {})
    required = ("width", "height", "fx", "fy", "cx", "cy")
    if any(key not in camera for key in required):
        raise ValueError("Camera configuration lacks required fields: {}".format(", ".join(required)))
    camera_data = {
        "model": "pinhole",
        "output_width": int(camera["width"]),
        "output_height": int(camera["height"]),
        "source_width": int(camera["width"]),
        "source_height": int(camera["height"]),
        "fx": float(camera["fx"]), "fy": float(camera["fy"]),
        "cx": float(camera["cx"]), "cy": float(camera["cy"]),
        "distortion": camera.get("distortion", [0, 0, 0, 0, 0]),
    }
    frame = config.get("pcd_frame", "imu_body")
    transform_key = "imu_to_camera" if frame == "imu_body" else "lidar_to_camera"
    rotation, translation = transform_config(config, transform_key)
    projection = config.get("projection", {})
    return make_internal(camera_data, frame, rotation, translation,
                         config.get("time_offset_seconds", 0.0),
                         config.get("max_time_diff_seconds", 0.05), projection,
                         source="explicit configuration")


def make_internal(camera, frame, rotation, translation, offset, max_diff, projection, source):
    if frame not in ("imu_body", "lidar"):
        raise ValueError("pcd_frame must be imu_body or lidar")
    width = int(camera.get("output_width", camera.get("width")))
    height = int(camera.get("output_height", camera.get("height")))
    if width < 1 or height < 1:
        raise ValueError("Camera output dimensions must be positive")
    distortion = vector(camera.get("distortion", [0, 0, 0, 0, 0]), 5, "camera.distortion")
    fx, fy = float(camera["fx"]), float(camera["fy"])
    if not math.isfinite(fx) or not math.isfinite(fy) or fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be finite and positive")
    intrinsics = np.array([[fx, 0.0, float(camera["cx"])],
                           [0.0, fy, float(camera["cy"])],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
    projection = projection or {}
    return {"width": width, "height": height, "K": intrinsics, "distortion": distortion,
            "frame": frame, "rotation": rotation, "translation": translation,
            "time_offset": float(offset), "max_time_diff": float(max_diff),
            "radius": max(1, int(projection.get("point_radius", 2))),
            "color_by": projection.get("color_by", "depth"),
            "max_points": int(projection.get("max_points", 200000)),
            "source": source}


def from_metadata(config):
    camera = config.get("camera", {})
    if camera.get("available") is False or "fx" not in camera:
        raise ValueError("Projection metadata has no usable camera calibration")
    frame = config.get("extraction", {}).get("pcd_frame",
            config.get("frames", {}).get("pcd_frame", "imu_body"))
    frames = config.get("frames", {})
    key = "imu_to_camera" if frame == "imu_body" else "lidar_to_camera"
    rotation, translation = transform_config(frames, key)
    time_data = config.get("time", {})
    return make_internal(camera, frame, rotation, translation,
                         time_data.get("time_offset_seconds", 0.0),
                         time_data.get("max_time_diff_seconds", 0.05),
                         config.get("projection", {}), METADATA_NAME)


def from_fastlivo_configs(avia_path, camera_path):
    with open(str(avia_path), "r") as stream:
        avia = yaml.safe_load(stream) or {}
    with open(str(camera_path), "r") as stream:
        camera_config = yaml.safe_load(stream) or {}

    extrinsic = avia.get("extrin_calib", {})
    ext_r = vector(extrinsic.get("extrinsic_R"), 9,
                   "extrin_calib.extrinsic_R").reshape(3, 3)
    ext_t = vector(extrinsic.get("extrinsic_T"), 3,
                   "extrin_calib.extrinsic_T")
    rcl = vector(extrinsic.get("Rcl"), 9,
                 "extrin_calib.Rcl").reshape(3, 3)
    pcl = vector(extrinsic.get("Pcl"), 3,
                 "extrin_calib.Pcl")

    # avia.yaml stores LiDAR-to-IMU and LiDAR-to-camera transforms.
    # Convert them to the IMU-body-to-camera transform used by the PCDs.
    imu_to_camera_rotation = rcl.dot(ext_r.T)
    imu_to_camera_translation = pcl - imu_to_camera_rotation.dot(ext_t)

    required_camera = ("cam_width", "cam_height", "cam_fx", "cam_fy",
                       "cam_cx", "cam_cy")
    missing = [key for key in required_camera if key not in camera_config]
    if missing:
        raise ValueError("Camera config lacks required fields: {}".format(
            ", ".join(missing)))
    source_width = int(camera_config["cam_width"])
    source_height = int(camera_config["cam_height"])
    scale = float(camera_config.get("scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("camera scale must be finite and positive")
    camera = {
        "output_width": max(1, int(round(source_width * scale))),
        "output_height": max(1, int(round(source_height * scale))),
        "fx": float(camera_config["cam_fx"]) * scale,
        "fy": float(camera_config["cam_fy"]) * scale,
        "cx": float(camera_config["cam_cx"]) * scale,
        "cy": float(camera_config["cam_cy"]) * scale,
        "distortion": [float(camera_config.get(key, 0.0)) for key in
                       ("cam_d0", "cam_d1", "cam_d2", "cam_d3", "cam_d4")],
    }
    time_config = avia.get("time_offset", {})
    projection = {"color_by": "depth", "point_radius": 2,
                  "max_points": 200000}
    return make_internal(camera, "imu_body", imu_to_camera_rotation,
                         imu_to_camera_translation, 0.0, 0.05, projection,
                         "FAST-LIVO2 configs: {}, {}".format(
                             avia_path, camera_path))


def load_config(path):
    with open(str(path), "r") as stream:
        config = yaml.safe_load(stream) or {}
    if config.get("schema_version") is not None or "frames" in config:
        return from_metadata(config)
    return from_old_config(config)


def find_config(dataset, explicit):
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError("Projection config does not exist: {}".format(explicit))
        return explicit
    candidates = (dataset / METADATA_NAME, dataset / "all_image" / METADATA_NAME,
                  dataset / "all_pcd_body" / METADATA_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("No {} found; pass a projection YAML explicitly".format(METADATA_NAME))


def parse_pcd_header(stream):
    values = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD header is incomplete")
        text = line.decode("ascii", errors="strict").strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        values[parts[0].upper()] = parts[1:]
        if parts[0].upper() == "DATA":
            return values


def pcd_points(path, limit):
    with open(str(path), "rb") as stream:
        header = parse_pcd_header(stream)
        fields = header.get("FIELDS", [])
        sizes = [int(value) for value in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if header.get("DATA", [""])[0].lower() == "binary":
            if any(count != 1 for count in counts) or any(size != 4 or kind != "F" for size, kind in zip(sizes, types)):
                raise ValueError("Only float32 scalar binary PCD fields are supported: {}".format(path))
            point_count = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
            stride = sum(sizes)
            raw = stream.read(point_count * stride)
            if len(raw) != point_count * stride:
                raise ValueError("PCD binary payload is truncated: {}".format(path))
            matrix = np.frombuffer(raw, dtype="<f4").reshape(point_count, len(fields))
        elif header.get("DATA", [""])[0].lower() == "ascii":
            rows = np.loadtxt(stream, dtype=np.float32, ndmin=2)
            matrix = rows[:int(header.get("POINTS", [len(rows)])[0])]
        else:
            raise ValueError("Unsupported PCD DATA encoding: {}".format(header.get("DATA")))
        try:
            x, y, z = (fields.index(name) for name in ("x", "y", "z"))
        except ValueError as exc:
            raise ValueError("PCD lacks x/y/z fields: {}".format(path)) from exc
        intensity_index = fields.index("intensity") if "intensity" in fields else None
        if limit > 0 and len(matrix) > limit:
            matrix = matrix[np.linspace(0, len(matrix) - 1, limit, dtype=np.int64)]
        xyz = matrix[:, [x, y, z]].astype(np.float64, copy=False)
        intensity = (matrix[:, intensity_index].astype(np.float64, copy=False)
                     if intensity_index is not None else np.zeros(len(xyz)))
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
        return xyz[valid], intensity[valid]


def nearest_image(images, image_times, target, max_diff):
    index = bisect_left(image_times, target)
    candidates = []
    if index < len(images):
        candidates.append((abs(image_times[index] - target), images[index]))
    if index > 0:
        candidates.append((abs(image_times[index - 1] - target), images[index - 1]))
    if not candidates:
        return None, None
    difference, path = min(candidates, key=lambda item: item[0])
    return (path, difference) if difference <= max_diff else (None, difference)


def colors(values, color_by):
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    if color_by == "intensity":
        gray = (normalized * 255).astype(np.uint8)
        return np.column_stack((gray, gray, gray))

    # OpenCV colors are BGR. Use a four-stop distance ramp so depth changes
    # remain visible across the full range: near=blue, then green/yellow,
    # far=red.
    stops = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    colors_bgr = np.array([
        [255.0, 0.0, 0.0],
        [0.0, 255.0, 0.0],
        [0.0, 255.0, 255.0],
        [0.0, 0.0, 255.0],
    ])
    return np.column_stack([
        np.interp(normalized, stops, colors_bgr[:, channel])
        for channel in range(3)
    ]).astype(np.uint8)


def progress(processed, total, matched, projected_frames, projected_points, started):
    elapsed = max(time.monotonic() - started, 1e-6)
    fraction = processed / float(total) if total else 1.0
    width = 30
    filled = int(width * fraction)
    rate = processed / elapsed
    remaining = (total - processed) / rate if rate else 0.0
    print("\r[{}] {:6.2f}% PCD {}/{} | matched {} | projected {} | points {} | {:.1f}/s | ETA {}".format(
        "#" * filled + "-" * (width - filled), fraction * 100.0,
        processed, total, matched, projected_frames, projected_points, rate,
        format_duration(remaining)), end="", flush=True)


def format_duration(seconds):
    if seconds <= 0:
        return "--"
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{}h{:02d}m".format(hours, minutes) if hours else (
        "{}m{:02d}s".format(minutes, seconds) if minutes else "{}s".format(seconds))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="directory containing all_image and all_pcd_body")
    parser.add_argument("config", type=Path, nargs="?", help="optional projection calibration YAML")
    parser.add_argument("--avia", type=Path,
                        help="FAST-LIVO2 avia.yaml; overrides projection metadata")
    parser.add_argument("--camera-config", type=Path,
                        help="FAST-LIVO2 camera_pinhole.yaml; requires --avia")
    parser.add_argument("--output", type=Path, help="output directory (default: dataset/projected)")
    parser.add_argument("--max-time-diff", type=float, help="override YAML matching threshold in seconds")
    parser.add_argument("--time-offset", type=float, help="override YAML: image time = PCD time + offset")
    parser.add_argument("--max-points", type=int, help="override YAML point sampling limit")
    parser.add_argument("--point-radius", type=int,
                        help="override projected point radius in pixels (default: YAML value)")
    parser.add_argument("--allow-size-mismatch", action="store_true",
                        help="continue when actual image dimensions differ from calibration")
    parser.add_argument("--overwrite", action="store_true", help="allow an existing output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    if (args.avia is None) != (args.camera_config is None):
        raise ValueError("--avia and --camera-config must be provided together")
    if args.avia is not None:
        avia_path = args.avia.expanduser().resolve()
        camera_path = args.camera_config.expanduser().resolve()
        if not avia_path.is_file():
            raise ValueError("FAST-LIVO2 avia config does not exist: {}".format(avia_path))
        if not camera_path.is_file():
            raise ValueError("FAST-LIVO2 camera config does not exist: {}".format(camera_path))
        config = from_fastlivo_configs(avia_path, camera_path)
        config_path = "{}, {}".format(avia_path, camera_path)
    else:
        config_path = find_config(dataset, args.config.expanduser() if args.config else None)
        config = load_config(config_path)
    max_diff = config["max_time_diff"] if args.max_time_diff is None else args.max_time_diff
    offset = config["time_offset"] if args.time_offset is None else args.time_offset
    max_points = config["max_points"] if args.max_points is None else args.max_points
    point_radius = config["radius"] if args.point_radius is None else args.point_radius
    if not math.isfinite(max_diff) or max_diff < 0:
        raise ValueError("max time difference must be finite and non-negative")
    if point_radius < 1:
        raise ValueError("point radius must be a positive integer")
    image_dir, pcd_dir = dataset / "all_image", dataset / "all_pcd_body"
    if not image_dir.is_dir() or not pcd_dir.is_dir():
        raise ValueError("Expected all_image and all_pcd_body under {}".format(dataset))
    output = (args.output or dataset / "projected").expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError("Output directory is not empty; use --overwrite: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    images = sorted((path for path in image_dir.iterdir()
                     if path.suffix.lower() in (".png", ".jpg", ".jpeg")), key=timestamp)
    pcds = sorted((path for path in pcd_dir.iterdir() if path.suffix.lower() == ".pcd"), key=timestamp)
    if not images or not pcds:
        raise ValueError("No images or PCDs found")
    image_times = [timestamp(path) for path in images]
    expected_size = (config["width"], config["height"])
    print("Using projection config: {}".format(config_path))
    print("PCD frame: {}".format(config["frame"]))
    print("Time matching: image_time = pcd_time + {:.6f}s, max error {:.6f}s".format(offset, max_diff))
    size_mismatches = 0
    image_sizes = {}
    for image in images:
        image_data = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image_data is None:
            raise ValueError("Could not read image: {}".format(image))
        actual_size = (image_data.shape[1], image_data.shape[0])
        image_sizes[image] = actual_size
        if actual_size != expected_size:
            size_mismatches += 1
            message = "Image {} has size {}x{}; calibration declares {}x{}".format(
                image, actual_size[0], actual_size[1], expected_size[0], expected_size[1])
            if not args.allow_size_mismatch:
                raise ValueError(message + "; use --allow-size-mismatch to continue")
    if size_mismatches and args.allow_size_mismatch:
        print("Warning: {} image(s) differ from calibration dimensions; continuing without changing intrinsics.".format(
            size_mismatches), file=sys.stderr)
    matched = projected_frames = projected_points = 0
    started = time.monotonic()
    with open(str(output / "matches.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pcd", "pcd_timestamp", "image", "image_timestamp", "time_error",
                         "points", "projected", "actual_width", "actual_height", "size_status"))
        for index, pcd in enumerate(pcds, 1):
            pcd_time = timestamp(pcd)
            image, error = nearest_image(images, image_times, pcd_time + offset, max_diff)
            if image is None:
                writer.writerow((pcd.name, pcd_time, "", "", "" if error is None else error,
                                 0, 0, "", "", "no_image_match"))
                progress(index, len(pcds), matched, projected_frames, projected_points, started)
                continue
            image_time = timestamp(image)
            image_data = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_data is None:
                raise ValueError("Could not read image: {}".format(image))
            actual_size = image_sizes[image]
            size_status = "ok" if actual_size == expected_size else "mismatch"
            xyz, intensity = pcd_points(pcd, max_points)
            camera_xyz = (config["rotation"].dot(xyz.T) + config["translation"][:, None]).T
            valid_depth = camera_xyz[:, 2] > 0.0
            projected, _ = cv2.projectPoints(camera_xyz[valid_depth].reshape(-1, 1, 3),
                                             np.zeros(3), np.zeros(3), config["K"], config["distortion"])
            pixels = projected.reshape(-1, 2)
            valid_image = ((pixels[:, 0] >= 0) & (pixels[:, 0] < actual_size[0]) &
                           (pixels[:, 1] >= 0) & (pixels[:, 1] < actual_size[1]))
            visible = pixels[valid_image]
            visible_values = (camera_xyz[valid_depth][valid_image, 2]
                              if config["color_by"] == "depth" else intensity[valid_depth][valid_image])
            for (u, v), color in zip(visible, colors(visible_values, config["color_by"])):
                cv2.circle(image_data, (int(round(u)), int(round(v))), point_radius,
                           tuple(int(value) for value in color), -1)
            output_image = output / image.name
            if not cv2.imwrite(str(output_image), image_data):
                raise IOError("Could not write {}".format(output_image))
            matched += 1
            projected_frames += 1 if len(visible) else 0
            projected_points += len(visible)
            writer.writerow((pcd.name, pcd_time, image.name, image_time,
                             abs(image_time - (pcd_time + offset)), len(xyz), len(visible),
                             actual_size[0], actual_size[1], size_status))
            progress(index, len(pcds), matched, projected_frames, projected_points, started)
    print()
    print("Projected {} of {} PCD files; {} visible points; size mismatches {}; results: {}".format(
        matched, len(pcds), projected_points, size_mismatches, output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (IOError, OSError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
