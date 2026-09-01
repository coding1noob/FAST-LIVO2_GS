#!/usr/bin/env python3
"""Run FAST-LIVO2 in LIVO mode and export timestamp-associated images and dewarped PCDs.

FAST-LIVO2 performs the image/LiDAR/IMU synchronization, LIO odometry, and
per-point LiDAR undistortion.  The output PCDs are single-scan IMU-body-frame
clouds and the PNGs are the color frames used by FAST-LIVO2.  If processing
ends with only trailing PCDs after an otherwise matching image/PCD sequence,
the trailing PCDs are discarded during final validation.
"""

import argparse
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to generate projection_metadata.yaml: {}".format(exc))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS1 bag containing LiDAR, IMU, and camera data")
    parser.add_argument("--config", type=Path, required=True, help="FAST-LIVO2 YAML")
    parser.add_argument("--camera-config", type=Path, required=True, help="FAST-LIVO2 camera YAML")
    parser.add_argument("--output", type=Path, required=True, help="dataset root to create")
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--imu-topic", default="/livox/imu")
    parser.add_argument("--image-topic", default="/left_camera/image/compressed",
                        help="bag image topic (compressed or raw)")
    parser.add_argument("--raw-image-topic", default="/left_camera/image",
                        help="raw topic consumed by FAST-LIVO2")
    parser.add_argument("--image-transport", choices=("compressed", "raw"), default=None,
                        help="transport of --image-topic (default: inferred from suffix)")
    parser.add_argument("--interval", type=int, default=1,
                        help="number of LIO scans per PCD (1 gives one scan per image)")
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--img-time-offset", type=float, default=0.0,
                        help="image timestamp offset applied by FAST-LIVO2 (seconds; default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing output directory")
    return parser.parse_args()


def scalar(data, key, default=None):
    value = data.get(key, default) if isinstance(data, dict) else default
    return value


def vector(data, key, size, default=None):
    value = scalar(data, key, default)
    if value is None or len(value) != size:
        return None
    return [float(item) for item in value]


def transform(rotation, translation):
    if rotation is None or translation is None:
        return None
    return {"rotation": rotation, "translation": translation}


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matvec(a, v):
    return [sum(a[i][j] * v[j] for j in range(3)) for i in range(3)]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def parse_metadata(config_path, camera_path, args, root):
    with config_path.open("r") as stream:
        main = yaml.safe_load(stream) or {}
    with camera_path.open("r") as stream:
        camera_config = yaml.safe_load(stream) or {}
    extrinsic = main.get("extrin_calib", {})
    ext_r = vector(extrinsic, "extrinsic_R", 9)
    ext_t = vector(extrinsic, "extrinsic_T", 3)
    rcl = vector(extrinsic, "Rcl", 9)
    pcl = vector(extrinsic, "Pcl", 3)
    ext_matrix = [ext_r[0:3], ext_r[3:6], ext_r[6:9]] if ext_r else None
    rcl_matrix = [rcl[0:3], rcl[3:6], rcl[6:9]] if rcl else None
    lidar_to_imu = transform(ext_r, ext_t)
    lidar_to_camera = transform(rcl, pcl)
    imu_to_camera = None
    if ext_matrix is not None and ext_t is not None and rcl_matrix is not None and pcl is not None:
        rci = matmul(rcl_matrix, transpose(ext_matrix))
        tci = add(pcl, [-x for x in matvec(rci, ext_t)])
        imu_to_camera = transform([item for row in rci for item in row], tci)

    width = int(camera_config.get("cam_width", 0))
    height = int(camera_config.get("cam_height", 0))
    scale = float(camera_config.get("scale", 1.0))
    if width < 1 or height < 1:
        raise ValueError("camera config must define positive cam_width/cam_height")
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    camera = {
        "available": True,
        "model": "pinhole",
        "source_width": width,
        "source_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "scale_in_fastlivo": scale,
        "intrinsics_coordinate_system": "processed_image_pixels",
        "fx": float(camera_config["cam_fx"]) * scale,
        "fy": float(camera_config["cam_fy"]) * scale,
        "cx": float(camera_config["cam_cx"]) * scale,
        "cy": float(camera_config["cam_cy"]) * scale,
        "distortion": [float(camera_config.get(key, 0.0)) for key in
                       ("cam_d0", "cam_d1", "cam_d2", "cam_d3", "cam_d4")],
        "source_file": str(camera_path.resolve()),
    }
    time_config = main.get("time_offset", {})
    lidar_offset = float(time_config.get("lidar_time_offset", 0.0))
    image_offset = float(args.img_time_offset)
    lidar_type = "avia" if int(main.get("preprocess", {}).get("lidar_type", 1)) == 1 else "pointcloud2"
    frame = "imu_body"
    metadata = {
        "schema_version": 1,
        "calibration_available": imu_to_camera is not None and camera["available"],
        "source": {"fastlivo_config": str(config_path.resolve()),
                   "camera_config": str(camera_path.resolve())},
        "topics": {"image": args.image_topic, "raw_image": args.raw_image_topic,
                   "lidar": args.lidar_topic, "imu": args.imu_topic,
                   "lidar_type": lidar_type},
        "extraction": {
            "image_interval": args.interval, "pcd_interval": args.interval,
            "pcd_frame": frame, "no_extrinsic": False,
            "images_resized": scale != 1.0,
            "pipeline": "FAST-LIVO2 LIVO; IMU point-time undistortion and LIO odometry",
        },
        "camera": camera,
        "frames": {
            "pcd_frame": frame, "lidar_to_imu": lidar_to_imu,
            "lidar_to_camera": lidar_to_camera, "imu_to_camera": imu_to_camera,
            "equations": {"lidar_to_imu": "p_imu = R*p_lidar + t",
                          "lidar_to_camera": "p_camera = R*p_lidar + t",
                          "imu_to_camera": "p_camera = R*p_imu + t"},
        },
        "time": {
            "lidar_time_offset_seconds": lidar_offset,
            "image_time_offset_seconds": image_offset,
            # FAST-LIVO2 names both files with the synchronized reference time;
            # sensor offsets are already applied during synchronization.
            "time_offset_seconds": 0.0,
            "image_time_from_pcd": "pcd_timestamp + time_offset_seconds",
            "max_time_diff_seconds": 0.05,
        },
        "projection": {"point_radius": 2, "color_by": "depth", "max_points": 200000},
        "provenance": {
            "pcd": "FAST-LIVO2 ImuProcess::UndistortPcl output, saved in IMU body frame",
            "odometry": "FAST-LIVO2 LIO state used during processing",
            "image": "FAST-LIVO2 VIO color frame (camera-model scaled resolution)",
            "poses_written": False,
            "dataset_root": str(root),
        },
    }
    return metadata


def write_metadata(metadata, root, image_dir, pcd_dir):
    text = yaml.safe_dump(metadata, sort_keys=False, default_flow_style=False)
    for path in (root / "projection_metadata.yaml",
                 image_dir / "projection_metadata.yaml",
                 pcd_dir / "projection_metadata.yaml"):
        path.write_text(text)


def launch_text(args, image_dir, pcd_dir):
    compressed = args.image_transport == "compressed"
    image_input_topic = args.image_topic[:-len("/compressed")] if compressed else args.image_topic
    republish = "" if not compressed else '''
  <node pkg="image_transport" type="republish" name="dewarp_image_republish"
        args="compressed in:={image_topic} raw out:={raw_topic}" output="screen" />'''.format(
            image_topic=image_input_topic, raw_topic=args.raw_image_topic)
    return """<launch>
  <rosparam command="load" file="{config}" />
  <param name="common/img_en" value="1" />
  <param name="common/lidar_en" value="1" />
  <param name="common/lid_topic" value="{lidar_topic}" />
  <param name="common/imu_topic" value="{imu_topic}" />
  <param name="common/img_topic" value="{raw_topic}" />
  <param name="time_offset/img_time_offset" value="{img_time_offset}" />
  <param name="imu/imu_en" value="true" />
  <param name="pcd_save/pcd_save_en" value="true" />
  <param name="pcd_save/interval" value="{interval}" />
  <param name="pcd_save/type" value="1" />
  <param name="pcd_save/colmap_output_en" value="false" />
  <param name="image_save/img_save_en" value="true" />
  <param name="image_save/interval" value="{interval}" />
  <param name="output/pcd_dir" value="{pcd_dir}" />
  <param name="output/image_dir" value="{image_dir}" />
  <param name="output/pcd_pose_output_en" value="false" />
  <param name="output/image_pose_output_en" value="false" />{republish}
  <node pkg="fast_livo" type="fastlivo_mapping" name="laserMapping" output="screen">
    <rosparam file="{camera_config}" />
  </node>
</launch>
""".format(config=str(args.config.resolve()), camera_config=str(args.camera_config.resolve()),
           lidar_topic=args.lidar_topic, imu_topic=args.imu_topic, raw_topic=args.raw_image_topic,
           img_time_offset=args.img_time_offset, interval=args.interval,
           image_dir=str(image_dir), pcd_dir=str(pcd_dir), republish=republish)


def playback_command(args):
    # Bags can mix absolute and relative topic spellings. Include both forms;
    # rosbag matches the stored connection name exactly.
    topics = []
    for topic in (args.lidar_topic, args.imu_topic, args.image_topic):
        for candidate in (topic, topic.lstrip("/")):
            if candidate and candidate not in topics:
                topics.append(candidate)
    command = ["rosbag", "play", str(args.bag.resolve()), "--rate", str(args.rate),
               "--topics"] + topics
    if args.start is not None:
        command.extend(["--start", str(args.start)])
    if args.duration is not None:
        command.extend(["--duration", str(args.duration)])
    return command


def discard_unpaired_trailing_pcds(image_dir, pcd_dir):
    images = sorted(image_dir.glob("*.png"), key=lambda path: float(path.stem))
    pcds = sorted(pcd_dir.glob("*.pcd"), key=lambda path: float(path.stem))
    paired_count = min(len(images), len(pcds))
    matching_prefix = all(images[index].stem == pcds[index].stem
                          for index in range(paired_count))
    trailing = pcds[paired_count:]
    if trailing and matching_prefix and len(pcds) > len(images):
        for path in trailing:
            path.unlink()
        print("Discarded {} trailing PCD(s) without matching PNG".format(len(trailing)))
        pcds = pcds[:paired_count]
    elif trailing:
        print("WARNING: found {} unmatched PCD(s) before the end of the sequence; keeping them".format(
            len(trailing)), file=sys.stderr)
    return images, pcds


def main():
    args = parse_args()
    if not args.bag.is_file() or not args.config.is_file() or not args.camera_config.is_file():
        raise ValueError("bag, FAST-LIVO2 config, and camera config must all exist")
    if args.interval < 1 or args.rate <= 0.0:
        raise ValueError("interval must be positive and rate must be greater than zero")
    if args.image_transport is None:
        args.image_transport = "compressed" if args.image_topic.endswith("/compressed") else "raw"
    root = args.output.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("Output is not a directory: {}".format(root))
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise ValueError("Output is not empty; use --overwrite: {}".format(root))
    image_dir, pcd_dir = root / "all_image", root / "all_pcd_body"
    image_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)
    metadata = parse_metadata(args.config, args.camera_config, args, root)
    write_metadata(metadata, root, image_dir, pcd_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".launch", delete=False) as stream:
        stream.write(launch_text(args, image_dir, pcd_dir))
        launch_path = Path(stream.name)
    launch_process = bag_process = None
    try:
        print("Starting FAST-LIVO2 LIVO export (IMU-dewarped body PCD + VIO image)...")
        launch_process = subprocess.Popen(["roslaunch", str(launch_path)])
        time.sleep(3.0)
        if launch_process.poll() is not None:
            raise RuntimeError("roslaunch exited before rosbag playback")
        command = playback_command(args)
        print("Playing: {}".format(" ".join(command)))
        bag_process = subprocess.Popen(command)
        bag_return = bag_process.wait()
        if bag_return != 0:
            raise RuntimeError("rosbag play failed with exit code {}".format(bag_return))
        time.sleep(2.0)
    except KeyboardInterrupt:
        print("Interrupted; stopping playback and FAST-LIVO2.", file=sys.stderr)
        raise
    finally:
        if bag_process is not None and bag_process.poll() is None:
            bag_process.send_signal(signal.SIGINT)
            bag_process.wait()
        if launch_process is not None and launch_process.poll() is None:
            launch_process.send_signal(signal.SIGINT)
            try:
                launch_process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                launch_process.kill()
                launch_process.wait()
        launch_path.unlink(missing_ok=True)

    images, pcds = discard_unpaired_trailing_pcds(image_dir, pcd_dir)
    if not images:
        raise RuntimeError("FAST-LIVO2 produced no PNG files; check image topic and republish")
    if not pcds:
        raise RuntimeError("FAST-LIVO2 produced no PCD files; check IMU coverage and initialization")
    if len(images) != len(pcds):
        print("WARNING: exported {} images but {} PCDs (interval={})".format(
            len(images), len(pcds), args.interval), file=sys.stderr)
    print("Exported {} FAST-LIVO2 images to {}".format(len(images), image_dir))
    print("Exported {} IMU-dewarped body PCDs to {}".format(len(pcds), pcd_dir))
    print("Wrote projection metadata to {}".format(root / "projection_metadata.yaml"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
