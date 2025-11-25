import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import numpy as np
import time
import logging
import os
import threading
from multiprocessing import Process
from cv_bridge import CvBridge
import cv2
from collections import defaultdict
from ultralytics import YOLO
from urdfpy import URDF
import yaml
import pathlib
import numpy as np

# --- Fix for deprecated np.float alias in newer NumPy versions ---
if not hasattr(np, "float"):
    np.float = float


# -------------------------------
# YOLO setup
# -------------------------------
YOLO_MODEL_PATH = os.path.expanduser("~/runs/detect/train5/weights/best.pt")
model = YOLO(YOLO_MODEL_PATH)


class HelperFunctions:
    @staticmethod
    def get_joint_names():
        """
        Return the list of joint names.
        SHOULD MATCH THE JOINT NAMES IN ISAAC SIM
        """
        return [
            "ewellix_lift_top_joint",
            "pan_joint",
            "tilt_joint",
            *[f"left_joint_{i}" for i in range(1, 8)],
            *[f"right_joint_{i}" for i in range(1, 8)],
            *[f"left_hand_joint_{i}" for i in range(16)],
            *[f"right_hand_joint_{i}" for i in range(16)],
        ]

    @staticmethod
    def is_valid_joint_command(node_instance, joint_command):
        """Validate joint positions."""
        if len(joint_command.name) != Config.JOINTS_COUNT:
            node_instance.get_logger().error(
                "Invalid joint command: Incorrect number of names for /joint_command."
            )
            return False
        if (len(joint_command.position) != Config.JOINTS_COUNT) or (
            len(joint_command.velocity) != Config.JOINTS_COUNT
        ):
            node_instance.get_logger().error(
                "Invalid joint command: Incorrect number of position or velocity for /joint_command."
            )
            return False
        if not all(isinstance(pos, (float, int)) for pos in joint_command.position):
            node_instance.get_logger().error(
                "Invalid joint command: All positions must be floats or ints."
            )
            return False
        if not all(isinstance(vel, (float, int)) for vel in joint_command.velocity):
            node_instance.get_logger().error(
                "Invalid joint command: All velocities must be floats or ints."
            )
            return False
        return True

    @staticmethod
    def get_default_joint_position():
        """
        Return the default joint position
        - Returns a NumPy array with default joint positions (in radians or meters).
        - The count matches 49 joints.
        """
        return np.array(
            [
                0.2,  # ewellix_lift_top_joint
                0,
                1.0559,
                1.5708,
                -1.5708,
                1.5708,
                -1.5708,
                0,
                0,
                0,  # left arm
                1.5708,
                1.5708,
                -1.5708,
                1.5708,
                0,
                0,
                0,  # right arm
                *(0 for _ in range(16)),  # left hand
                *(0 for _ in range(16)),  # right hand
            ]
        )

    @staticmethod
    def get_movement_description(joint_name, curr_pos, prev_pos):
        """Return semantic description of movement based on joint name and direction."""
        if curr_pos > prev_pos:
            if "lift" in joint_name:
                return "moving up"
            elif "tilt" in joint_name:
                return "tilting up"
            elif "pan" in joint_name:
                return "panning left"
            elif "hand" in joint_name:
                return "closing/grabbing"
            else:
                return "moving forward"
        elif curr_pos < prev_pos:
            if "lift" in joint_name:
                return "moving down"
            elif "tilt" in joint_name:
                return "tilting down"
            elif "pan" in joint_name:
                return "panning right"
            elif "hand" in joint_name:
                return "opening/spreading"
            else:
                return "moving backward"
        else:
            return "holding position"

    def load_camera_calibration(
        path="/home/nadeen/ETRI-Dual-Hand-Arm-Robot/src/sample_etri_dualarm_ctr/sample_etri_dualarm_ctr/camera_calib.yaml",
    ):
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Camera calibration file not found: {path}")
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeff"], dtype=np.float64).reshape(-1, 1)
        image_size = tuple(data.get("image_size", (1280, 720)))
        return K, dist, image_size


class Config:
    # Motion control constants
    SINE_FREQUENCY = 1
    PUBLISH_FREQUENCY = 30.0
    ITERATIONS = 2

    JOINTS_NAME = HelperFunctions.get_joint_names()
    JOINTS_COUNT = len(JOINTS_NAME)

    # Stall detection
    VEL_THRESH = 0.01

    # Thermal model parameters
    K_T = 0.12  # Nm/A
    R_MOTOR = 0.2  # Ω
    C_TH = 250.0  # J/°C (thermal mass)
    R_TH = 5.0  # °C/W (thermal resistance)
    MAX_TORQUE_NM = 200.0
    MAX_DISSIPATION_W = 500.0
    MECH_LOSS_FRAC = 0.2
    T_AMBIENT = 25.0  # Ambient temperature (°C)
    LEAD_SCREW_M_PER_REV = 0.01  # Example: 10 mm lead screw
    GEAR_RATIO_LIFT = 5.0  # Example gearbox ratio for lift actuator

    # Image processing
    CAMERA_RGB = "/head_camera/color/image_raw"
    CAMERA_DEPTH = "/head_camera/depth/image_rect_raw"
    QUEUE = 10


class ImageProcessor(Node):
    def __init__(self):
        super().__init__("image_processor")
        self.bridge = CvBridge()

        try:
            self.K, self.dist, self.image_size = (
                HelperFunctions.load_camera_calibration()
            )
            self.get_logger().info("✅ Loaded camera calibration successfully.")
        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            self.K, self.dist, self.image_size = None, None, None

        # YOLO class names from your dataset
        self.CLASS_ID_MAP = {
            0: "objects",
            1: "dumbbell",
            2: "mustard bottle",
            3: "robot arm",
            4: "robot base",
            5: "robot hand",
            6: "spam can",
        }
        self.SELF_CLASS_IDS = [3, 4, 5]  # mark self

        self.model = model
        self.confidence_threshold = 0.5

        # ROS Subscriptions
        self.rgb_subscription = self.create_subscription(
            Image, "/head_camera/color/image_raw", self.rgb_callback, 10
        )
        self.depth_subscription = self.create_subscription(
            Image, "/head_camera/depth/image_rect_raw", self.depth_callback, 10
        )
        self.joint_state_subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.joint_positions = {}

        self.rgb_image = None
        self.depth_image = None
        self.frame_count = 0

        # Output folders
        base_path = "/home/nadeen/ETRI-Dual-Hand-Arm-Robot/camera_frames"
        self.rgb_save_path = os.path.join(base_path, "rgb")
        self.depth_save_path = os.path.join(base_path, "depth")
        os.makedirs(self.rgb_save_path, exist_ok=True)
        os.makedirs(self.depth_save_path, exist_ok=True)

    def detect_objects(self, frame, score_threshold=0.5):
        """Run YOLOv8 on a frame and return boxes, scores, labels."""
        results = self.model(frame, conf=score_threshold, verbose=False)
        pred = results[0]

        if pred.boxes is not None and len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            labels = pred.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes, scores, labels = np.array([]), np.array([]), np.array([])
        return boxes, scores, labels

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )
        if self.rgb_image is not None:
            self.process_images()

    def process_images(self):
        boxes, scores, labels = self.detect_objects(self.rgb_image)

        # Prepare depth visualization
        depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        object_depths = defaultdict(list)

        # Analyze each detected object - no merged boxes
        for idx, label in enumerate(labels):
            box = boxes[idx].astype(int)
            x1, y1, x2, y2 = box
            score = scores[idx]
            label = labels[idx]

            class_name = self.CLASS_ID_MAP.get(label, str(label))
            category = "self" if label in self.SELF_CLASS_IDS else "external"

            roi_depth = self.depth_image[y1:y2, x1:x2]
            valid_depth = roi_depth[roi_depth > 0]

            if valid_depth.size > 0:
                avg_depth = np.mean(valid_depth)

            self.get_logger().info(
                f"Detected {category}: {class_name}, "
                f"Score={score:.2f}, Depth={avg_depth:.2f} m, Box={box.tolist()}"
            )

            # Draw bounding boxes
            if len(boxes) > 0:
                box = boxes[idx].astype(int)
                x1, y1, x2, y2 = box
                color = (128, 128, 128) if category == "self" else (0, 0, 0)
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    depth_color,
                    # f"{class_name}:{score:.2f} ({avg_depth:.2f}m)",
                    f"{class_name}:{category}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

            # Get robot skeleton from joint states and URDF
            joints_2d, links = self.compute_forward_kinematics()
            if joints_2d:
                self.draw_skeleton(self.rgb_image, joints_2d, links)

        # TODO: Analyze merged boxes per class

        # Show and save visualization
        cv2.imshow("Depth Visualization", depth_color)
        cv2.waitKey(1)

        cv2.imshow("RGB Skeleton", self.rgb_image)
        cv2.waitKey(1)

        # Save images
        depth_filename = f"depth_{self.frame_count:06d}.png"
        cv2.imwrite(os.path.join(self.depth_save_path, depth_filename), depth_color)

        rgb_filename = f"rgb_{self.frame_count:06d}.png"
        cv2.imwrite(os.path.join(self.rgb_save_path, rgb_filename), self.rgb_image)

        self.frame_count += 1

    def joint_state_callback(self, msg):
        """Store latest joint positions."""
        self.joint_positions = {name: pos for name, pos in zip(msg.name, msg.position)}

    def compute_forward_kinematics(self):
        """
        Compute 2D coordinates for left/right arms and hands only.
        Returns: (joints_2d, links)
        """
        if not self.joint_positions:
            return [], []

        # Load URDF once
        if not hasattr(self, "robot_model"):
            self.robot_model = URDF.load(
                "/home/nadeen/ETRI-Dual-Hand-Arm-Robot/urdf/etri_dualarm_robot (original).urdf"
            )

        # Filter relevant joints
        target_prefixes = [
            "left_joint_",
            "right_joint_",
            "left_hand_joint_",
            "right_hand_joint_",
        ]
        filtered_positions = {
            name: pos
            for name, pos in self.joint_positions.items()
            if any(name.startswith(p) for p in target_prefixes)
        }

        if not filtered_positions:
            return [], []

        # Compute forward kinematics for only these joints
        link_poses = self.robot_model.link_fk(cfg=filtered_positions)

        # Get link names and 3D positions
        link_names = []
        joints_3d = []
        for link, pose in link_poses.items():
            link_names.append(link.name)
            pos = pose[:3, 3]
            joints_3d.append(pos)

        # TODO: verify these values with actual calibration, this might be the casue of incorrect projection

        # Camera extrinsics (from URDF)
        R_cam_base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        t_cam_base = np.array([[0.056], [0.038], [0.351]])  # meters

        # Project to 2D (approximate intrinsics; replace with calibrated values if known)
        fx, fy, cx, cy = 634.0862399675711, 634.0862399675711, 640.0, 360.0
        # Project 3D joints (robot base → camera → pixel)
        joints_2d = []
        for pos in joints_3d:
            p_base = np.array(pos).reshape(3, 1)
            p_cam = R_cam_base @ p_base + t_cam_base
            X, Y, Z = p_cam.flatten()
            if Z <= 0:
                continue
            u = int(fx * X / Z + cx)
            v = int(fy * Y / Z + cy)
            joints_2d.append((u, v))

        # Build links only for left/right arms & hands
        links = []
        for joint in self.robot_model.joints:
            # only keep joints connecting left/right arms or hands
            if not any(
                k in joint.name
                for k in ["left_joint", "right_joint", "left_hand", "right_hand"]
            ):
                continue
            parent = joint.parent
            child = joint.child
            if parent in link_names and child in link_names:
                i1 = link_names.index(parent)
                i2 = link_names.index(child)
                if i1 < len(joints_2d) and i2 < len(joints_2d):
                    links.append((i1, i2))

        return joints_2d, links

    def draw_skeleton(
        self, img, joints, links, joint_color=(0, 255, 0), link_color=(0, 0, 255)
    ):
        """
        Draw joints and links on the RGB image.
        - img: OpenCV image (BGR)
        - joints: list or np.array of 2D joint positions [(x, y), ...]
        - links: list of tuples indicating connections [(i1, i2), (i2, i3), ...]
        """
        for x, y in joints:
            cv2.circle(img, (int(x), int(y)), 4, joint_color, -1)

        for i1, i2 in links:
            x1, y1 = joints[i1]
            x2, y2 = joints[i2]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), link_color, 2)


class CommandProcessor(Node):
    def __init__(self):
        super().__init__("command_processor")

        # Publisher setup
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)
        self.subscriber_ = self.create_subscription(
            JointState, "/joint_states", self.estimate_temp_callback, 10
        )

        # State storage
        self.joint_command = JointState()
        self.default_position = HelperFunctions.get_default_joint_position()
        self.prev_positions = {}
        self.latest_velocity = {}
        self.latest_effort = {}
        self.latest_temp = {name: Config.T_AMBIENT for name in Config.JOINTS_NAME}
        self.latest_time = {name: time.time() for name in Config.JOINTS_NAME}
        self.joint_history = defaultdict(list)
        self.lock = threading.Lock()

        # Logging setup
        self.file_logger = logging.getLogger("joint_logger")
        self.file_logger.setLevel(logging.INFO)
        self.logger = logging.getLogger("self_aware_logger")
        self.logger.setLevel(logging.INFO)
        log_dir = os.path.expanduser("~/ETRI-Dual-Hand-Arm-Robot/joints_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "holding_objects.txt")
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.file_logger.addHandler(fh)

        self.R_motor = Config.R_MOTOR
        self.dt = 1.0 / Config.PUBLISH_FREQUENCY
        self.vel_thresh = Config.VEL_THRESH
        self.active_joints = set()

    def estimate_temp_callback(self, data):
        for i, name in enumerate(data.name):
            if name not in self.latest_temp:
                with self.lock:
                    self.latest_temp[name] = Config.T_AMBIENT
                    self.latest_time[name] = time.time()

            if name not in self.active_joints:
                continue

            curr_pos = data.position[i] if i < len(data.position) else float("nan")
            vel = data.velocity[i] if i < len(data.velocity) else 0.0
            eff_raw = data.effort[i] if i < len(data.effort) else 0.0

            # --- Torque conversion ---
            if name == "ewellix_lift_top_joint":
                # Prismatic → rotational torque at motor
                torque_motor = (
                    eff_raw
                    * (Config.LEAD_SCREW_M_PER_REV / (2 * np.pi))
                    * Config.GEAR_RATIO_LIFT
                )
            else:
                torque_motor = eff_raw

            # Clamp torque
            if abs(torque_motor) > Config.MAX_TORQUE_NM:
                self.get_logger().warn(f"{name}: Torque {torque_motor:.1f} Nm clipped")
                torque_motor = np.clip(
                    torque_motor, -Config.MAX_TORQUE_NM, Config.MAX_TORQUE_NM
                )

            # --- Current from torque ---
            # Use the effective K_T. NOTE: Config contains two K_T assignments in original file;
            # Python will use the last assignment. Ensure K_T is set to the desired physical value.
            I = torque_motor / Config.K_T if Config.K_T != 0 else 0.0

            # --- Electrical copper losses (always present) ---
            P_cu = (I**2) * Config.R_MOTOR

            # --- Mechanical power ---
            P_mech = abs(torque_motor * vel)

            # --- Mechanical dissipation ---
            P_mech_loss = Config.MECH_LOSS_FRAC * P_mech

            # --- Total heat generation (cap to realistic dissipation) ---
            raw_heat = P_cu + P_mech_loss
            # Prevent absurd instantaneous heating by clamping to a motor's max dissipation
            heat = min(raw_heat, Config.MAX_DISSIPATION_W)

            # --- Cooling model ---
            # Convection increases with speed (simple fan effect).
            # Add a weak nonlinear temperature term so cooling grows for larger ΔT.
            with self.lock:
                T_prev = self.latest_temp.get(name, Config.T_AMBIENT)

            deltaT = T_prev - Config.T_AMBIENT
            # cooling_coeff scales base heat transfer (1/R_th), and increases with movement
            cooling_coeff = (1.0 / Config.R_TH) * (1.0 + 0.25 * min(abs(vel), 1.0))
            # Use a weak nonlinearity so cooling becomes stronger for larger temp differences
            # Ensure sign is preserved (heat flow direction).
            if deltaT == 0:
                cooling = 0.0
            else:
                cooling = cooling_coeff * (abs(deltaT) ** 1.08) * np.sign(deltaT)

            # --- Temperature update ---
            now = time.time()
            with self.lock:
                last_time = self.latest_time.get(name, now)
            dt = now - last_time
            if dt <= 0:
                dt = self.dt  # fallback to publish interval

            # dT = (heat_in - cooling_out) * dt / C_th
            # Note: cooling is positive when temp > ambient (i.e., it removes heat)
            # so we subtract cooling here (cooling is positive when removing heat)
            with self.lock:
                T_new = (
                    self.latest_temp.get(name, Config.T_AMBIENT)
                    + (heat - cooling) * dt / Config.C_TH
                )
                # safety cap: avoid temperatures > reasonable extreme (protect from runaway due to bugs)
                # This cap is generous; tune as needed per hardware.
                T_new = float(np.clip(T_new, -40.0, 200.0))
                self.latest_temp[name] = T_new
                self.latest_time[name] = now

            # --- Logging ---
            prev_pos = self.prev_positions.get(name, curr_pos)
            movement_desc = HelperFunctions.get_movement_description(
                name, curr_pos, prev_pos
            )
            movement_delta = abs(curr_pos - prev_pos)

            if movement_delta > 1e-4 or abs(vel) > 1e-4 or abs(eff_raw) > 1e-2:
                self.file_logger.info(
                    f"Joint: {name}, "
                    f"Initial Pos: {prev_pos:.4f}, "
                    f"Current Pos: {curr_pos:.4f}, "
                    f"Movement: {movement_desc}, "
                    f"Velocity: {vel:.4f}, "
                    f"Effort(raw): {eff_raw:.4f}, "
                    f"Torque(Nm): {torque_motor:.4f}, "
                    f"P_cu(W): {P_cu:.4f}, "
                    f"P_mech_loss(W): {P_mech_loss:.4f}, "
                    f"TotalHeat(W): {heat:.4f}, "
                    f"Temp(°C): {self.latest_temp[name]:.2f}"
                )

            self.prev_positions[name] = curr_pos
            self.latest_effort[name] = torque_motor
            self.latest_velocity[name] = vel

    def publish_1D_joint_position(self, joint_names, amplitude):
        """Move specified joints once in one direction."""
        self.active_joints = set(joint_names)
        # total_duration = Config.ITERATIONS / Config.SINE_FREQUENCY
        total_duration = 1
        start_time = time.time()
        joint_indices = [Config.JOINTS_NAME.index(name) for name in joint_names]
        initial_positions = self.default_position[joint_indices].copy()

        while time.time() - start_time < total_duration:
            elapsed_time = time.time() - start_time

            # Move gradually in one direction
            progress = min(elapsed_time / total_duration, 1.0)
            new_positions = initial_positions + amplitude * progress

            # Apply to default positions
            for idx, joint_idx in enumerate(joint_indices):
                self.default_position[joint_idx] = new_positions[idx]

            self.joint_command.header.stamp = self.get_clock().now().to_msg()
            self.joint_command.name = Config.JOINTS_NAME
            self.joint_command.position = self.default_position.tolist()
            self.joint_command.velocity = [0.0] * Config.JOINTS_COUNT

            if HelperFunctions.is_valid_joint_command(self, self.joint_command):
                self.publisher_.publish(self.joint_command)
            time.sleep(1.0 / Config.PUBLISH_FREQUENCY)

        self.active_joints = set()

    def grab_object(self, joint, movement, amplitude, extra):
        # Separate fingers and thumb
        finger_joints = [f"{joint}{i}" for i in range(0, 12) if i not in [0, 4, 8]]
        thumb_joints = [f"{joint}{i}" for i in range(13, 16)]

        self.publish_1D_joint_position([f"{joint}12"], amplitude)

        # Gradual thumb closure
        for step in np.linspace(*movement):
            self.publish_1D_joint_position(thumb_joints, step)
            time.sleep(0.05)

        if extra["active"]:
            self.publish_1D_joint_position([extra["joint"]], extra["amplitude"])

        # Gradual finger closure
        for step in np.linspace(*movement):
            self.publish_1D_joint_position(finger_joints, step)
            time.sleep(0.05)

    def release_object(self, joint, movement, amplitude):
        # Separate fingers and thumb
        finger_joints = [f"{joint}{i}" for i in range(0, 12) if i not in [0, 4, 8]]
        thumb_joints = [f"{joint}{i}" for i in range(13, 16)]

        # Gradual thumb openning
        for step in np.linspace(*movement):
            self.publish_1D_joint_position(thumb_joints, step)
            time.sleep(0.05)

        self.publish_1D_joint_position([f"{joint}{12}"], amplitude)

        # Gradual fingers openning
        for step in np.linspace(*movement):
            self.publish_1D_joint_position(finger_joints, step)
            time.sleep(0.05)

    def move_to_default_pose(self):
        """Move all joints back to their default position instantly"""
        self.joint_command.header.stamp = self.get_clock().now().to_msg()
        self.joint_command.position = (
            HelperFunctions.get_default_joint_position().tolist()
        )
        if HelperFunctions.is_valid_joint_command(self, self.joint_command):
            self.publisher_.publish(self.joint_command)

    def check_default_pose(self):
        errors = []
        for i, joint_name in enumerate(Config.JOINTS_NAME):
            curr_pos = self.prev_positions.get(joint_name, None)
            default_pos = self.default_position[i]
            if curr_pos is None or abs(curr_pos - default_pos) > Config.POSITION_TOL:
                errors.append((joint_name, curr_pos, default_pos))
                self.get_logger().error(
                    f"Joint {joint_name} out of default position: "
                    f"current={curr_pos}, expected={default_pos}"
                )
        return errors

    def joint_state_callback(self, data: JointState):
        now = time.time()
        for i, name in enumerate(data.name):
            curr_pos = data.position[i] if i < len(data.position) else float("nan")
            vel = data.velocity[i] if i < len(data.velocity) else 0.0
            effort = data.effort[i] if i < len(data.effort) else 0.0
            temp = self.latest_temp.get(name, 25.0)

            # Record joint state
            with self.lock:
                self.prev_positions[name] = curr_pos
                self.latest_velocity[name] = vel
                self.latest_effort[name] = effort
                self.latest_time[name] = now
                self.joint_history[name].append((curr_pos, vel, effort, temp))
                if len(self.joint_history[name]) > 50:
                    self.joint_history[name].pop(0)

            # Alerts
            if (
                abs(curr_pos - self.default_position[Config.JOINTS_NAME.index(name)])
                > Config.POSITION_TOL
            ):
                self.logger.warning(f"{name} deviates from default!")
            if effort > Config.TORQUE_MAX:
                self.logger.warning(f"{name} torque high: {effort:.2f}")
            if temp > Config.TEMP_MAX:
                self.logger.warning(f"{name} temperature high: {temp:.2f}")

    def move_mustard(self):
        self.get_logger().info("reach the mustard..")
        self.publish_1D_joint_position(["right_joint_1"], -0.4)
        self.publish_1D_joint_position(["right_joint_4"], -0.6)
        self.publish_1D_joint_position(["right_joint_2"], 0.175)

        self.get_logger().info("grab the mustard..")
        self.publish_1D_joint_position(["right_joint_7"], 0.175)
        self.grab_object("right_hand_joint_", (0, 0.25, 8), 2, {"active": False})
        time.sleep(0.5)

        self.get_logger().info("left the mustard..")
        self.publish_1D_joint_position(["right_joint_4"], 0.6)
        time.sleep(0.5)

        self.get_logger().info("move the mustard..")
        self.publish_1D_joint_position(["right_joint_2"], -0.3)
        self.publish_1D_joint_position(["right_joint_4"], -0.6)
        time.sleep(0.5)

        self.get_logger().info("release the mustard..")
        self.publish_1D_joint_position(["right_joint_7"], -0.175)
        self.release_object("right_hand_joint_", (-0.25, 0.0, 8), -2)
        self.publish_1D_joint_position(["right_joint_4"], 0.8)
        time.sleep(0.5)

        self.get_logger().info("back to default position..")
        self.move_to_default_pose()

    def move_spam(self):
        self.get_logger().info("reach the spam..")
        self.publish_1D_joint_position(["left_joint_1"], 0.8)
        self.publish_1D_joint_position(["left_joint_7"], -1.6)
        self.publish_1D_joint_position(["left_joint_4"], 1)

        self.get_logger().info("grab the spam..")
        self.grab_object(
            "left_hand_joint_",
            (0, 0.25, 5),
            5,
            {"active": True, "joint": "left_joint_4", "amplitude": 0.1},
        )
        time.sleep(0.5)

        self.get_logger().info("left the spam..")
        self.publish_1D_joint_position(["left_joint_4"], -0.6)
        time.sleep(0.5)

        self.get_logger().info("move the spam..")
        self.publish_1D_joint_position(["left_joint_2"], -0.3)
        self.publish_1D_joint_position(["left_joint_4"], 0.6)
        time.sleep(0.5)

        self.get_logger().info("release the spam..")
        self.release_object("left_hand_joint_", (-0.25, 0.0, 5), -5)
        self.publish_1D_joint_position(["left_joint_7"], 1.6)
        self.publish_1D_joint_position(["left_joint_4"], -1.0)
        time.sleep(0.5)

        self.get_logger().info("back to default position..")
        self.move_to_default_pose()

    def move_dumbbell(self):

        self.get_logger().info("reach the dumbbell..")
        self.publish_1D_joint_position(["left_joint_4"], -0.1)
        time.sleep(0.3)
        self.publish_1D_joint_position(["left_joint_7"], 1.6)
        time.sleep(2)
        self.publish_1D_joint_position(["left_joint_4"], 0.65)
        time.sleep(0.3)
        self.publish_1D_joint_position(["left_joint_1"], 0.3)
        time.sleep(0.3)

        self.get_logger().info("grab the dumbbell..")
        # Separate fingers and thumb
        finger_joints = [
            f"left_hand_joint_{i}" for i in range(0, 12) if i not in [0, 4, 8]
        ]
        thumb_joints = [f"left_hand_joint_{i}" for i in range(14, 16)]
        # Positioning thumb
        self.publish_1D_joint_position(["left_hand_joint_12"], 5)
        time.sleep(0.3)
        # Gradual finger closure
        for step in np.linspace(0, 0.3, 8):
            self.publish_1D_joint_position(finger_joints, step)
            time.sleep(0.08)
        time.sleep(1)
        # Gradual thumb closure
        for step in np.linspace(0, 0.3, 8):
            self.publish_1D_joint_position(thumb_joints, step)
            time.sleep(0.08)
        self.publish_1D_joint_position(["left_joint_4"], 0.05)
        time.sleep(0.3)

        self.get_logger().info("left the dumbbell..")
        self.publish_1D_joint_position(["left_joint_4"], -0.6)
        time.sleep(1)

        self.get_logger().info("move the dumbbell..")
        self.publish_1D_joint_position(["left_joint_4"], 0.4)
        time.sleep(0.5)

        self.publish_1D_joint_position(["left_joint_4"], -0.4)
        time.sleep(0.5)

        self.publish_1D_joint_position(["left_joint_4"], 0.4)
        time.sleep(0.5)

        self.publish_1D_joint_position(["left_joint_4"], -0.4)
        time.sleep(0.5)

    def execute_robot_motions(self):
        # self.move_mustard()
        self.move_spam()
        # self.move_dumbbell()


def run_command_node():
    rclpy.init()
    node = CommandProcessor()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.execute_robot_motions()
    finally:
        node.destroy_node()
        spin_thread.join(timeout=1)


def run_image_node():
    rclpy.init()
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# both exit
def main():
    # Start the command node in a separate process
    command_process = Process(target=run_command_node)
    command_process.start()

    # Initialize image node
    rclpy.init()
    image_node = ImageProcessor()

    try:
        # Loop until command process finishes
        while command_process.is_alive():
            rclpy.spin_once(image_node, timeout_sec=0.1)  # spin without blocking
            # Optional: small sleep to reduce CPU usage
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Shutdown
        if command_process.is_alive():
            command_process.terminate()
        command_process.join()

        image_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
