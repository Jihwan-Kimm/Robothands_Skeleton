from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider

from urchin import URDF
import time


# ---------------------------------------------------------------------------
# 조인트 이름 & 기본 포즈 정의
# ---------------------------------------------------------------------------

JOINT_NAMES: Sequence[str] = [
    "ewellix_lift_top_joint",
    "pan_joint",
    "tilt_joint",
    *[f"left_joint_{i}" for i in range(1, 8)],
    *[f"right_joint_{i}" for i in range(1, 8)],
    *[f"left_hand_joint_{i}" for i in range(16)],
    *[f"right_hand_joint_{i}" for i in range(16)],
]

# 기본 포즈
DEFAULT_JOINT_VALUES = np.array([
    0.2,      # ewellix_lift_top_joint (prismatic)
    0.0,      # pan_joint
    1.0559,   # tilt_joint

    1.5708,   # left_joint_1
    -1.5708,  # left_joint_2
    1.5708,   # left_joint_3
    -1.5708,  # left_joint_4
    0.0,      # left_joint_5
    0.0,      # left_joint_6
    0.0,      # left_joint_7

    1.5708,   # right_joint_1
    1.5708,   # right_joint_2
    -1.5708,  # right_joint_3
    1.5708,   # right_joint_4
    0.0,      # right_joint_5
    0.0,      # right_joint_6
    0.0,      # right_joint_7

    # left_hand_joint_0 ~ 15
    *[0.0] * 16,
    # right_hand_joint_0 ~ 15
    *[0.0] * 16,
], dtype=float)

URDF_FILENAME = "etri_dualarm_robot.urdf"
HEAD_LINK_NAME = "head_camera_color_optical_frame"


def get_head_link_position(robot: URDF, joint_cfg: Dict[str, float]) -> np.ndarray:
    """HEAD_LINK_NAME 링크의 월드 좌표를 URDF FK로부터 구해서 반환."""
    fk = robot.link_fk(cfg=joint_cfg)
    for link, T in fk.items():
        if link.name == HEAD_LINK_NAME:
            pos = np.asarray(T[:3, 3], dtype=float).reshape(3)
            return pos
    raise KeyError(f"HEAD_LINK_NAME='{HEAD_LINK_NAME}' 링크를 FK 결과에서 찾을 수 없습니다.")


def get_joint_names() -> Sequence[str]:
    return list(JOINT_NAMES)


def get_default_pose() -> Dict[str, float]:
    return dict(zip(JOINT_NAMES, DEFAULT_JOINT_VALUES))


def make_joint_cfg(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    cfg = get_default_pose()
    if overrides:
        for name, value in overrides.items():
            if name not in cfg:
                raise KeyError(f"알 수 없는 조인트 이름입니다: {name}")
            cfg[name] = float(value)
    return cfg


# ---------------------------------------------------------------------------
# URDF 로딩 및 Forward Kinematics
# ---------------------------------------------------------------------------

def load_robot(urdf_path: Optional[Path] = None) -> URDF:
    if urdf_path is None:
        urdf_path = Path(__file__).with_name(URDF_FILENAME)

    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF 파일을 찾을 수 없습니다: {urdf_path}")

    return URDF.load(str(urdf_path), lazy_load_meshes=True)


def compute_link_positions(robot: URDF, joint_cfg: Dict[str, float]) -> Dict[str, np.ndarray]:
    fk = robot.link_fk(cfg=joint_cfg)
    positions: Dict[str, np.ndarray] = {}
    for link, transform in fk.items():
        pos = np.asarray(transform[:3, 3]).reshape(3)
        positions[link.name] = pos
    return positions


# ---------------------------------------------------------------------------
# 스켈레톤 초기화 / 업데이트
#   - 카메라: pitch(=elev), yaw(=azim), FOV 슬라이더
#   - 스켈레톤: tx, ty, tz 슬라이더 (로봇 전체 평행이동)
#   + 슬라이더 값 상태 텍스트/콘솔 출력
# ---------------------------------------------------------------------------
def _init_skeleton_artists(
    robot: URDF,
    joint_cfg: Dict[str, float],
    *,
    view_elev: float = 30.0,
    view_azim: float = -60.0,
    init_fov_deg: float = 120.0,
):
    positions = compute_link_positions(robot, joint_cfg)

    # ------------------------------------------------------------------
    # 여기서부터: 그릴 링크만 명시적으로 정의
    # ------------------------------------------------------------------
    left_arm_core = [
        "left_spherical_wrist_1_link",
        "left_spherical_wrist_2_link",
        "left_bracelet_link",
    ]
    right_arm_core = [
        "right_spherical_wrist_1_link",
        "right_spherical_wrist_2_link",
        "right_bracelet_link",
    ]
    # SIMPLIFIED: Only 2 joints per finger (base + tip)
    left_fingers = [
        ["left_hand_link_0", "left_hand_link_3_tip"],
        ["left_hand_link_4", "left_hand_link_7_tip"],
        ["left_hand_link_8", "left_hand_link_11_tip"],
        ["left_hand_link_12", "left_hand_link_15_tip"],
    ]
    right_fingers = [
        ["right_hand_link_0", "right_hand_link_3_tip"],
        ["right_hand_link_4", "right_hand_link_7_tip"],
        ["right_hand_link_8", "right_hand_link_11_tip"],
        ["right_hand_link_12", "right_hand_link_15_tip"],
    ]
    finger_bases = [
        "left_hand_link_0",
        "left_hand_link_4",
        "left_hand_link_8",
        "left_hand_link_12",
        "right_hand_link_0",
        "right_hand_link_4",
        "right_hand_link_8",
        "right_hand_link_12",
    ]

    # 선으로 이을 (parent, child) 쌍들을 구성
    line_pairs: List[tuple[str, str]] = []

    def add_chain(chain: List[str]) -> None:
        for a, b in zip(chain[:-1], chain[1:]):
            if a in positions and b in positions:
                line_pairs.append((a, b))

    # 팔 코어: 3개 링크를 연속으로 이음
    add_chain(left_arm_core)
    add_chain(right_arm_core)

    # 손가락: base - tip만 직선으로
    for base, tip in left_fingers + right_fingers:
        if base in positions and tip in positions:
            line_pairs.append((base, tip))

    # 실제로 사용할 링크 이름 집합 (선에 등장하는 것만)
    link_names = sorted({name for pair in line_pairs for name in pair})

    # ------------------------------------------------------------------
    # Figure / Axes 설정
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax: Axes3D = fig.add_axes([0.05, 0.32, 0.9, 0.63], projection="3d")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    def draw_segment(link_a, link_b, color="magenta"):
        if link_a not in positions or link_b not in positions:
            return None
        p0, p1 = positions[link_a], positions[link_b]
        return ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=color, linewidth=2.0, alpha=1.0
        )[0]

    # ----------------- 링크 라인 생성 -----------------
    line_list: List[Optional[plt.Line2D]] = []
    for parent_name, child_name in line_pairs:
        p0 = positions[parent_name]
        p1 = positions[child_name]
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]
        line, = ax.plot(
            xs, ys, zs,
            linewidth=1.5,
            color="magenta",
            alpha=0.9,
        )
        line_list.append(line)

    # ----------------- 점 생성 (선에 등장하는 링크들만) -----------------
    pts = np.stack([positions[name] for name in link_names], axis=0)
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=20.0,
        c="yellow",
        depthshade=False,  # 진하게
        alpha=1.0,         # 투명도 없이
    )

    # ===========================
    # 1) 머리 위치 & "척추 방향" 계산 (프레이밍 용)
    # ===========================
    head_pos = get_head_link_position(robot, joint_cfg)

    # z가 가장 낮은 점을 "바닥 쪽"으로 가정
    min_z_idx = np.argmin(pts[:, 2])
    root_pos = pts[min_z_idx]

    spine_vec = root_pos - head_pos          # head -> root
    spine_len = float(np.linalg.norm(spine_vec))
    if spine_len < 1e-6:
        spine_len = 0.5

    # ===========================
    # 2) 프레이밍: 머리 주변으로 축 잡기
    # ===========================
    center = head_pos + 0.3 * spine_vec
    half = 0.5 * spine_len

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # ===========================
    # 3) 카메라 초기 pitch/yaw
    # ===========================
    cam_vec = head_pos - center
    vx, vy, vz = float(cam_vec[0]), float(cam_vec[1]), float(cam_vec[2])
    yaw_init = np.degrees(np.arctan2(vy, vx))                         # azim
    pitch_init = np.degrees(np.arctan2(vz, np.hypot(vx, vy)))         # elev

    if not np.isfinite(yaw_init):
        yaw_init = view_azim
    if not np.isfinite(pitch_init):
        pitch_init = view_elev

    ax.view_init(elev=pitch_init, azim=yaw_init)

    # ===========================
    # 4) 투영/FOV 설정
    # ===========================
    def f_from_fov_deg(fov_deg: float) -> float:
        # FOV = 2 * atan(1 / f)  ->  f = 1 / tan(FOV/2)
        fov_rad = math.radians(max(1.0, min(179.0, fov_deg)))
        return 1.0 / math.tan(0.5 * fov_rad)

    focal_init = f_from_fov_deg(init_fov_deg)
    ax.set_proj_type('persp', focal_length=focal_init)

    # ------------------------------------------------------------------
    # 슬라이더 설정
    # ------------------------------------------------------------------
    axcolor = "#222222"
    ax_pitch = fig.add_axes([0.05, 0.25, 0.9, 0.02], facecolor=axcolor)
    ax_yaw   = fig.add_axes([0.05, 0.21, 0.9, 0.02], facecolor=axcolor)
    ax_fov   = fig.add_axes([0.05, 0.17, 0.9, 0.02], facecolor=axcolor)
    ax_tx    = fig.add_axes([0.05, 0.13, 0.9, 0.02], facecolor=axcolor)
    ax_ty    = fig.add_axes([0.05, 0.09, 0.9, 0.02], facecolor=axcolor)
    ax_tz    = fig.add_axes([0.05, 0.05, 0.9, 0.02], facecolor=axcolor)

    s_pitch = Slider(ax_pitch, "pitch", -180.0, 180.0, valinit=59.33)
    s_yaw   = Slider(ax_yaw,   "yaw",   -180.0, 180.0, valinit=180)
    s_fov   = Slider(ax_fov,   "FOV",    5.0, 175.0, valinit=161.78)

    trans_range = spine_len * 2
    s_tx = Slider(ax_tx, "tx", -trans_range, trans_range, valinit=-0.923)
    s_ty = Slider(ax_ty, "ty", -trans_range, trans_range, valinit=0.005)
    s_tz = Slider(ax_tz, "tz", -trans_range, trans_range, valinit=1.435)

    fig._last_positions = positions

    status_text = fig.text(
        0.02, 0.98,
        "",
        color="white",
        ha="left",
        va="top",
        fontsize=9,
    )

    def update_status_text() -> None:
        pitch = s_pitch.val
        yaw = s_yaw.val
        fov_deg = s_fov.val
        tx = s_tx.val
        ty = s_ty.val
        tz = s_tz.val
        status = (
            f"pitch={pitch:.2f}, yaw={yaw:.2f}, FOV={fov_deg:.2f} deg\n"
            f"tx={tx:.3f}, ty={ty:.3f}, tz={tz:.3f}"
        )
        status_text.set_text(status)
        print(status)

    def update_camera(_val=None) -> None:
        pitch = s_pitch.val
        yaw = s_yaw.val
        fov_deg = s_fov.val

        ax.view_init(elev=pitch, azim=yaw)
        focal = f_from_fov_deg(fov_deg)
        ax.set_proj_type("persp", focal_length=focal)

        update_status_text()
        fig.canvas.draw_idle()

    def update_skeleton(_val=None) -> None:
        positions_cur = getattr(fig, "_last_positions", None)
        if positions_cur is None:
            return
        offset = np.array([s_tx.val, s_ty.val, s_tz.val], dtype=float)
        _update_skeleton_artists(
            robot,
            positions_cur,
            line_list,
            scatter,
            link_names,
            offset=offset,
        )
        update_status_text()
        fig.canvas.draw_idle()

    for s in (s_pitch, s_yaw, s_fov):
        s.on_changed(update_camera)
    for s in (s_tx, s_ty, s_tz):
        s.on_changed(update_skeleton)

    update_camera(None)
    update_skeleton(None)

    fig._cam_sliders = {
        "pitch": s_pitch,
        "yaw": s_yaw,
        "fov": s_fov,
    }
    fig._skel_sliders = {
        "tx": s_tx,
        "ty": s_ty,
        "tz": s_tz,
    }

    return fig, ax, line_list, scatter, link_names


def _update_skeleton_artists(
    robot: URDF,  # robot는 더 이상 안 써도 되지만 시그니처 유지
    positions: Dict[str, np.ndarray],
    line_list: List,
    scatter,
    link_names: List[str],
    offset: Optional[np.ndarray] = None,
) -> None:
    """현재 positions + offset에 맞게 라인/스캐터 아티스트를 업데이트."""
    if offset is None:
        offset = np.zeros(3, dtype=float)

    # _init_skeleton_artists와 동일한 방식으로 line_pairs 재구성
    left_arm_core = [
        "left_spherical_wrist_1_link",
        "left_spherical_wrist_2_link",
        "left_bracelet_link",
    ]
    right_arm_core = [
        "right_spherical_wrist_1_link",
        "right_spherical_wrist_2_link",
        "right_bracelet_link",
    ]
    left_fingers = [
        ["left_hand_link_0", "left_hand_link_3_tip"],
        ["left_hand_link_4", "left_hand_link_7_tip"],
        ["left_hand_link_8", "left_hand_link_11_tip"],
        ["left_hand_link_12", "left_hand_link_15_tip"],
    ]
    right_fingers = [
        ["right_hand_link_0", "right_hand_link_3_tip"],
        ["right_hand_link_4", "right_hand_link_7_tip"],
        ["right_hand_link_8", "right_hand_link_11_tip"],
        ["right_hand_link_12", "right_hand_link_15_tip"],
    ]

    line_pairs: List[tuple[str, str]] = []

    def add_chain(chain: List[str]) -> None:
        for a, b in zip(chain[:-1], chain[1:]):
            if a in positions and b in positions:
                line_pairs.append((a, b))

    add_chain(left_arm_core)
    add_chain(right_arm_core)
    for base, tip in left_fingers + right_fingers:
        if base in positions and tip in positions:
            line_pairs.append((base, tip))

    # 라인 업데이트: line_pairs와 line_list 순서가 동일하다는 가정
    for (parent_name, child_name), line in zip(line_pairs, line_list):
        if line is None:
            continue
        if parent_name not in positions or child_name not in positions:
            continue
        p0 = positions[parent_name] + offset
        p1 = positions[child_name] + offset
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    # 스캐터 업데이트: link_names에 포함된 링크만
    pts = np.stack([positions[name] + offset for name in link_names], axis=0)
    scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])


# ---------------------------------------------------------------------------
# 조인트 모션 애니메이션
# ---------------------------------------------------------------------------

def animate_joint_motion(
    robot: URDF,
    moving_joint: str,
    start_val: float,
    end_val: float,
    *,
    duration: float = 1.0,
    fps: int = 30,
    view_elev: float = 30.0,
    view_azim: float = -60.0,
    init_fov_deg: float = 120.0,
) -> None:
    """
    moving_joint 하나를 start_val -> end_val 로 duration초 동안 선형 보간.
    카메라는 pitch/yaw/FOV 슬라이더로만 바꾸고,
    스켈레톤은 tx,ty,tz 슬라이더로 평행이동.
    """
    if moving_joint not in JOINT_NAMES:
        raise KeyError(f"알 수 없는 조인트 이름입니다: {moving_joint}")

    base_cfg = make_joint_cfg()
    base_cfg[moving_joint] = start_val

    fig, ax, line_list, scatter, link_names = _init_skeleton_artists(
        robot,
        base_cfg,
        view_elev=view_elev,
        view_azim=view_azim,
        init_fov_deg=init_fov_deg,
    )

    plt.ion()
    fig.show()

    skel_sliders = getattr(fig, "_skel_sliders", {})
    s_tx = skel_sliders.get("tx", None)
    s_ty = skel_sliders.get("ty", None)
    s_tz = skel_sliders.get("tz", None)

    def get_offset() -> np.ndarray:
        if s_tx is None:
            return np.zeros(3, dtype=float)
        return np.array([s_tx.val, s_ty.val, s_tz.val], dtype=float)

    n_frames = max(1, int(duration * fps))
    dt = duration / n_frames

    print(f"[ANIM] moving joint={moving_joint}, "
          f"{start_val:.3f} -> {end_val:.3f}, "
          f"duration={duration}s, fps={fps}, frames={n_frames}")

    t_start = time.perf_counter()

    for i in range(n_frames + 1):
        alpha = i / n_frames
        current_val = (1.0 - alpha) * start_val + alpha * end_val

        cfg = dict(base_cfg)
        cfg[moving_joint] = current_val

        positions = compute_link_positions(robot, cfg)
        fig._last_positions = positions.copy()

        offset = get_offset()
        _update_skeleton_artists(robot, positions, line_list, scatter, link_names, offset)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(dt)

    t_end = time.perf_counter()
    print(f"[ANIM] actual elapsed: {t_end - t_start:.3f} s")

    plt.ioff()
    plt.show()



class SkeletonSpamAnimator:
    def __init__(
        self,
        robot: URDF,
        *,
        fps: int = 30,
        view_elev: float = 180.0,
        view_azim: float = 60.0,
        init_fov_deg: float = 170.0,
    ) -> None:
        self.robot = robot
        self.fps = fps
        self.joint_cfg = make_joint_cfg()  # {joint_name: value}

        # 한 번만 스켈레톤 초기화
        self.fig, self.ax, self.line_list, self.scatter, self.link_names = _init_skeleton_artists(
            self.robot,
            self.joint_cfg,
            view_elev=view_elev,
            view_azim=view_azim,
            init_fov_deg=init_fov_deg,
        )

        plt.ion()
        self.fig.show()

    # 현재 슬라이더(tx, ty, tz)에서 offset 읽기
    def _get_offset_from_sliders(self) -> np.ndarray:
        skel_sliders = getattr(self.fig, "_skel_sliders", {})
        s_tx = skel_sliders.get("tx", None)
        s_ty = skel_sliders.get("ty", None)
        s_tz = skel_sliders.get("tz", None)
        if s_tx is None or s_ty is None or s_tz is None:
            return np.zeros(3, dtype=float)
        return np.array([s_tx.val, s_ty.val, s_tz.val], dtype=float)

    # joint_cfg 기준으로 1프레임 그리기
    def _redraw(self) -> None:
        positions = compute_link_positions(self.robot, self.joint_cfg)
        self.fig._last_positions = positions.copy()
        offset = self._get_offset_from_sliders()
        _update_skeleton_artists(
            self.robot,
            positions,
            self.line_list,
            self.scatter,
            self.link_names,
            offset,
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # joint_names를 target 값으로 duration 동안 선형 보간
    def _animate_to_targets(
        self,
        joint_names: Sequence[str],
        target_vals: np.ndarray,
        duration: float = 1.0,
    ) -> None:
        if len(joint_names) == 0:
            return

        start_vals = np.array(
            [self.joint_cfg[name] for name in joint_names],
            dtype=float,
        )
        n_frames = max(1, int(self.fps * duration))
        dt = duration / n_frames

        for i in range(n_frames + 1):
            alpha = i / n_frames
            cur_vals = (1.0 - alpha) * start_vals + alpha * target_vals

            for name, v in zip(joint_names, cur_vals):
                self.joint_cfg[name] = float(v)

            self._redraw()
            time.sleep(dt)

    # -------------------------------------------------------------------
    # ROS 쪽 publish_1D_joint_position과 동등한 의미로 구현
    #   - amplitude: 현재 값에서 더해줄 Δ값
    #   - duration: 한 번 움직이는 데 걸리는 시간(초)
    # -------------------------------------------------------------------
    def publish_1D_joint_position(
        self,
        joint_names: Sequence[str],
        amplitude: float,
        duration: float = 1.0,
    ) -> None:
        joint_names = list(joint_names)
        # 현재 joint 값에서 amplitude 만큼 이동
        current_vals = np.array(
            [self.joint_cfg[name] for name in joint_names],
            dtype=float,
        )
        target_vals = current_vals + amplitude
        self._animate_to_targets(joint_names, target_vals, duration)

    # -------------------------------------------------------------------
    # ROS grab_object 포트
    # joint: "left_hand_joint_"
    # movement: (start, stop, num_steps)
    # amplitude: joint_12에 한 번에 줄 amplitude
    # extra: {"active": bool, "joint": "left_joint_4", "amplitude": 0.1}
    # -------------------------------------------------------------------
    def grab_object(
        self,
        joint: str,
        movement: tuple,
        amplitude: float,
        extra: Dict,
    ) -> None:
        # fingers: 0~11 중 0,4,8 제외
        finger_joints = [f"{joint}{i}" for i in range(0, 12) if i not in [0, 4, 8]]
        # thumb: 13,14,15
        thumb_joints = [f"{joint}{i}" for i in range(13, 16)]

        # joint12 먼저 닫기
        self.publish_1D_joint_position([f"{joint}12"], amplitude)
        print('closed joint 12')

        # 엄지(thumb) 서서히 닫기
        start, stop, num = movement
        num = int(num)
        for step in np.linspace(start, stop, num):
            self.publish_1D_joint_position(thumb_joints, step)
            time.sleep(0.05)
        print('closed thumb joints')

        # extra joint 추가 동작 (예: left_joint_4 살짝 더 굽히기)
        if extra.get("active", False):
            extra_joint = extra.get("joint")
            extra_amp = extra.get("amplitude", 0.0)
            if extra_joint is not None:
                self.publish_1D_joint_position([extra_joint], extra_amp)
        print('moved extra joint if any')

        # 손가락(fingers) 서서히 닫기
        for step in np.linspace(start, stop, num):
            self.publish_1D_joint_position(finger_joints, step)
            time.sleep(0.05)

    # -------------------------------------------------------------------
    # ROS release_object 포트
    # -------------------------------------------------------------------
    def release_object(
        self,
        joint: str,
        movement: tuple,
        amplitude: float,
    ) -> None:
        finger_joints = [f"{joint}{i}" for i in range(0, 12) if i not in [0, 4, 8]]
        thumb_joints = [f"{joint}{i}" for i in range(13, 16)]

        # 엄지 서서히 펴기
        start, stop, num = movement
        num = int(num)
        for step in np.linspace(start, stop, num):
            self.publish_1D_joint_position(thumb_joints, step)
            time.sleep(0.05)

        # joint12 열기
        self.publish_1D_joint_position([f"{joint}12"], amplitude)

        # 손가락 서서히 펴기
        for step in np.linspace(start, stop, num):
            self.publish_1D_joint_position(finger_joints, step)
            time.sleep(0.05)

    # -------------------------------------------------------------------
    # 실제 스팸 집기-이동-놓기 시퀀스
    #   원래 ROS 코드와 동일한 순서/명령
    # -------------------------------------------------------------------
    def run_spam_sequence(self) -> None:
        print("reach the spam..")
        self.publish_1D_joint_position(["left_joint_1"], 0.8)
        self.publish_1D_joint_position(["left_joint_7"], -1.6)
        self.publish_1D_joint_position(["left_joint_4"], 1.0)

        print("grab the spam..")
        self.grab_object(
            "left_hand_joint_",
            (0.0, 0.25, 5),  # np.linspace(0, 0.25, 5)
            1.25,
            {"active": True, "joint": "left_joint_4", "amplitude": 0.02},
        )
        time.sleep(0.5)

        print("lift the spam..")
        self.publish_1D_joint_position(["left_joint_4"], -0.6)
        time.sleep(0.5)

        print("move the spam..")
        self.publish_1D_joint_position(["left_joint_2"], -0.3)
        self.publish_1D_joint_position(["left_joint_4"], 0.6)
        time.sleep(0.5)

        print("release the spam..")
        self.release_object("left_hand_joint_", (-0.25, 0.0, 5), -5.0)
        self.publish_1D_joint_position(["left_joint_7"], 1.6)
        self.publish_1D_joint_position(["left_joint_4"], -1.0)
        time.sleep(0.5)

# ---------------------------------------------------------------------------
# 예시 실행
# ---------------------------------------------------------------------------

def main() -> None:
    robot = load_robot()

    # 스켈레톤에서 스팸 집기 시퀀스 실행
    animator = SkeletonSpamAnimator(
        robot,
        fps=30,
        view_elev=180.0,
        view_azim=60.0,
        init_fov_deg=170.0,
    )
    animator.run_spam_sequence()

    # 애니메이션 끝나도 창 유지
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
