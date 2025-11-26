from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

DEFAULT_JOINT_VALUES = np.array([
    0.2,
    0.0,
    1.0559,

    1.5708,
    -1.5708,
    1.5708,
    -1.5708,
    0.0,
    0.0,
    0.0,

    1.5708,
    1.5708,
    -1.5708,
    1.5708,
    0.0,
    0.0,
    0.0,

    *[0.0] * 16,
    *[0.0] * 16,
], dtype=float)

URDF_FILENAME = "etri_dualarm_robot.urdf"
HEAD_LINK_NAME = "head_camera_color_optical_frame"


def get_head_link_position(robot: URDF, joint_cfg: Dict[str, float]) -> np.ndarray:
    fk = robot.link_fk(cfg=joint_cfg)
    for link, T in fk.items():
        if link.name == HEAD_LINK_NAME:
            pos = np.asarray(T[:3, 3], dtype=float).reshape(3)
            return pos
    raise KeyError


def compute_link_positions(robot: URDF, joint_cfg: Dict[str, float]) -> Dict[str, np.ndarray]:
    fk = robot.link_fk(cfg=joint_cfg)
    return {link.name: np.asarray(T[:3, 3]).reshape(3) for link, T in fk.items()}


# ---------------------------------------------------------------------------
# 스켈레톤 초기화 (수정됨)
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

    # ----------------------------
    # 요청한 구성으로만 라인 생성
    # ----------------------------

    left_arm_core = [
        "left_spherical_wrist_1_link",
        "left_spherical_wirst_2_link",
        "left_bracelet_link",
    ]
    right_arm_core = [
        "right_spherical_wrist_1_link",
        "right_spherical_wirst_2_link",
        "right_bracelet_link",
    ]

    left_finger_bases = [
        "left_hand_link_0",
        "left_hand_link_4",
        "left_hand_link_8",
        "left_hand_link_12",
    ]
    right_finger_bases = [
        "right_hand_link_0",
        "right_hand_link_4",
        "right_hand_link_8",
        "right_hand_link_12",
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

    def add_chain(chain: List[str]):
        for a, b in zip(chain[:-1], chain[1:]):
            if a in positions and b in positions:
                line_pairs.append((a, b))

    add_chain(left_arm_core)
    add_chain(right_arm_core)

    # wrist → finger base 연결
    for base in left_finger_bases:
        if "left_bracelet_link" in positions and base in positions:
            line_pairs.append(("left_bracelet_link", base))
    for base in right_finger_bases:
        if "right_bracelet_link" in positions and base in positions:
            line_pairs.append(("right_bracelet_link", base))

    # finger base → tip
    for base, tip in left_fingers + right_fingers:
        if base in positions and tip in positions:
            line_pairs.append((base, tip))

    # -------- 점을 찍을 링크들(선에 등장한 링크만)
    link_names = sorted({x for pair in line_pairs for x in pair})

    # ------------------------
    # Figure 생성
    # ------------------------
    fig = plt.figure(figsize=(8, 8))
    ax: Axes3D = fig.add_axes([0.05, 0.32, 0.9, 0.63], projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    # ---- 라인 생성 ----
    line_list: List = []
    for p, c in line_pairs:
        p0 = positions[p]
        p1 = positions[c]
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]
        line, = ax.plot(xs, ys, zs, color="magenta", linewidth=1.5, alpha=0.9)
        line_list.append(line)

    # ---- 점 생성 ----
    pts = np.stack([positions[n] for n in link_names], axis=0)
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=20,
        c="yellow",
        depthshade=False,
        alpha=1.0,
    )


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
