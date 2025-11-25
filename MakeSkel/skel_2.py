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

    fig = plt.figure(figsize=(8, 8))
    ax: Axes3D = fig.add_axes([0.05, 0.32, 0.9, 0.63], projection="3d")

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    # ----------------- 링크 라인 / 점 생성 -----------------
    line_list: List[Optional[plt.Line2D]] = []
    for joint in robot.joints:
        parent_name = joint.parent
        child_name = joint.child
        if parent_name in positions and child_name in positions:
            p0 = positions[parent_name]
            p1 = positions[child_name]
            xs = [p0[0], p1[0]]
            ys = [p0[1], p1[1]]
            zs = [p0[2], p1[2]]
            line, = ax.plot(xs, ys, zs,
                            linewidth=1.5,
                            color="magenta",
                            alpha=0.9)
            line_list.append(line)
        else:
            line_list.append(None)

    link_names = list(positions.keys())
    pts = np.stack([positions[name] for name in link_names], axis=0)
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=20.0,
        c="yellow",
        depthshade=True,
    )

    # ===========================
    # 1) 머리 위치 & "척추 방향" 계산
    # ===========================
    head_pos = get_head_link_position(robot, joint_cfg)

    # z가 가장 낮은 점을 "발/바닥 쪽"으로 가정
    min_z_idx = np.argmin(pts[:, 2])
    root_pos = pts[min_z_idx]

    spine_vec = root_pos - head_pos          # head -> root 방향 (대략 척추 방향)
    spine_len = float(np.linalg.norm(spine_vec))
    if spine_len < 1e-6:
        spine_len = 0.5

    # ===========================
    # 2) 프레이밍: 머리~가슴 기준으로 축 고정
    # ===========================
    center = head_pos + 0.3 * spine_vec      # 머리에서 약간 아래
    half = 0.5 * spine_len                   # 필요하면 숫자 키워도 됨

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # ===========================
    # 3) 카메라 초기 pitch/yaw
    # ===========================
    cam_vec = head_pos - center  # center 기준 카메라 위치 벡터
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
    #   - pitch, yaw, FOV (카메라)
    #   - tx, ty, tz (스켈레톤 평행이동)
    # ------------------------------------------------------------------
    axcolor = "#222222"
    ax_pitch = fig.add_axes([0.05, 0.25, 0.9, 0.02], facecolor=axcolor)
    ax_yaw   = fig.add_axes([0.05, 0.21, 0.9, 0.02], facecolor=axcolor)
    ax_fov   = fig.add_axes([0.05, 0.17, 0.9, 0.02], facecolor=axcolor)
    ax_tx    = fig.add_axes([0.05, 0.13, 0.9, 0.02], facecolor=axcolor)
    ax_ty    = fig.add_axes([0.05, 0.09, 0.9, 0.02], facecolor=axcolor)
    ax_tz    = fig.add_axes([0.05, 0.05, 0.9, 0.02], facecolor=axcolor)

    # 카메라 RPY(여기서는 pitch, yaw 두 개) + FOV
    s_pitch = Slider(ax_pitch, "pitch", -180.0, 180.0, valinit=pitch_init)
    s_yaw   = Slider(ax_yaw,   "yaw",   -180.0, 180.0, valinit=yaw_init)
    s_fov   = Slider(ax_fov,   "FOV",    10.0, 170.0, valinit=init_fov_deg)

    # 스켈레톤 평행이동 범위 (척추 길이 기준으로 대략 설정)
    trans_range = spine_len*0.5
    s_tx = Slider(ax_tx, "tx", -trans_range, trans_range, valinit=0.0)
    s_ty = Slider(ax_ty, "ty", -trans_range, trans_range, valinit=0.0)
    s_tz = Slider(ax_tz, "tz", -trans_range, trans_range, valinit=0.0)

    # fig에 현재 로봇 좌표 저장 (슬라이더에서 다시 그릴 때 사용)
    fig._last_positions = positions

    # ----------------- 콜백: 카메라 -----------------
    def update_camera(_val=None) -> None:
        pitch = s_pitch.val
        yaw = s_yaw.val
        fov_deg = s_fov.val

        ax.view_init(elev=pitch, azim=yaw)
        focal = f_from_fov_deg(fov_deg)
        ax.set_proj_type("persp", focal_length=focal)
        fig.canvas.draw_idle()

    # ----------------- 콜백: 스켈레톤 평행이동 -----------------
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
        fig.canvas.draw_idle()

    for s in (s_pitch, s_yaw, s_fov):
        s.on_changed(update_camera)
    for s in (s_tx, s_ty, s_tz):
        s.on_changed(update_skeleton)

    # 초기 1회 적용
    update_camera(None)
    update_skeleton(None)

    # GC 방지용 레퍼런스
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
    robot: URDF,
    positions: Dict[str, np.ndarray],
    line_list: List,
    scatter,
    link_names: List[str],
    offset: Optional[np.ndarray] = None,
) -> None:
    """현재 positions + offset에 맞게 라인/스캐터 아티스트를 업데이트."""
    if offset is None:
        offset = np.zeros(3, dtype=float)

    # 라인 업데이트
    for joint, line in zip(robot.joints, line_list):
        if line is None:
            continue
        parent_name = joint.parent
        child_name = joint.child
        if parent_name not in positions or child_name not in positions:
            continue
        p0 = positions[parent_name] + offset
        p1 = positions[child_name] + offset
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    # 스캐터 업데이트
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

    # 스켈레톤 슬라이더 참조 (매 프레임 offset 읽어오기 위함)
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
        # 슬라이더에서 사용할 현재 포즈 저장
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


# ---------------------------------------------------------------------------
# 예시 실행
# ---------------------------------------------------------------------------

def main() -> None:
    robot = load_robot()

    # 조인트 모션 애니메이션 예시:
    animate_joint_motion(
        robot,
        moving_joint="left_joint_3",
        start_val=math.pi / 2,
        end_val=math.pi / 2,   # 지금은 고정 포즈용으로 동일 값
        duration=1000.0,
        fps=30,
        view_elev=45.0,
        view_azim=180.0,
        init_fov_deg=120.0,
    )


if __name__ == "__main__":
    main()
