def _init_skeleton_artists(
    robot: URDF,
    joint_cfg: Dict[str, float],
    *,
    view_elev: float = 30.0,
    view_azim: float = -60.0,
    view_zoom: float = 1.0,
):
    """
    스켈레톤 + 카메라 슬라이더 초기화.

    슬라이더:
      - elev, azim : 시점 회전
      - zoom      : 전체 크기 확대/축소
      - H span    : 좌우 시야(가로 범위)만 추가로 넓힘
      - dx, dy, dz: 중심 위치 이동
      - FOV (deg) : persp 투영의 field of view (10 ~ 170 deg)
                    -> focal_length = 1 / tan(FOV/2)
    """

    positions = compute_link_positions(robot, joint_cfg)

    fig = plt.figure(figsize=(8, 8))
    # 3D 영역을 약간 위로 올려서 슬라이더 8개 들어갈 자리 확보
    ax: Axes3D = fig.add_axes([0.05, 0.36, 0.9, 0.6], projection="3d")

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
                            color="red",
                            alpha=0.9)
            line_list.append(line)
        else:
            line_list.append(None)

    link_names = list(positions.keys())
    pts = np.stack([positions[name] for name in link_names], axis=0)
    scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=20.0,
        c="lime",
        depthshade=True,
    )

    # ----------------- 기본 center / range -----------------
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    base_center = 0.5 * (xyz_min + xyz_max)
    base_half = float((xyz_max - xyz_min).max())
    if base_half == 0.0:
        base_half = 0.1
    base_half *= 0.6  # 약간 여유

    # ----------------- 슬라이더 축 -----------------
    axcolor = "#222222"
    # 위에서부터: H span, elev, azim, zoom, dx, dy, dz, FOV
    ax_hspan = fig.add_axes([0.05, 0.29, 0.9, 0.02], facecolor=axcolor)
    ax_elev  = fig.add_axes([0.05, 0.25, 0.9, 0.02], facecolor=axcolor)
    ax_azim  = fig.add_axes([0.05, 0.21, 0.9, 0.02], facecolor=axcolor)
    ax_zoom  = fig.add_axes([0.05, 0.17, 0.9, 0.02], facecolor=axcolor)
    ax_cx    = fig.add_axes([0.05, 0.13, 0.9, 0.02], facecolor=axcolor)
    ax_cy    = fig.add_axes([0.05, 0.09, 0.9, 0.02], facecolor=axcolor)
    ax_cz    = fig.add_axes([0.05, 0.05, 0.9, 0.02], facecolor=axcolor)
    ax_fov   = fig.add_axes([0.05, 0.01, 0.9, 0.02], facecolor=axcolor)

    # ----------------- 슬라이더 생성 -----------------
    zoom_init = min(max(view_zoom, 0.05), 20.0)
    fov_init = 90.0  # 기본 FOV

    # 가로 시야 배율 (1.0이면 정방형, 2.0이면 X축 범위를 2배로 넓게 잡음)
    s_hspan = Slider(ax_hspan, "H span", 1.0, 5.0, valinit=1.0)

    s_elev = Slider(ax_elev, "elev", -270.0, 270.0, valinit=view_elev)
    s_azim = Slider(ax_azim, "azim", -360.0, 360.0, valinit=view_azim)
    s_zoom = Slider(ax_zoom, "zoom", 0.05, 20.0,  valinit=zoom_init)

    s_cx = Slider(ax_cx, "dx", -3.0, 3.0, valinit=0.0)
    s_cy = Slider(ax_cy, "dy", -3.0, 3.0, valinit=0.0)
    s_cz = Slider(ax_cz, "dz", -3.0, 3.0, valinit=0.0)

    s_fov = Slider(ax_fov, "FOV (deg)", 10.0, 170.0, valinit=fov_init)

    def update_view(_val=None) -> None:
        # center 이동
        cx = base_center[0] + s_cx.val
        cy = base_center[1] + s_cy.val
        cz = base_center[2] + s_cz.val

        zoom = max(s_zoom.val, 0.05)

        # FOV → focal_length 변환
        fov_deg = float(s_fov.val)
        fov_deg = min(max(fov_deg, 1.0), 179.0)
        fov_rad = math.radians(fov_deg)
        focal_length = 1.0 / math.tan(fov_rad / 2.0)

        # persp + focal_length 적용
        ax.set_proj_type("persp", focal_length=focal_length)

        # H span: 좌우(X축) 범위만 추가로 넓힘
        hspan = max(s_hspan.val, 1.0)

        half_y = base_half / zoom
        half_z = base_half / zoom
        half_x = base_half * hspan / zoom  # 여기서만 hspan 곱

        ax.set_xlim(cx - half_x, cx + half_x)
        ax.set_ylim(cy - half_y, cy + half_y)
        ax.set_zlim(cz - half_z, cz + half_z)

        ax.view_init(elev=s_elev.val, azim=s_azim.val)
        fig.canvas.draw_idle()

    for s in (s_hspan, s_elev, s_azim, s_zoom, s_cx, s_cy, s_cz, s_fov):
        s.on_changed(update_view)

    # 초기 1회 적용
    update_view(None)

    # GC 방지용 레퍼런스
    fig._view_sliders = {
        "H span": s_hspan,
        "elev": s_elev,
        "azim": s_azim,
        "zoom": s_zoom,
        "dx": s_cx,
        "dy": s_cy,
        "dz": s_cz,
        "fov": s_fov,
    }

    return fig, ax, line_list, scatter, link_names