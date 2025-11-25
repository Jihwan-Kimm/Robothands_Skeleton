import cv2
import numpy as np
import math

# 같은 조인트로 간주할 두 중심점 사이 최대 거리 (픽셀)
MERGE_DIST = 8.0   # 필요하면 6~12 정도에서 튜닝

def merge_close_centers(centers, areas, merge_dist):
    """
    centers : (N, 2) 배열, 각 연결요소의 중심 (x, y)
    areas   : (N,) 배열, 각 연결요소의 픽셀 수 (면적)
    merge_dist : 같은 조인트로 묶을 거리 임계값
    """
    pts = [np.array(c, dtype=float) for c in centers]
    used = [False] * len(pts)
    merged = []

    for i, p in enumerate(pts):
        if used[i]:
            continue

        cluster_idx = [i]
        used[i] = True

        # 거리 merge_dist 이내에 있는 조각들을 같은 클러스터로 묶기
        changed = True
        while changed:
            changed = False
            for j, q in enumerate(pts):
                if used[j]:
                    continue
                # 이미 클러스터에 들어간 어떤 점과라도 가까우면 같은 클러스터로
                if any(math.dist(q, pts[k]) <= merge_dist for k in cluster_idx):
                    used[j] = True
                    cluster_idx.append(j)
                    changed = True

        # 면적 가중 평균으로 최종 조인트 중심 계산
        total_area = sum(areas[k] for k in cluster_idx)
        cx = sum(areas[k] * pts[k][0] for k in cluster_idx) / total_area
        cy = sum(areas[k] * pts[k][1] for k in cluster_idx) / total_area
        merged.append((cx, cy))

    return merged


def detect_joint_centers(image_path, visualize=False):
    # 1. 이미지 읽기 (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")

    # 2. 노란색 조인트만 바이너리 마스크로 추출
    #   이 이미지의 색: 배경 (0,0,0), 조인트 (0,255,255), 뼈 (255,0,255) [BGR]
    yellow = np.array([0, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, yellow, yellow)

    # 3. 연결요소 분석으로 각 조각의 중심과 면적 계산
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # label 0은 배경이므로 제외
    raw_centers = centroids[1:]                      # shape (N, 2), (x, y)
    areas = stats[1:, cv2.CC_STAT_AREA]             # shape (N,)

    # 4. 가까운 조각들을 하나의 조인트로 병합
    joint_centers = merge_close_centers(raw_centers, areas, MERGE_DIST)

    # 보기 좋게 x, y 기준으로 정렬 (원하지 않으면 생략)
    joint_centers.sort(key=lambda p: (p[0], p[1]))

    # 5. 결과 출력
    for i, (x, y) in enumerate(joint_centers):
        print(f"joint {i:02d}: x={x:.2f}, y={y:.2f}")

    # 6. 디버그용 시각화
    if visualize:
        vis = img.copy()
        for (x, y) in joint_centers:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), 2)  # 최종 조인트 중심 표시
        cv2.imshow("joints", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return joint_centers


if __name__ == "__main__":
    image_path = "C:\\Users\\rtss\\Desktop\\git\\Robothands_Skeleton\\DetectJoints\\images\\skeleton_000037.png"  # 파일 이름/경로에 맞게 수정
    centers = detect_joint_centers(image_path, visualize=True)
