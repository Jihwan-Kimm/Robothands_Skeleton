import cv2
import numpy as np

def detect_joint_centers(image_path, visualize=False):
    # 이미지 읽기 (BGR 형식)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")

    # ----- 1. 노란색 조인트만 마스크 -----
    # 이 이미지에서 조인트 색은 RGB(255,255,0) -> OpenCV BGR 기준으로 (0,255,255)
    yellow = np.array([0, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, yellow, yellow)  # 완전 동일한 색만 사용한다고 했으니 lower=upper

    # ----- 2. 연결 요소별로 분리 & 중심점 계산 -----
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # labels 0번은 배경이므로 제외
    centers = []
    for idx, (cx, cy) in enumerate(centroids[1:], start=1):
        centers.append((float(cx), float(cy)))

    # 보기 좋게 x, y 기준으로 정렬 (원하지 않으면 이 줄 삭제)
    centers.sort(key=lambda p: (p[0], p[1]))

    # ----- 3. 결과 출력 -----
    for i, (x, y) in enumerate(centers):
        # OpenCV 좌표계: (x=열, y=행), 원점은 좌상단
        print(f"joint {i:02d}: x={x:.2f}, y={y:.2f}")

    # ----- 4. 디버그용 시각화 (옵션) -----
    if visualize:
        vis = img.copy()
        for (x, y) in centers:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)  # 빨간 점으로 표시
        cv2.imshow("joints", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return centers

if __name__ == "__main__":
    image_path = "images/skeleton_000010.png"  # 파일 이름/경로에 맞게 수정
    centers = detect_joint_centers(image_path, visualize=True)
