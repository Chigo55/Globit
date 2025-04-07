import argparse
import torch

from Src.FishSize import VideoProcessor


def parse_opt():
    """
    명령줄 인자를 파싱하여 옵션 객체를 반환합니다.

    Returns:
        argparse.Namespace: 파싱된 옵션 객체.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./Model/Custom/pose_nano.pt', help='모델 가중치 파일 경로')
    parser.add_argument('--source', type=str, default='./Data/Images/infrared/train/003500.jpg', help='데이터 소스 (파일/폴더 또는 0: 웹캠)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='객체 신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS에 사용할 IOU 임계값')
    parser.add_argument('--device', default='cuda', help='사용할 디바이스 (예: 0 또는 cpu)')

    parser.add_argument("--use-webcam", action="store_true", help="웹캠 사용 여부")
    parser.add_argument('--verbose', action='store_true', help='각 작업에 대한 로그 반환 여부')
    parser.add_argument('--visualize', action='store_true', help='결과를 화면에 표시')
    parser.add_argument('--save', action='store_true', help='이미지/비디오 결과 저장 여부')
    parser.add_argument('--save-txt', action='store_true', help='예측 결과를 텍스트 파일로 저장 여부')
    parser.add_argument('--project', default='runs/pose', help='결과 저장 경로 (프로젝트 경로)')
    parser.add_argument('--name', default='predict', help='결과 저장 경로 내 폴더 이름')
    parser.add_argument('--exist-ok', action='store_true', help='기존 폴더가 있어도 덮어쓰기')
    opt = parser.parse_args()
    return opt


def main(opt):
    """
    추론 모드를 사용하여 VideoProcessor를 실행합니다.

    Args:
        opt (argparse.Namespace): 파싱된 명령줄 옵션 객체.
    """
    with torch.inference_mode():
        video_processor = VideoProcessor(opt)
        # video_processor.run()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
