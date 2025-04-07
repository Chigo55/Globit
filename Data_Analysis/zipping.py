import os
import zipfile
from tqdm.auto import tqdm


def zip_folders(folder):
    # 압축할 폴더들을 생성 (옵티마이저 리스트 기준)
    optimizers = ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]
    folders_to_zip = [
        f"Data_Analysis/runs/{folder}/{idx+1}_{opt}" for idx, opt in enumerate(optimizers)]

    # 출력 ZIP 파일 경로 설정
    output_zip_file = f"./Data_Analysis/runs/zip/{folder}.zip"

    # 압축 대상 폴더 목록 출력
    print("📝 압축 대상 폴더:")
    for path in folders_to_zip:
        print(f" - {path}")

    # 유효한 폴더만 필터링
    valid_folders = []
    for f_path in folders_to_zip:
        if os.path.exists(f_path):
            valid_folders.append(f_path)
        else:
            print(f"⚠️ 경고: 폴더가 존재하지 않습니다 → {f_path}")

    if not valid_folders:
        print("⛔ 압축할 유효한 폴더가 없습니다. 작업을 종료합니다.")
        return

    # 기존 zip 파일이 존재하면 덮어쓰기 확인
    if os.path.exists(output_zip_file):
        response = input(
            f"⚠️ '{output_zip_file}' 파일이 이미 존재합니다. 덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("⛔ 압축 작업이 취소되었습니다.")
            return

    # 압축 시작
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_path in tqdm(valid_folders, desc="📁 폴더 압축 중"):
            # 폴더 이름만 추출 (경로 통일)
            folder_base = os.path.basename(folder_path.rstrip('/\\'))
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_full_path = os.path.join(root, file)
                    arcname = os.path.join(
                        folder_base, os.path.relpath(file_full_path, folder_path))
                    arcname = arcname.replace("\\", "/")
                    zipf.write(file_full_path, arcname)

    zip_size = os.path.getsize(output_zip_file) / (1024 * 1024)
    print(f"✅ 압축 완료: {output_zip_file}")
    print(f"📦 ZIP 파일 크기: {zip_size:.2f} MB")


if __name__ == '__main__':
    # 사용자는 folder 변수만 입력하면 됩니다.
    folder = input("압축할 폴더: ")
    zip_folders(folder)
