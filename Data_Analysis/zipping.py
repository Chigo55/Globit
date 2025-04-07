import os
import zipfile
from tqdm.auto import tqdm


def zip_multiple_folders(folder_paths, output_zip_path):
    # 유효한 폴더만 필터링
    valid_folders = []
    for folder in folder_paths:
        if os.path.exists(folder):
            valid_folders.append(folder)
        else:
            print(f"⚠️ 경고: 폴더가 존재하지 않습니다 → {folder}")

    if not valid_folders:
        print("⛔ 압축할 유효한 폴더가 없습니다. 작업을 종료합니다.")
        return

    # 기존 zip 파일이 존재할 경우 덮어쓰기 확인
    if os.path.exists(output_zip_path):
        response = input(f"⚠️ '{output_zip_path}' 파일이 이미 존재합니다. 덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("⛔ 압축 작업이 취소되었습니다.")
            return

    # 압축 시작
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_path in tqdm(valid_folders, desc="📁 폴더 압축 중"):
            folder_base = os.path.basename(folder_path.rstrip('/\\'))  # 폴더 이름만 추출
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_full_path = os.path.join(root, file)
                    arcname = os.path.join(folder_base, os.path.relpath(file_full_path, folder_path))
                    arcname = arcname.replace("\\", "/")  # 경로 통일
                    zipf.write(file_full_path, arcname)

    # 압축 결과 출력
    zip_size = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"\n✅ 압축 완료: {output_zip_path}")
    print(f"📦 ZIP 파일 크기: {zip_size:.2f} MB")


# 출력 zip 파일 경로
folder = "predict"
output_zip_file = f'./zip/{folder}.zip'

# 압축할 여러 폴더 설정
folders_to_zip = []
optimizers = ["auto", "SGD", "Adam", 'AdamW', "NAdam", "RAdam", "RMSProp"]
for idx, opt in enumerate(optimizers):
    folders_to_zip.append(f'runs/pose/{folder}/{idx+1}_{opt}')

# 압축 실행
print("📝 압축 대상 폴더:")
for path in folders_to_zip:
    print(f" - {path}")

zip_multiple_folders(folders_to_zip, output_zip_file)
