import os

# 변환할 파일 확장자 목록
TARGET_EXTENSIONS = ('.c', '.h', '.cpp', '.cl', '.txt')

def convert_to_utf8_bom(path):
    # 1. 파일 읽기 시도 (UTF-8 -> CP949 순서로 시도)
    content = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='cp949') as f:
                content = f.read()
        except Exception as e:
            print(f"[Skip] {path} : 읽기 실패 ({e})")
            return

    if content is None: return

    # 2. UTF-8 (BOM 포함)으로 덮어쓰기
    # 'utf-8-sig'가 BOM을 추가해주는 인코딩입니다.
    try:
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write(content)
        print(f"[Converted] {path}")
    except Exception as e:
        print(f"[Error] {path} : 쓰기 실패 ({e})")

def main():
    # 현재 폴더(.)부터 하위 폴더까지 모두 탐색
    current_dir = os.getcwd()
    print(f"Start scanning in: {current_dir}")
    
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.lower().endswith(TARGET_EXTENSIONS):
                full_path = os.path.join(root, file)
                convert_to_utf8_bom(full_path)

if __name__ == "__main__":
    main()