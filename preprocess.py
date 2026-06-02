import json
import os
import re
import glob
import pandas as pd

# --- 경로 설정 ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# 수집된 jsonl 파일들이 모여있는 steam_data 폴더 지정
INPUT_DIR = os.path.join(BASE_PATH, "steam_data")
OUTPUT_FILE = os.path.join(INPUT_DIR, "merged_cleaned_reviews.csv")

def is_valid_korean(text, threshold=0.3):
    """리뷰 본문에서 한글이 차지하는 비율이 기준 이상인 경우만 통과시킵니다."""
    if not text or not isinstance(text, str):
        return False
    
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return False
    
    # 한글 유니코드 범위 매칭
    korean_chars = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text))
    korean_ratio = korean_chars / total_chars
    return korean_ratio >= threshold

def clean_text(text):
    """스팀 리뷰 내의 불필요한 BBCode 태그 및 연속 공백을 정제합니다."""
    text = re.sub(r'\[.*?\]', '', text)  # [b], [/b] 등 태그 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 줄바꿈 및 연속 공백을 공백 하나로 통일
    return text

def merge_and_preprocess():
    # 1. 대상 폴더 안의 모든 jsonl 파일 목록 가져오기
    search_pattern = os.path.join(INPUT_DIR, "*.jsonl")
    jsonl_files = glob.glob(search_pattern)
    
    if not jsonl_files:
        print(f"오류: {INPUT_DIR} 경로에서 .jsonl 파일을 찾을 수 없습니다.")
        return

    print(">>> 1. 장르별 데이터 파일 읽기 및 통합 시작...")
    all_raw_data = []
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        print(f"  - 파일 로드 중: {file_name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_raw_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(all_raw_data)
    print(f"  - 통합 완료: 총 {len(df):,}개 로우 데이터 결합됨.")

    # 2. 중복 제거 (핵심 단계)
    print("\n>>> 2. 중복 데이터(중복 게임 및 리뷰) 제거 중...")
    initial_count = len(df)
    
    # 데이터 구조에 recommendationid가 포함되어 있다면 그것을 우선으로 하고, 
    # 없을 경우 'review' 본문 내용 자체를 기준으로 중복을 완전히 제거합니다.
    if 'recommendationid' in df.columns:
        df.drop_duplicates(subset=['recommendationid'], keep='first', inplace=True)
    else:
        df.drop_duplicates(subset=['review'], keep='first', inplace=True)
        
    print(f"  - 중복 제거 완료: {initial_count:,}개 -> {len(df):,}개 (차이: {initial_count - len(df):,}개 제거됨)")

    # 3. 데이터 정제 및 외국어 필터링
    print("\n>>> 3. 한글 리뷰 검수 및 불필요 텍스트 정제 중...")
    df.dropna(subset=['review'], inplace=True)
    
    # 텍스트 정제 (태그 제거)
    df['review'] = df['review'].apply(clean_text)
    
    # 외국어 및 단순 이모티콘/자음 나열 리뷰 필터링 (한글 비율 30% 이상만 통과)
    df = df[df['review'].apply(lambda x: is_valid_korean(x, threshold=0.3))]
    print(f"  - 최종 유효 리뷰 선별 완료: {len(df):,}개 남음.")

    # 4. 최종 CSV 저장
    print("\n>>> 4. 병합 및 정제가 완료된 대시보드용 CSV 파일 저장 중...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  - 원스톱 전처리 완료! 파일 경로: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_and_preprocess()