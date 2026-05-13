import requests
import time
import json
import os

# --- 설정 (Configuration) ---
TARGET_TOTAL_REVIEWS = 500000
REVIEWS_PER_GAME_LIMIT = 50000
LANGUAGES = ['koreana']  # 한국어 전용 설정

# 수집하고 싶은 장르 리스트 (SteamSpy 기준 명칭)
TARGET_GENRES = ['Simulation']

# 경로 설정
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_PATH, "steam_data")
PROGRESS_FILE = os.path.join(SAVE_DIR, "progress.json")
DATA_FILE = os.path.join(SAVE_DIR, "all_reviews.jsonl")

# 가변 슬립 설정
BASE_SLEEP = 1.5
MAX_SLEEP = 60.0
BACKOFF_FACTOR = 2.0
SUCCESS_REDUCTION = 0.1

os.makedirs(SAVE_DIR, exist_ok=True)

def load_progress():
    """저장된 진행 상황을 불러옵니다. genre_index 항목을 추가했습니다."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "genre_index": 0,    # 현재 수집 중인 장르 번호
        "game_index": 0,     # 해당 장르 내의 게임 번호
        "lang_index": 0,
        "cursor": "*",
        "total_collected": 0,
        "current_game_collected": 0
    }

def save_progress(progress):
    """현재 진행 상황을 파일에 저장합니다."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

def get_games_by_genre(genre_name):
    """지정한 장르의 게임 리스트를 총 리뷰 수(긍정+부정) 순으로 정렬하여 가져옵니다."""
    print(f"\n>>> '{genre_name}' 장르 게임 리스트 로드 중...")
    url = f"https://steamspy.com/api.php?request=genre&genre={genre_name}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # 정렬 기준 수정: positive + negative 합계가 큰 순서대로
            return sorted(data.values(), 
                        key=lambda x: int(x.get('positive', 0)) + int(x.get('negative', 0)), 
                        reverse=True)
        else:
            print(f"SteamSpy 오류: {response.status_code}")
    except Exception as e:
        print(f"리스트 로드 중 오류 발생: {e}")
    return []

def collect_reviews():
    progress = load_progress()
    current_sleep = BASE_SLEEP
    
    print(f"재개 지점: 장르[{TARGET_GENRES[progress['genre_index']]}] (인덱스 {progress['genre_index']}), "
        f"게임 인덱스 {progress['game_index']}")

    # 'a' 모드로 열어 기존 데이터 뒤에 추가
    with open(DATA_FILE, "a", encoding="utf-8") as f_out:
        # 1. 장르 루프
        for r_idx in range(progress['genre_index'], len(TARGET_GENRES)):
            current_genre = TARGET_GENRES[r_idx]
            genre_games = get_games_by_genre(current_genre)
            
            if not genre_games:
                continue

            # 2. 게임 루프 (장르가 바뀌면 0부터, 아니면 저장된 index부터)
            start_g_idx = progress['game_index'] if r_idx == progress['genre_index'] else 0
            
            for g_idx in range(start_g_idx, len(genre_games)):
                game = genre_games[g_idx]
                appid = game['appid']
                game_name = game['name']
                
                # 장르나 게임이 바뀌면 언어 인덱스 초기화
                start_l_idx = progress['lang_index'] if (r_idx == progress['genre_index'] and g_idx == progress['game_index']) else 0
                
                for l_idx in range(start_l_idx, len(LANGUAGES)):
                    lang = LANGUAGES[l_idx]
                    
                    # 수집 위치가 바뀌면 커서와 수집량 초기화
                    if r_idx != progress['genre_index'] or g_idx != progress['game_index'] or l_idx != progress['lang_index']:
                        progress['cursor'] = "*"
                        progress['current_game_collected'] = 0
                    
                    print(f"\n>>> [{current_genre}] {game_name} ({lang}) 수집 중...")

                    while progress['current_game_collected'] < REVIEWS_PER_GAME_LIMIT:
                        if progress['total_collected'] >= TARGET_TOTAL_REVIEWS:
                            print("\n전체 목표 수치 달성!")
                            return

                        url = f"https://store.steampowered.com/appreviews/{appid}"
                        params = {
                            'json': 1, 'filter': 'all', 'language': lang,
                            'cursor': progress['cursor'], 'num_per_page': 100, 'purchase_type': 'all'
                        }

                        try:
                            resp = requests.get(url, params=params, timeout=15)
                            
                            if resp.status_code == 429:
                                print(f"\n[제한] 429 에러. {current_sleep}초 대기 후 슬립 증가.")
                                time.sleep(current_sleep)
                                current_sleep = min(MAX_SLEEP, current_sleep * BACKOFF_FACTOR)
                                continue
                            
                            if resp.status_code != 200:
                                time.sleep(10)
                                continue

                            data = resp.json()
                            reviews = data.get('reviews', [])
                            if not reviews: break

                            for r in reviews:
                                clean = {
                                    'appid': appid, 
                                    'game': game_name, 
                                    'genre': current_genre, # 장르 정보 추가
                                    'lang': lang,
                                    'review': r['review'], 
                                    'voted_up': r['voted_up']
                                }
                                f_out.write(json.dumps(clean, ensure_ascii=False) + "\n")

                            # 파일 버퍼 비우기 (실시간 반영)
                            f_out.flush()

                            # 상태 업데이트
                            count = len(reviews)
                            progress['total_collected'] += count
                            progress['current_game_collected'] += count
                            progress['cursor'] = data['cursor']
                            progress['genre_index'] = r_idx # 장르 인덱스 저장
                            progress['game_index'] = g_idx
                            progress['lang_index'] = l_idx
                            
                            save_progress(progress)
                            
                            current_sleep = max(BASE_SLEEP, current_sleep - SUCCESS_REDUCTION)
                            print(f"  - 진행: {progress['total_collected']} (슬립: {current_sleep:.1f}s)", end='\r')
                            time.sleep(current_sleep)

                        except Exception as e:
                            print(f"\n오류: {e}. 10초 대기...")
                            time.sleep(10)

    print("\n모든 장르 수집 작업 완료.")

if __name__ == "__main__":
    collect_reviews()