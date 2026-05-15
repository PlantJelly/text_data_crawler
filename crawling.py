import requests
import time
import json
import os
# urllib.parse는 requests가 자동으로 처리하므로 수동 인코딩은 제거합니다.

# --- 설정 (Configuration) ---
TARGET_TOTAL_REVIEWS = 500000
REVIEWS_PER_GAME_LIMIT = 5000
LANGUAGES = ['koreana'] 

# 수집하고 싶은 장르 리스트 (현재는 임시로 통합 인기순 사용 중)
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
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "genre_index": 0,
        "game_index": 0,
        "lang_index": 0,
        "cursor": "*",
        "total_collected": 0,
        "current_game_collected": 0
    }

def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

def get_popular_games_combined(pages=5):
    """여러 페이지의 데이터를 통합하여 인기순 리스트를 만듭니다."""
    combined_games = []
    for p in range(pages):
        url = f"https://steamspy.com/api.php?request=all&page={p}"
        print(f">>> SteamSpy {p}페이지 데이터를 가져오는 중...")
        try:
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                combined_games.extend(res.json().values())
            else:
                print(f"  - {p}페이지 로드 실패 ({res.status_code})")
        except Exception as e:
            print(f"  - 에러 발생: {e}")
        time.sleep(2) 

    print(f">>> 총 {len(combined_games)}개 게임 통합 정렬 중...")
    return sorted(combined_games, 
                key=lambda x: int(x.get('positive', 0)) + int(x.get('negative', 0)), 
                reverse=True)

def collect_reviews():
    progress = load_progress()
    current_sleep = BASE_SLEEP
    
    # 시작 시 게임 리스트 확보
    genre_games = get_popular_games_combined()
    
    print(f"재개 지점: 장르[{TARGET_GENRES[progress['genre_index']]}] (인덱스 {progress['genre_index']}), "
        f"게임 인덱스 {progress['game_index']}")

    with open(DATA_FILE, "a", encoding="utf-8") as f_out:
        for r_idx in range(progress['genre_index'], len(TARGET_GENRES)):
            current_genre = TARGET_GENRES[r_idx]
            
            if not genre_games:
                continue

            start_g_idx = progress['game_index'] if r_idx == progress['genre_index'] else 0
            
            for g_idx in range(start_g_idx, len(genre_games)):
                game = genre_games[g_idx]
                appid = game['appid']
                game_name = game['name']
                
                # 중복 방지를 위한 셋(Set)
                seen_reviews = set()
                # 커서 정체 확인을 위한 변수
                last_cursor = None

                start_l_idx = progress['lang_index'] if (r_idx == progress['genre_index'] and g_idx == progress['game_index']) else 0
                
                for l_idx in range(start_l_idx, len(LANGUAGES)):
                    lang = LANGUAGES[l_idx]
                    
                    # 새 게임/언어 시작 시 초기화
                    if r_idx != progress['genre_index'] or g_idx != progress['game_index'] or l_idx != progress['lang_index']:
                        progress['cursor'] = "*"
                        progress['current_game_collected'] = 0
                        last_cursor = None
                    
                    print(f"\n>>> [{current_genre}] {game_name} ({lang}) 수집 시작...")

                    while progress['current_game_collected'] < REVIEWS_PER_GAME_LIMIT:
                        # 1. 커서 변화 체크 (무한 루프 방지)
                        if last_cursor == progress['cursor'] and progress['cursor'] != "*":
                            print(f"\n[{game_name}] 커서가 갱신되지 않아 다음 게임으로 넘어갑니다.")
                            break
                        
                        last_cursor = progress['cursor']

                        if progress['total_collected'] >= TARGET_TOTAL_REVIEWS:
                            print("\n전체 목표 수치 달성!")
                            return

                        url = f"https://store.steampowered.com/appreviews/{appid}"
                        # requests가 자동으로 인코딩하도록 원본 커서를 전달합니다.
                        params = {
                            'json': 1, 
                            'filter': 'recent', 
                            'language': lang,
                            'cursor': progress['cursor'], 
                            'num_per_page': 100, 
                            'purchase_type': 'all'
                        }

                        try:
                            resp = requests.get(url, params=params, timeout=15)
                            
                            if resp.status_code == 429:
                                print(f"\n[제한] 429 에러. {current_sleep}초 대기.")
                                time.sleep(current_sleep)
                                current_sleep = min(MAX_SLEEP, current_sleep * BACKOFF_FACTOR)
                                continue
                            
                            if resp.status_code != 200:
                                time.sleep(10)
                                continue

                            data = resp.json()
                            reviews = data.get('reviews', [])
                            
                            # 리뷰가 없으면 종료
                            if not reviews: 
                                print(f"\n[{game_name}] 더 이상의 리뷰가 없습니다.")
                                break

                            new_count = 0
                            for r in reviews:
                                rid = r['recommendationid']
                                if rid not in seen_reviews:
                                    clean = {
                                        'appid': appid, 'game': game_name, 'genre': current_genre,
                                        'lang': lang, 'review': r['review'], 'voted_up': r['voted_up']
                                    }
                                    f_out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                                    seen_reviews.add(rid)
                                    new_count += 1

                            # 데이터가 모두 중복이면 더 이상 페이지가 안 넘어가는 것으로 간주
                            if new_count == 0 and progress['current_game_collected'] > 0:
                                print(f"\n[{game_name}] 새로운 리뷰를 찾을 수 없어 종료합니다.")
                                break

                            f_out.flush()

                            # 2. 커서 업데이트 및 상태 저장
                            progress['total_collected'] += new_count
                            progress['current_game_collected'] += new_count
                            progress['cursor'] = data.get('cursor', '*')
                            progress['genre_index'] = r_idx
                            progress['game_index'] = g_idx
                            progress['lang_index'] = l_idx
                            
                            save_progress(progress)
                            
                            current_sleep = max(BASE_SLEEP, current_sleep - SUCCESS_REDUCTION)
                            print(f"  - 진행: {progress['total_collected']} (누적: {progress['current_game_collected']}, 슬립: {current_sleep:.1f}s)", end='\r')
                            time.sleep(current_sleep)

                        except Exception as e:
                            print(f"\n오류 발생: {e}. 10초 대기...")
                            time.sleep(10)

    print("\n모든 작업 완료.")

if __name__ == "__main__":
    collect_reviews()