import requests
import time
import json
import os

# --- 설정 (Configuration) ---
TARGET_TOTAL_REVIEWS = 500000
REVIEWS_PER_GAME_LIMIT = 5000
LANGUAGES = ['koreana'] 

# 수집하고 싶은 장르 리스트 (SteamSpy 공식 명칭 준수)
TARGET_GENRES = ['Simulation']

# 경로 설정
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_PATH, "steam_data")
PROGRESS_FILE = os.path.join(SAVE_DIR, "progress.json")
DATA_FILE = os.path.join(SAVE_DIR, "all_reviews.jsonl")

# 가변 슬립 및 재시도 설정
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

def get_games_by_genre(genre_name):
    """[복원] SteamSpy API를 통해 특정 장르의 게임 리스트를 가져옵니다."""
    print(f"\n>>> '{genre_name}' 장르 게임 리스트 로드 중...")
    url = f"https://steamspy.com/api.php?request=genre&genre={genre_name}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if not data:
                print(f"!!! '{genre_name}' 장르 데이터를 찾을 수 없습니다. (API 불안정 가능성)")
                return []
            # 리뷰 합계순(인기순) 정렬
            return sorted(data.values(), 
                        key=lambda x: int(x.get('positive', 0)) + int(x.get('negative', 0)), 
                        reverse=True)
        else:
            print(f"SteamSpy 오류: {response.status_code}")
    except Exception as e:
        print(f"리스트 로드 중 오류 발생: {e}")
    return []

def get_popular_games_combined(pages=5):
    """[대안] 장르 API 장애 시 사용할 수 있는 전체 인기 게임 통합 함수입니다."""
    combined_games = []
    for p in range(pages):
        url = f"https://steamspy.com/api.php?request=all&page={p}"
        print(f">>> SteamSpy {p}페이지 데이터 통합 중...")
        try:
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                combined_games.extend(res.json().values())
        except: pass
        time.sleep(2) 
    return sorted(combined_games, key=lambda x: int(x.get('positive', 0)) + int(x.get('negative', 0)), reverse=True)

def collect_reviews():
    progress = load_progress()
    current_sleep = BASE_SLEEP
    
    print(f"재개 지점: 장르[{TARGET_GENRES[progress['genre_index']]}] (Idx {progress['genre_index']}), "
        f"게임 Idx {progress['game_index']}")

    with open(DATA_FILE, "a", encoding="utf-8") as f_out:
        for r_idx in range(progress['genre_index'], len(TARGET_GENRES)):
            current_genre = TARGET_GENRES[r_idx]
            
            # 1. 장르별 게임 리스트 로드 (맛이 갔을 경우 combined 함수로 교체 가능)
            genre_games = get_games_by_genre(current_genre)
            
            if not genre_games:
                print(f"장르 '{current_genre}'에 게임이 없거나 API 응답이 없습니다.")
                continue

            start_g_idx = progress['game_index'] if r_idx == progress['genre_index'] else 0
            
            for g_idx in range(start_g_idx, len(genre_games)):
                game = genre_games[g_idx]
                appid = game['appid']
                game_name = game['name']
                
                seen_reviews = set()
                last_cursor = None

                start_l_idx = progress['lang_index'] if (r_idx == progress['genre_index'] and g_idx == progress['game_index']) else 0
                
                for l_idx in range(start_l_idx, len(LANGUAGES)):
                    lang = LANGUAGES[l_idx]
                    
                    if r_idx != progress['genre_index'] or g_idx != progress['game_index'] or l_idx != progress['lang_index']:
                        progress['cursor'] = "*"
                        progress['current_game_collected'] = 0
                        last_cursor = None
                    
                    print(f"\n>>> [{current_genre}] {game_name} ({lang}) 수집 시작...")

                    while progress['current_game_collected'] < REVIEWS_PER_GAME_LIMIT:
                        # 커서 정체 확인
                        if last_cursor == progress['cursor'] and progress['cursor'] != "*":
                            print(f"\n[{game_name}] 커서 정체로 종료.")
                            break
                        last_cursor = progress['cursor']

                        if progress['total_collected'] >= TARGET_TOTAL_REVIEWS:
                            print("\n전체 목표 수치 달성!")
                            return

                        url = f"https://store.steampowered.com/appreviews/{appid}"
                        # filter='recent'를 적용하여 대량 리뷰 게임의 커서 꼬임 방지
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
                                print(f"\n[제한] 429. {current_sleep:.1f}s 대기.")
                                time.sleep(current_sleep)
                                current_sleep = min(MAX_SLEEP, current_sleep * BACKOFF_FACTOR)
                                continue
                            
                            if resp.status_code != 200:
                                time.sleep(10)
                                continue

                            data = resp.json()
                            reviews = data.get('reviews', [])
                            
                            if not reviews: 
                                print(f"\n[{game_name}] 데이터 소진.")
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

                            if new_count == 0 and progress['current_game_collected'] > 0:
                                print(f"\n[{game_name}] 새로운 유효 리뷰 없음.")
                                break

                            f_out.flush()

                            # 상태 업데이트 및 저장
                            progress['total_collected'] += new_count
                            progress['current_game_collected'] += new_count
                            progress['cursor'] = data.get('cursor', '*')
                            progress['genre_index'] = r_idx
                            progress['game_index'] = g_idx
                            progress['lang_index'] = l_idx
                            save_progress(progress)
                            
                            current_sleep = max(BASE_SLEEP, current_sleep - SUCCESS_REDUCTION)
                            print(f"  - 수집: {progress['total_collected']} (게임내: {progress['current_game_collected']}, 슬립: {current_sleep:.1f}s)", end='\r')
                            time.sleep(current_sleep)

                        except Exception as e:
                            print(f"\n오류: {e}. 10초 대기...")
                            time.sleep(10)

    print("\n모든 작업 완료.")

if __name__ == "__main__":
    collect_reviews()