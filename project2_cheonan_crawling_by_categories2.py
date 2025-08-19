# 천안시 MT1(대형마트), SC4(학교) 중 '대학', CT1(문화시설), PO3(공공기관)
# AT4(관광명소), HP8 중 '종합병원' 또는 '대학병원' 또는 '대형병원', '요양병원', '재활병원' 만 필터링
# 공영/민영 주차장 정보 지도에 함께 표시
# 천안시 경계면 정보 가져오기
# 천안시 교통량 수집기 위치 함께 표시
# 배경 위성사진으로 변경, 밝은 테마, 라벨 오버레이
# html 따로 저장
# 위성사진 VWorld에서 끌어옴 (VWORLD API KEY 사용)
# 천안시 경계 선으로 표시
# 서북구, 동남구 나눠서 경계 따로 표시
# 컬러 잘 보이도록 변환
# 불법 주정차 단속건수 히트맵으로 표시
# 격자로 쪼개서 스코어 비교
# 총 점 100점 만점에 5점 이상인 격자들 소격자로 쪼개서 다시 비교

# -*- coding: utf-8 -*-
"""
Cheonan (Category-based) POI + Parking + Traffic Sensors + Boundary Map (VWorld Satellite)
- 카카오 '카테고리' 기반 수집 + 후처리(대학/대형병원 필터)
- 동남구/서북구 경계를 개별 레이어로 on/off 가능
- 팝업 줄겹침 개선, 범례 추가, VWorld 위성+라벨
- 최초 1회 생성 후: REBUILD_MAP=False로 두면 기존 HTML만 즉시 열기
"""

import os
import time
import json
import re
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import folium
from folium import Element
from folium.plugins import MiniMap, MarkerCluster
import webbrowser
from datetime import datetime
from html import escape

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from folium.plugins import MiniMap, MarkerCluster, HeatMap


# =========================
# 설정
# =========================
REBUILD_MAP = True
SAVE_DIR = "./project2_cheonan_data"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_HTML = os.path.join(SAVE_DIR, "cheonan_map.html")

# API Keys
KAKAO_REST_KEY = (os.getenv("KAKAO_REST_KEY") or "").strip()
VWORLD_KEY     = (os.getenv("VWORLD_KEY") or "").strip()
if REBUILD_MAP and not KAKAO_REST_KEY:
    raise RuntimeError("REBUILD_MAP=True인데 환경변수 KAKAO_REST_KEY가 비어있습니다.")
if REBUILD_MAP and not VWORLD_KEY:
    raise RuntimeError("REBUILD_MAP=True인데 환경변수 VWORLD_KEY가 비어있습니다.")

HEADERS = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"} if KAKAO_REST_KEY else {}
KAKAO_CAT_URL  = "https://dapi.kakao.com/v2/local/search/category.json"
KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

PAGE_SIZE = 15
MAX_PAGES = 45
MAX_FETCHABLE = PAGE_SIZE * MAX_PAGES  # 675
SLEEP_SEC = 0.25
GEOCODE_SLEEP_SEC = 0.2

# 안정적 세션
SESSION = requests.Session()
_retry = Retry(total=5, connect=5, read=5, backoff_factor=0.6,
               status_forcelist=[429,500,502,503,504], allowed_methods=["GET"], raise_on_status=False)
_adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)
TIMEOUT_TUPLE = (8, 25)

# 경로
SHP_PATH = "./project2_cheonan_data/N3A_G0100000/N3A_G0100000.shp"
PUBLIC_PARKING_CSV = "./project2_cheonan_data/천안도시공사_주차장 현황_20250716.csv"
PRIVATE_PARKING_XLSX = "./project2_cheonan_data/충청남도_천안시_민영주차장정보.xlsx"
SENSORS_CSV = "./project2_cheonan_data/천안_교차로_행정동_정확매핑.csv"
TRAFFIC_STATS_CSV = "./project2_cheonan_data/스마트교차로_통계.csv"
ENFORCEMENT_CSV_23 = "./project2_cheonan_data/천안시_단속장소_위도경도_23년.csv"
ENFORCEMENT_CSV_24 = "./project2_cheonan_data/천안시_단속장소_위도경도_24년.csv"

# 지도 중심
MAP_CENTER_LAT, MAP_CENTER_LON = 36.815, 127.147
MAP_ZOOM = 12

# 경계선 스타일 (업데이트: 색 대비 강화)
BOUNDARY_COLOR = "#00E5FF"       # 천안시 전체 경계(시안)
BOUNDARY_HALO_COLOR = "#FFFFFF"  # 하이라이트(흰색)
BOUNDARY_HALO_WEIGHT = 7
BOUNDARY_LINE_WEIGHT = 3.5

DN_COLOR = "#BA2FE5"   # 동남구: 밝은 보라 (가시성 ↑)
SB_COLOR = "#FF5722"   # 서북구: 강한 주황 (전체 경계와 확실한 대비)

# 지오코딩 캐시
GEOCODE_CACHE_PATH = os.path.join(SAVE_DIR, "geocode_cache.json")
_geocode_cache = {}
if os.path.exists(GEOCODE_CACHE_PATH):
    try:
        with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
            _geocode_cache = json.load(f)
    except Exception:
        _geocode_cache = {}

# =========================
# 빠른 종료
# =========================
if not REBUILD_MAP and os.path.exists(SAVE_HTML):
    print(f"[INFO] Opening existing map: {SAVE_HTML}")
    webbrowser.open('file://' + os.path.realpath(SAVE_HTML))
    raise SystemExit(0)
elif not REBUILD_MAP and not os.path.exists(SAVE_HTML):
    print(f"[WARN] {SAVE_HTML} 이(가) 없어 새로 생성합니다.")
    REBUILD_MAP = True

# =========================
# 천안 경계 로더 (구별 GDF 포함)
# =========================
def load_cheonan_boundary_shp(shp_path: str):
    """
    천안시 전체 geometry + 동남구/서북구 개별 GDF 반환
    반환:
      - cheonan_gdf: 천안 관련 폴리곤들(GeoDataFrame, WGS84)
      - cheonan_geom: 천안 전체 단일 geometry
      - gu_gdf_map: {"동남구": GDF, "서북구": GDF}
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    target_sig5 = {"44130", "44131", "44133"}
    sel = gdf.iloc[0:0].copy()

    # 1) BJCD → SIG5
    if "BJCD" in gdf.columns:
        bj = gdf["BJCD"].astype(str).str.replace(r"\.0$","",regex=True).str.strip()
        sig5 = bj.str.slice(0,5)
        hit = gdf[sig5.isin(target_sig5)]
        if len(hit): sel = hit.copy()

    # 2) NAME 보조
    if len(sel)==0 and "NAME" in gdf.columns:
        nm = gdf["NAME"].astype(str)
        hit2 = gdf[nm.str.contains("천안|동남구|서북구|Cheonan", na=False)]
        if len(hit2): sel = hit2.copy()

    if len(sel)==0:
        raise ValueError("SHP에서 천안시(44130/44131/44133) 또는 이름('천안')이 매칭되지 않았습니다.")

    # 전체 geometry
    try:
        sel = sel.assign(SIG5=sel["BJCD"].astype(str).str.replace(r"\.0$","",regex=True).str.slice(0,5))
        sel = sel[sel["SIG5"].isin(target_sig5)]
        cheonan_geom = sel.dissolve(by="SIG5").unary_union
    except Exception:
        cheonan_geom = sel.unary_union

    # 구별 GDF
    gu_gdf_map = {"동남구": gpd.GeoDataFrame(sel.iloc[0:0].copy()),
                  "서북구": gpd.GeoDataFrame(sel.iloc[0:0].copy())}

    if "SIG5" in sel.columns:
        dn = sel[sel["SIG5"] == "44131"]
        sb = sel[sel["SIG5"] == "44133"]
        if len(dn): gu_gdf_map["동남구"] = dn.copy()
        if len(sb): gu_gdf_map["서북구"] = sb.copy()

    if "NAME" in sel.columns:
        if len(gu_gdf_map["동남구"]) == 0:
            dn2 = sel[sel["NAME"].astype(str).str.contains("동남구", na=False)]
            if len(dn2): gu_gdf_map["동남구"] = dn2.copy()
        if len(gu_gdf_map["서북구"]) == 0:
            sb2 = sel[sel["NAME"].astype(str).str.contains("서북구", na=False)]
            if len(sb2): gu_gdf_map["서북구"] = sb2.copy()

    return sel.copy(), cheonan_geom, gu_gdf_map

# =========================
# Kakao API (Category)
# =========================
def _kakao_get(url, params, headers, max_retries=3):
    last = None
    for attempt in range(1, max_retries+1):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=TIMEOUT_TUPLE)
        except requests.exceptions.RequestException as e:
            time.sleep(SLEEP_SEC * attempt); last = e; continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429,500,502,503,504):
            time.sleep(SLEEP_SEC * attempt); last = resp; continue
        raise requests.HTTPError(f"{resp.status_code} {resp.reason} | url={resp.url}\nbody={resp.text}", response=resp)
    if isinstance(last, requests.Response):
        raise requests.HTTPError(f"Request failed after retries. last_status={last.status_code}, body={last.text}", response=last)
    elif last is not None:
        raise requests.HTTPError(f"Request failed after retries due to network error: {last}")
    else:
        raise requests.HTTPError("Request failed with unknown error")

def search_category_rect(group_code, minX, minY, maxX, maxY, *, headers=HEADERS, page_size=PAGE_SIZE):
    # 정렬
    minX, maxX = (minX, maxX) if minX <= maxX else (maxX, minX)
    minY, maxY = (minY, maxY) if minY <= maxY else (maxY, minY)

    page_num = 1
    base_params = {
        "category_group_code": group_code,
        "page": page_num,
        "size": page_size,
        "rect": f"{minX},{minY},{maxX},{maxY}"
    }
    resp = _kakao_get(KAKAO_CAT_URL, base_params, headers)
    payload = resp.json()
    total_count = payload.get("meta", {}).get("total_count", 0)

    if total_count > MAX_FETCHABLE:
        docs = []
        midX = (minX + maxX)/2.0
        midY = (minY + maxY)/2.0
        docs.extend(search_category_rect(group_code, minX, minY, midX,  midY, headers=headers, page_size=page_size))
        docs.extend(search_category_rect(group_code, midX,  minY, maxX,  midY, headers=headers, page_size=page_size))
        docs.extend(search_category_rect(group_code, minX, midY,  midX,  maxY, headers=headers, page_size=page_size))
        docs.extend(search_category_rect(group_code, midX,  midY,  maxX,  maxY, headers=headers, page_size=page_size))
        return docs

    documents = []
    while True:
        cur = payload if page_num == 1 else _kakao_get(KAKAO_CAT_URL, {**base_params, "page": page_num}, headers).json()
        docs = cur.get("documents", [])
        documents.extend(docs)
        if cur.get("meta", {}).get("is_end", True) or page_num >= MAX_PAGES:
            break
        page_num += 1
        time.sleep(SLEEP_SEC)
    return documents

def overlapped_category_in_polygon(group_code, bbox, num_x, num_y, poly, *, headers=HEADERS, page_size=PAGE_SIZE):
    minX, minY, maxX, maxY = bbox
    step_x = (maxX - minX)/float(num_x)
    step_y = (maxY - minY)/float(num_y)

    results = []
    for i in range(num_x):
        for j in range(num_y):
            cell_minX = minX + i*step_x
            cell_maxX = cell_minX + step_x
            cell_minY = minY + j*step_y
            cell_maxY = cell_minY + step_y
            cell_geom = box(cell_minX, cell_minY, cell_maxX, cell_maxY)
            if not poly.intersects(cell_geom):
                continue
            docs = search_category_rect(group_code, cell_minX, cell_minY, cell_maxX, cell_maxY,
                                        headers=headers, page_size=page_size)
            results.extend(docs)
            time.sleep(SLEEP_SEC)
    return results

def search_keyword_rect(keyword, minX, minY, maxX, maxY, *, headers=HEADERS, page_size=PAGE_SIZE):
    # 정렬
    minX, maxX = (minX, maxX) if minX <= maxX else (maxX, minX)
    minY, maxY = (minY, maxY) if minY <= maxY else (maxY, minY)

    page_num = 1
    base_params = {
        "query": str(keyword),
        "page": page_num,
        "size": page_size,
        "rect": f"{minX},{minY},{maxX},{maxY}"
    }
    resp = _kakao_get(KAKAO_KEYWORD_URL, base_params, headers)
    payload = resp.json()
    documents = []
    while True:
        cur = payload if page_num == 1 else _kakao_get(KAKAO_KEYWORD_URL, {**base_params, "page": page_num}, headers).json()
        docs = cur.get("documents", [])
        documents.extend(docs)
        if cur.get("meta", {}).get("is_end", True) or page_num >= MAX_PAGES:
            break
        page_num += 1
        time.sleep(SLEEP_SEC)
    return documents

def overlapped_keyword_in_polygon(keyword, bbox, num_x, num_y, poly, *, headers=HEADERS, page_size=PAGE_SIZE):
    minX, minY, maxX, maxY = bbox
    step_x = (maxX - minX)/float(num_x)
    step_y = (maxY - minY)/float(num_y)
    results = []
    for i in range(num_x):
        for j in range(num_y):
            cell_minX = minX + i*step_x
            cell_maxX = cell_minX + step_x
            cell_minY = minY + j*step_y
            cell_maxY = cell_minY + step_y
            cell_geom = box(cell_minX, cell_minY, cell_maxX, cell_maxY)
            if not poly.intersects(cell_geom):
                continue
            docs = search_keyword_rect(keyword, cell_minX, cell_minY, cell_maxX, cell_maxY,
                                       headers=headers, page_size=page_size)
            results.extend(docs)
            time.sleep(SLEEP_SEC)
    return results


# =========================
# 카테고리 & 필터 규칙
# =========================
TARGET_GROUPS = ["MT1", "SC4", "CT1", "PO3", "HP8"]

CATEGORY_LEGEND = {
    "MT1": ("대형마트/백화점", "대형마트·백화점"),
    "SC4": ("학교(대학)", "대학교·대학원 등"),
    "CT1": ("문화시설", "도서관·공연장·미술관·박물관 등"),
    "PO3": ("공공기관", "시청·구청·주민센터 등"),
    "HP8": ("병원", "종합·대학·대형·요양·재활 병원"),
}

ICON_BY_GROUP = {
    "MT1": ("shopping-cart", "fa", "red"),
    "SC4": ("university", "fa", "blue"),
    "CT1": ("book", "fa", "orange"),
    "PO3": ("institution", "fa", "green"),
    "HP8": ("hospital-o", "fa", "darkred"),
}
DEFAULT_ICON = ("info-sign", "glyphicon", "cadetblue")

_re_univ = re.compile(r"(대학교|대학|University)", re.IGNORECASE)
_re_hospital_big = re.compile(r"(종합병원|대학병원|대형병원|요양병원|재활병원)", re.IGNORECASE)
_re_waffle = re.compile(r"와플대학")

_re_po3_drop = re.compile(r"(ATM|민팃|무인|무인민원발급기|카페|커피|할리스|스타벅스|이디야|빽다방|파스쿠찌|투썸|엔젤리너스|폴바셋)", re.IGNORECASE)
_re_trim_paren = re.compile(r"[（(].*?[）)]")

def normalize_po3_core(name: str) -> str | None:
    """
    공공기관(PO3)명 정규화:
      - 부서/팀/센터 등 세부조직 제거
      - 핵심 기관명(행정복지센터/보건소/우체국 등)까지만 축약
      - 카페/ATM 등 잘못 분류된 항목은 None 반환(제외)
    """
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.strip()

    # 카페/ATM 등 명백한 오분류 제거
    if _re_po3_drop.search(n):
        return None

    # 괄호 안 부가설명 제거
    n = _re_trim_paren.sub("", n).strip()

    # 흔한 세부조직 접미어 정리
    # 예) "동남구보건소 정신건강복지센터팀" → "동남구보건소"
    #     "○○행정복지센터 민원팀" → "○○행정복지센터"
    trunk_keys = ["행정복지센터", "보건소", "우체국", "주민센터", "구청", "시청"]
    for key in trunk_keys:
        if key in n:
            # key 이전의 지명 + key 까지만 남김
            left = n.split(key)[0]
            n = f"{left}{key}".strip()
            break

    # 남아있는 '팀/센터/실/과/담당/창구/민원' 등 제거(맨 끝 위주)
    n = re.sub(r"(정신건강복지|치매안심|마음건강|의약|건강생활|민원|산모신생아|모자보건|예방접종)\s*(센터|팀)$", "", n).strip()
    n = re.sub(r"(센터|팀|과|실|담당)$", "", n).strip()

    # 너무 짧아지면 원본 유지
    return n if len(n) >= 2 else name.strip()


def dedup_po3_public_institutions(df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    PO3(공공기관)만 정규화명 기준으로 중복 제거.
    - normalize_po3_core(name) == None 인 행은 제거
    - 동일 정규화명은 대표 1개만 남김(이름 짧은 것 우선)
    """
    if df_cat is None or not len(df_cat):
        return df_cat

    df = df_cat.copy()
    po3 = df[df["group_code"] == "PO3"].copy()
    if not len(po3):
        return df

    po3["po3_core"] = po3["name"].map(normalize_po3_core)
    po3 = po3.dropna(subset=["po3_core"])

    # 대표 선택 규칙: (1) 이름 길이, (2) 도로명 주소 유무
    po3["_name_len"] = po3["name"].astype(str).str.len()
    po3["_addr_ok"]  = po3["road_address"].astype(str).str.len().gt(0).astype(int)

    rep = (
        po3.sort_values(by=["po3_core", "_name_len", "_addr_ok"], ascending=[True, True, False])
           .groupby("po3_core", as_index=False, sort=False)
           .first()
    )

    others = df[df["group_code"] != "PO3"]
    keep_cols = df.columns.tolist()
    rep = rep[keep_cols]

    return pd.concat([others, rep], ignore_index=True)

def category_passes_filter(group_code: str, doc: dict) -> bool:
    name = str(doc.get("place_name", "") or "")
    catname = str(doc.get("category_name", "") or "")

    if group_code == "SC4":
        if _re_waffle.search(name):
            return False
        return bool(_re_univ.search(name) or _re_univ.search(catname))

    if group_code == "HP8":
        return bool(_re_hospital_big.search(name) or _re_hospital_big.search(catname))

    return True

# =========================
# 캠퍼스 중복 제거 함수
# =========================
def dedup_campus_pois(df_cat: pd.DataFrame,
                      categories=("CT1",),          # 문화시설만 기본 대상
                      merge_radius_m=350,           # 반경 이내 묶기 (독립기념관은 300~450m 권장)
                      min_cluster_size=3,           # 최소 몇 개 이상 모이면 '캠퍼스'로 간주
                      name_guard=True):             # 이름 가드(같은 핵심어 공유) 사용할지
    """
    입력 df_cat: columns 포함 [name, lat, lon, group_code, ...]
    반환: 원본 df_cat에서 '캠퍼스 병합된 대표 포인트'만 남긴 DataFrame
          (타 카테고리는 원본 그대로 유지, 대상 카테고리만 축약)
    """
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    from pyproj import Transformer

    if df_cat is None or len(df_cat) == 0:
        return df_cat

    # 분리: 대상 카테고리 vs 비대상
    df_target = df_cat[df_cat["group_code"].isin(categories)].copy()
    df_others = df_cat[~df_cat["group_code"].isin(categories)].copy()

    if len(df_target) == 0:
        return df_cat

    # 좌표 → 미터 좌표계로 변환 (국가좌표 EPSG:5186: Korea Central Belt 2010)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    x, y = tf.transform(df_target["lon"].values.astype(float),
                        df_target["lat"].values.astype(float))
    df_target["_x"] = x
    df_target["_y"] = y

    # 이름 핵심어 추출(간단): '독립기념관', '박물관', '센터' 등 접미어 이전 핵심 토큰만
    def core_token(s: str):
        if not isinstance(s, str): return ""
        s = re.sub(r"\s+", " ", s.strip())
        # 제N관/본관/전시관/체험관/별관 등 제거
        s = re.sub(r"(제\s*\d+\s*관|본관|별관|전시관|체험관|관|홀)$", "", s)
        # 너무 짧으면 공백
        return s.strip()

    df_target["_core"] = df_target["name"].map(core_token)

    # 간단 greedy 클러스터링 (O(n^2) 가능성 있지만 보통 n이 크지 않음)
    used = np.zeros(len(df_target), dtype=bool)
    idxs = np.arange(len(df_target))
    clusters = []

    for i in idxs:
        if used[i]: 
            continue
        # 시드
        cx, cy = df_target["_x"].iat[i], df_target["_y"].iat[i]
        cname = df_target["_core"].iat[i]

        # 반경 내 포인트 뽑기
        dx = df_target["_x"].values - cx
        dy = df_target["_y"].values - cy
        dist = np.hypot(dx, dy)
        cand = (dist <= merge_radius_m) & (~used)

        if cand.sum() == 1:
            # 혼자이면 그냥 클러스터 1개
            clusters.append([i])
            used[i] = True
            continue

        # 이름 가드: 같은 핵심어가 하나라도 공유되면 우선 묶기
        member_idx = np.where(cand)[0].tolist()
        if name_guard:
            cores = df_target["_core"].iloc[member_idx].values
            # 가장 많이 등장하는 핵심어 찾기
            vals, cnts = np.unique(cores[cores != ""], return_counts=True)
            if len(vals):
                top = vals[np.argmax(cnts)]
                # 그 핵심어를 포함하는 애들만 남김 (너무 다른 시설 오합지졸 방지)
                member_idx = [j for j in member_idx if (df_target["_core"].iat[j] == top or top in df_target["name"].iat[j])]
        
        # 최소 군집 크기 조건
        if len(member_idx) >= min_cluster_size:
            clusters.append(member_idx)
            used[member_idx] = True
        else:
            # 군집 조건 못 채우면 각자 독립 포인트 처리
            clusters.append([i])
            used[i] = True

    # 각 클러스터에서 '대표' 1개만 남김 (근사: 첫 항목)
    keep_idx = []
    for c in clusters:
        if len(c) == 1:
            keep_idx.extend(c)
        else:
            # 대표 선택 규칙: 이름 길이가 짧은 것(상위 개념일 가능성↑) 우선
            c_sorted = sorted(c, key=lambda j: len(str(df_target["name"].iat[j])))
            keep_idx.append(c_sorted[0])

    df_target_dedup = df_target.iloc[sorted(set(keep_idx))].copy()

    # 정리 컬럼 제거
    df_target_dedup = df_target_dedup.drop(columns=["_x","_y","_core"], errors="ignore")
    return pd.concat([df_others, df_target_dedup], ignore_index=True)


# =========================
# Geocoding & Data loaders (주차장/수집기)
# =========================
def _kakao_get_addr(query: str):
    if not query or not str(query).strip():
        return (np.nan, np.nan)
    key = str(query).strip()
    if key in _geocode_cache:
        lon, lat = _geocode_cache[key]
        return (float(lon), float(lat)) if (lon is not None and lat is not None) else (np.nan, np.nan)

    params = {"query": key}
    last_exc = None
    for attempt in range(1, 3+1):
        try:
            resp = _kakao_get(KAKAO_ADDR_URL, params, HEADERS)
            docs = resp.json().get("documents", [])
            if not docs:
                _geocode_cache[key] = (None, None)
                break
            x = docs[0].get("x"); y = docs[0].get("y")
            lon = float(x); lat = float(y)
            _geocode_cache[key] = (lon, lat)
            try:
                with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(_geocode_cache, f, ensure_ascii=False)
            except Exception:
                pass
            time.sleep(GEOCODE_SLEEP_SEC)
            return (lon, lat)
        except (requests.exceptions.RequestException, requests.HTTPError) as e:
            last_exc = e
            time.sleep(SLEEP_SEC * attempt)
            continue
    if last_exc:
        print(f"[WARN] Geocode failed after retries: {key} | {last_exc}")
    return (np.nan, np.nan)

def load_public_parking(csv_path: str):
    try:
        dfp = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        dfp = pd.read_csv(csv_path, encoding="cp949")
    rename = {}
    if "주차장명" in dfp.columns: rename["주차장명"]="name"
    if "주소" in dfp.columns: rename["주소"]="address"
    if "위도" in dfp.columns: rename["위도"]="lat"
    if "경도" in dfp.columns: rename["경도"]="lon"
    dfp = dfp.rename(columns=rename)
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp["road_address"] = dfp["address"].fillna("")
    dfp["jibun_address"] = ""
    dfp["category"] = "공영주차장"
    dfp["source"] = "천안도시공사"
    dfp["id"] = "public_" + dfp.index.astype(str)
    return dfp[["id","name","lat","lon","road_address","jibun_address","category","source"]]

def load_private_parking(xlsx_path: str):
    dfm = pd.read_excel(xlsx_path)
    rename = {}
    if "주차장명" in dfm.columns: rename["주차장명"]="name"
    if "소재지도로명주소" in dfm.columns: rename["소재지도로명주소"]="road_address"
    if "소재지지번주소" in dfm.columns: rename["소재지지번주소"]="jibun_address"
    dfm = dfm.rename(columns=rename)
    lons, lats = [], []
    for _, row in dfm.iterrows():
        cand1 = str(row.get("road_address") or "").strip()
        cand2 = str(row.get("jibun_address") or "").strip()
        query = cand1 if cand1 else cand2
        lon, lat = _kakao_get_addr(query)
        lons.append(lon); lats.append(lat)
    dfm["lon"] = lons; dfm["lat"] = lats
    dfm["category"] = "민영주차장"
    dfm["source"] = "천안시/민영"
    dfm["id"] = "private_" + dfm.index.astype(str)
    return dfm[["id","name","lat","lon","road_address","jibun_address","category","source"]]

def load_enforcement_points(csv_path: str):
    """불법주정차 단속 포인트 로드: 위도/경도/단속건수 → lat/lon/count 로 표준화"""
    # 인코딩 유연 처리
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(csv_path)

    # 열 이름 표준화
    rename = {}
    if "위도" in df.columns: rename["위도"] = "lat"
    if "경도" in df.columns: rename["경도"] = "lon"
    if "단속건수" in df.columns: rename["단속건수"] = "count"
    df = df.rename(columns=rename)

    req = {"lat", "lon", "count"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"단속 CSV에 필요한 열이 부족합니다: {sorted(miss)}")

    # 숫자 변환 & 결측 제거
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "count"]).reset_index(drop=True)

    # 부정확 좌표 제거(대한민국 대략 범위)
    df = df[(df["lon"].between(124, 132)) & (df["lat"].between(33, 39))].copy()

    # 공통 스키마 정렬
    df["id"] = "enf_" + df.index.astype(str)
    df["name"] = ""  # 팝업 타이틀은 공란(필요시 지점명 추가 가능)
    df["road_address"] = ""
    df["jibun_address"] = ""
    df["category"] = "불법주정차 단속"
    df["source"] = "천안시(사용자 CSV)"
    return df[["id","name","lat","lon","count","road_address","jibun_address","category","source"]]

def load_traffic_sensors_exact(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("수집기 CSV에 lon/lat 열이 없습니다.")

    # 조인 키(원본명)는 반드시 살려둠: 통계의 '교차로명'과 1:1 매칭
    join_key_col = "원본명" if "원본명" in df.columns else None
    if join_key_col is None:
        # 최후방어: 이름 열 중 하나를 조인키로 사용 (정확도↓)
        join_key_col = "매칭_교차로명" if "매칭_교차로명" in df.columns else ("정규화명" if "정규화명" in df.columns else None)

    # 표시용 이름(팝업/툴팁)
    name_col = "매칭_교차로명" if "매칭_교차로명" in df.columns else ("정규화명" if "정규화명" in df.columns else ("원본명" if "원본명" in df.columns else None))
    addr_col = "주소(있으면)" if "주소(있으면)" in df.columns else None

    out = pd.DataFrame()
    out["name"] = df[name_col] if name_col else df.index.map(lambda i: f"수집기_{i}")
    out["join_key"] = df[join_key_col] if join_key_col else out["name"]   # ← 통계와 조인할 키
    out["road_address"] = df[addr_col] if addr_col else ""
    out["jibun_address"] = ""
    out["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    out["category"] = "교통량수집기"
    out["source"] = "천안시(스마트교통)"
    out["id"] = "sensor_" + out.index.astype(str)
    return out[["id","name","join_key","lat","lon","road_address","jibun_address","category","source"]]

# =========================
# 클러스터 (그라데이션)
# =========================
def make_cluster(thresholds=(5, 10)):
    t1, t2 = thresholds
    js = f"""
    function(cluster) {{
        var count = cluster.getChildCount();
        var grad = 'radial-gradient(circle at 30% 30%, #fff3b0 0%, #ffe066 55%, #ffc107 100%)';
        if (count >= {t2}) {{
            grad = 'radial-gradient(circle at 30% 30%, #ffb3b3 0%, #ff6b6b 55%, #e03131 100%)';
        }} else if (count >= {t1}) {{
            grad = 'radial-gradient(circle at 30% 30%, #ffd6a5 0%, #ff922b 55%, #f76707 100%)';
        }}
        var html = ''
            + '<div style="background:' + grad + ';border:1px solid rgba(0,0,0,0.25);border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 2px rgba(255,255,255,0.6) inset;">'
            + '<span style="color:black;font-weight:700;">' + count + '</span></div>';
        return new L.DivIcon({{ html: html, className: 'marker-cluster-custom', iconSize: new L.Point(40, 40) }});
    }}"""
    return MarkerCluster(icon_create_function=js)

def load_traffic_stats(csv_path: str):
    """스마트교차로_통계.csv → 7월(2025-07-01~31)만 필터, 교차로명별 일평균 계산"""
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    # 필수 컬럼 체크
    for col in ["일자", "교차로명", "합계"]:
        if col not in df.columns:
            raise ValueError(f"교통량 통계 CSV에 '{col}' 열이 없습니다.")

    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
    df["합계"] = pd.to_numeric(df["합계"], errors="coerce")

    # 2025-07-01 ~ 2025-07-31
    mask = (df["일자"] >= pd.Timestamp("2025-07-01")) & (df["일자"] <= pd.Timestamp("2025-07-31"))
    df_july = df.loc[mask].copy()

    # 교차로명별 7월 일평균
    grp = df_july.groupby("교차로명", as_index=False).agg(
        july_mean=("합계", "mean"),
        july_sum =("합계", "sum"),
        days     =("합계", "count"),
    )
    return grp  # columns: 교차로명, july_mean, july_sum, days


# =========================
# 팝업 HTML 빌더 (줄겹침 방지)
# =========================
def build_popup_html(title: str, rows: list, link: str = None, width: int = 330, height: int = 180) -> folium.Popup:
    safe_title = escape(str(title or ""))
    rows_html = ""
    for label, value in rows:
        if value is not None and str(value).strip():
            rows_html += f'<div><b>{escape(str(label))}</b> : {escape(str(value))}</div>'
    link_html = ""
    if link and str(link).strip():
        safe_link = escape(str(link), quote=True)
        link_html = f'<div style="margin-top:6px;"><a href="{safe_link}" target="_blank" rel="noopener">카카오 장소 페이지</a></div>'
    html = f"""
    <div style="font-size:14px; line-height:1.5; white-space: normal; word-break: keep-all;">
        <div style="font-weight:700; margin-bottom:6px;">{safe_title}</div>
        {rows_html}
        {link_html}
    </div>
    """
    iframe = folium.IFrame(html=html, width=width, height=height)
    return folium.Popup(iframe, max_width=width + 10)

# === build_popup_html 바로 아래에 추가 ===
def build_popup_html_str(title: str, rows: list, link: str = None, width: int = 330) -> str:
    """팝업용 HTML 문자열만 반환 (Folium 객체 아님)"""
    safe_title = escape(str(title or ""))
    rows_html = ""
    for label, value in rows:
        if value is not None and str(value).strip():
            rows_html += f'<div><b>{escape(str(label))}</b> : {escape(str(value))}</div>'
    link_html = ""
    if link and str(link).strip():
        safe_link = escape(str(link), quote=True)
        link_html = f'<div style="margin-top:6px;"><a href="{safe_link}" target="_blank" rel="noopener">카카오 장소 페이지</a></div>'
    html = f"""
    <div style="font-size:14px; line-height:1.5; white-space: normal; word-break: keep-all;">
        <div style="font-weight:700; margin-bottom:6px;">{safe_title}</div>
        {rows_html}
        {link_html}
    </div>
    """
    return html


# =========================
# 베이스맵(VWorld) + 라벨
# =========================
def add_vworld_base_layers(m):
    folium.TileLayer(
        tiles=f"https://api.vworld.kr/req/wmts/1.0.0/{VWORLD_KEY}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr="© VWorld / NGII",
        name="위성 (VWorld)",
        show=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attr="© CartoDB, OSM contributors",
        name="밝은 지도 (Carto Positron)",
        show=False
    ).add_to(m)
    folium.TileLayer(
        tiles=f"https://api.vworld.kr/req/wmts/1.0.0/{VWORLD_KEY}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr="© VWorld / NGII",
        name="라벨 (VWorld Hybrid)",
        overlay=True,
        control=True,
        opacity=0.9
    ).add_to(m)

# =========================
# 카테고리 POI 레이어 + 범례
# =========================
def add_category_layers(m, df_cat):
    if not len(df_cat):
        return m

    # 팝업 줄겹침 방지 CSS
    css = Element("""
    <style>
    /* 팝업 전체 컨테이너 폭 가이드 */
    .leaflet-popup-content-wrapper{
      min-width: 300px !important;
      max-width: 360px !important;
    }

    /* 실제 콘텐츠 영역 폭 가이드 */
    .leaflet-popup-content{
      white-space: normal !important;
      line-height: 1.5 !important;
      min-width: 280px !important;
      max-width: 340px !important;
      word-break: break-word;  /* 긴 단어/URL 줄바꿈 허용 */
    }

    /* 링크/행 간격은 기존 유지 */
    .leaflet-popup-content a { display: block; margin-top: 6px; }
    .leaflet-popup-content div { margin: 0 0 2px 0; }
    </style>
    """)

    m.get_root().html.add_child(css)

    groups = df_cat["group_code"].unique().tolist()
    layer_objs = {}
    for gc in groups:
        label = CATEGORY_LEGEND.get(gc, (gc,""))[0]
        fg = folium.FeatureGroup(name=f"[카테고리] {label}")
        cluster = make_cluster().add_to(fg)
        m.add_child(fg)
        layer_objs[gc] = {"fg": fg, "cluster": cluster}

    for _, row in df_cat.iterrows():
        name = str(row.get("name",""))
        road_addr = str(row.get("road_address",""))
        jibun_addr = str(row.get("jibun_address",""))
        url = str(row.get("url",""))
        catname = str(row.get("category_name",""))
        group_code = str(row.get("group_code",""))

        icon_name, icon_prefix, color = ICON_BY_GROUP.get(group_code, DEFAULT_ICON)
        tooltip = folium.Tooltip(f"{name}\n도로명: {road_addr}\n지번: {jibun_addr}", sticky=True)
        popup = build_popup_html(
            title=name,
            rows=[("카테고리", catname), ("도로명", road_addr), ("지번", jibun_addr)],
            link=url
        )
        folium.Marker(
            [float(row["lat"]), float(row["lon"])],
            tooltip=tooltip,
            popup=popup,
            icon=folium.Icon(color=color, icon=icon_name, prefix=icon_prefix),
        ).add_to(layer_objs[group_code]["cluster"])

    # 범례 (오른쪽 아래)
    legend_items = []
    for gc in TARGET_GROUPS:
        title, desc = CATEGORY_LEGEND.get(gc, (gc, ""))
        _, _, color = ICON_BY_GROUP.get(gc, DEFAULT_ICON)
        legend_items.append(f"""
            <div style="margin-bottom:6px;">
                <span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px;"></span>
                <b>{escape(title)}</b><br><span style="opacity:0.85;">{escape(desc)}</span>
            </div>
        """)

    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 12px; right: 12px; z-index: 9999; 
        background: rgba(255,255,255,0.92); 
        padding: 10px 12px; border: 1px solid rgba(0,0,0,0.2); 
        border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 13px; line-height: 1.3; max-width: 260px;">
        <div style="font-weight:700; margin-bottom:6px;">카테고리 안내</div>
        {''.join(legend_items)}
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))
    return m

# =========================
# 주차장/수집기 레이어
# =========================
def add_parking_layers_to_map(m, df_public, df_private):
    # 공영
    fg_pub = folium.FeatureGroup(name="[주차장] 공영")
    cluster_pub = make_cluster().add_to(fg_pub)
    for _, r in df_public.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]): continue
        name = str(r.get("name",""))
        road_addr = str(r.get("road_address","")); jibun_addr = str(r.get("jibun_address",""))
        tooltip = folium.Tooltip(f"{name}\n도로명: {road_addr}\n지번: {jibun_addr}", sticky=True)
        popup = build_popup_html(
            title=name,
            rows=[("유형","공영주차장"),("도로명",road_addr),("지번",jibun_addr),("출처",str(r.get("source","")))]
        )
        folium.Marker([float(r["lat"]), float(r["lon"])], tooltip=tooltip, popup=popup,
                      icon=folium.Icon(color="black", icon="car", prefix="fa")).add_to(cluster_pub)
    m.add_child(fg_pub)

    # 민영
    fg_pri = folium.FeatureGroup(name="[주차장] 민영")
    cluster_pri = make_cluster().add_to(fg_pri)
    for _, r in df_private.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]): continue
        name = str(r.get("name",""))
        road_addr = str(r.get("road_address","")); jibun_addr = str(r.get("jibun_address",""))
        tooltip = folium.Tooltip(f"{name}\n도로명: {road_addr}\n지번: {jibun_addr}", sticky=True)
        popup = build_popup_html(
            title=name,
            rows=[("유형","민영주차장"),("도로명",road_addr),("지번",jibun_addr),("출처",str(r.get("source","")))]
        )
        folium.Marker([float(r["lat"]), float(r["lon"])], tooltip=tooltip, popup=popup,
                      icon=folium.Icon(color="gray", icon="car", prefix="fa")).add_to(cluster_pri)
    m.add_child(fg_pri)
    return m

def add_enforcement_layer(m, df_enf: pd.DataFrame):
    """불법주정차 단속 포인트 레이어: 단속건수에 따라 원 크기/색상 표시 (+연도 표시 지원)"""
    try:
        import branca
        use_cmap = True
    except Exception:
        branca = None
        use_cmap = False

    fg = folium.FeatureGroup(name="[단속] 불법주정차 단속")

    # 값 범위
    vmin = float(df_enf["count"].min())
    vmax = float(df_enf["count"].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return m  # 유효 데이터 없음
    if vmax <= vmin:
        vmax = vmin + 1.0

    # 색상 맵(가능하면 OrRd, 아니면 고정색)
    if use_cmap:
        cmap = branca.colormap.linear.OrRd_09.scale(vmin, vmax)
        cmap.caption = "단속건수(건)"
        cmap.add_to(m)
    else:
        cmap = None

    for _, r in df_enf.iterrows():
        lat, lon, cnt = float(r["lat"]), float(r["lon"]), float(r["count"])

        # (추가) 연도 표기 준비
        year = r["year"] if "year" in r and pd.notna(r["year"]) else None
        year_txt = f"{int(year)}년 " if year is not None else ""

        # 반지름(4~12px 스케일)
        radius = 4.0 + 8.0 * ((cnt - vmin) / (vmax - vmin))
        radius = max(4.0, min(radius, 12.0))  # 안전 가드

        # 색상
        fill_color = cmap(cnt) if cmap else "#8B0000"

        # (수정) 연도 포함 툴팁
        tooltip = folium.Tooltip(
            f"{year_txt}단속건수: {int(cnt):,}건\n위도: {lat:.6f}\n경도: {lon:.6f}",
            sticky=True
        )

        # (수정) 연도 포함 팝업
        rows = [
            ("위도", f"{lat:.6f}"),
            ("경도", f"{lon:.6f}"),
            ("단속건수", f"{int(cnt):,}건"),
        ]
        if year is not None:
            rows.append(("연도", f"{int(year)}"))

        popup = build_popup_html(
            title=f"{year_txt}불법주정차 단속 지점",
            rows=rows,
            link=None
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            weight=1,
            color="#303030",
            fill=True,
            fill_opacity=0.85,
            fill_color=fill_color,
            tooltip=tooltip,
            popup=popup
        ).add_to(fg)

    m.add_child(fg)
    return m

def add_enforcement_heatmap_layer(m, df_enf: pd.DataFrame):
    """
    불법주정차 단속 히트맵 (+맞춤 그라데이션 & 범례):
    - 좌표 5자리 반올림으로 집계
    - 단속건수를 가중치로 사용
    - 파랑→초록→노랑→주황→빨강 그라데이션
    - 범례 캡션: 2023년 1월 ~ 2024년 8월까지 불법주정차 단속건수(건)
    """
    if df_enf is None or not len(df_enf):
        return m

    # 유효 값만
    df = df_enf.dropna(subset=["lat", "lon", "count"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "count"])
    if not len(df):
        return m

    # 좌표 그리드 집계(혼잡 축약) — 필요시 4(≈10m)~6(≈0.1m)로 조절
    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)
    agg = df.groupby(["lat_r", "lon_r"], as_index=False)["count"].sum()

    minc = float(agg["count"].min())
    maxc = float(agg["count"].max())
    if maxc <= 0:
        return m

    # 가중치(0~1)
    agg["weight"] = (agg["count"] - minc) / (maxc - minc) if maxc > minc else 1.0

    # ----- 맞춤 컬러맵 & 범례 -----
    import branca
    # 색상: 낮음(파랑) → 높음(빨강)
    colors = ["#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]  # Blue, Green, Yellow, Orange, Red
    colormap = branca.colormap.LinearColormap(colors=colors, vmin=minc, vmax=maxc)
    colormap.caption = "2023년 1월 ~ 2024년 8월까지 불법주정차 단속건수(건)"
    colormap.add_to(m)

    # HeatMap은 0~1 키를 사용하므로 같은 팔레트로 변환
    # (0, 0.25, 0.5, 0.75, 1.0) 구간에 동일 색상 지정
    stops01 = [0.00, 0.25, 0.50, 0.75, 1.00]
    gradient = {
        0.00: colors[0],
        0.25: colors[1],
        0.50: colors[2],
        0.75: colors[3],
        1.00: colors[4],
    }

    # HeatMap 데이터: [lat, lon, weight]
    heat_data = agg[["lat_r", "lon_r", "weight"]].values.tolist()

    fg = folium.FeatureGroup(name="[단속] 히트맵(불법주정차)")
    HeatMap(
        heat_data,
        radius=18,       # 혼잡하면 14~22 사이 조절
        blur=22,
        min_opacity=0.25,
        max_zoom=15,
        gradient=gradient
    ).add_to(fg)

    m.add_child(fg)
    return m


def add_traffic_sensors_layer(m, df_sensors):
    fg = folium.FeatureGroup(name="[수집기] 스마트 교통량")
    cluster = make_cluster().add_to(fg)

    for _, r in df_sensors.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]): 
            continue

        name = str(r.get("name",""))
        road_addr = str(r.get("road_address",""))
        jibun_addr = str(r.get("jibun_address",""))

        # 7월 평균 유동량(있으면 표시)
        july_mean = r.get("july_mean", np.nan)
        if pd.notna(july_mean):
            mean_txt = f"\n7월 일평균 유동량: {int(round(july_mean)):,}대"
        else:
            mean_txt = ""

        tooltip = folium.Tooltip(
            f"{name}{mean_txt}\n도로명: {road_addr}\n지번: {jibun_addr}",
            sticky=True
        )
        popup = build_popup_html(
            title=name,
            rows=[
                ("구분", "스마트 교통량 수집기"),
                ("7월 일평균 유동량", f"{int(round(july_mean)):,}대" if pd.notna(july_mean) else "데이터 없음"),
                ("도로명", road_addr),
                ("지번", jibun_addr),
            ]
        )
        folium.Marker(
            [float(r["lat"]), float(r["lon"])],
            tooltip=tooltip,
            popup=popup,
            icon=folium.Icon(color="blue", icon="wifi", prefix="fa")
        ).add_to(cluster)

    m.add_child(fg)
    return m

# =========================
# 혼잡도 격자: 생성/집계/시각화
# =========================
def make_adaptive_grid_over_geom(geom, sensors_df,
                                 base_target_cells=80,
                                 refine_factor=10,
                                 sensor_buffer_m=800.0):
    """
    1) 폴리곤 면적 기준으로 '대략 base_target_cells' 개의 큰 셀 생성
    2) 스마트교통 수집기 주변(sensor_buffer_m)과 겹치는 큰 셀만 refine_factor^2로 분할
    3) 최종 격자를 EPSG:4326으로 반환
    - row/col 은 '큰 셀' 인덱스(세분 셀도 같은 row/col 공유)
    - sub_row/sub_col 로 세분 위치 부여(원하면 CSV에 같이 내보낼 수 있음)
    """
    import math
    from shapely.geometry import box

    # 0) 준비: 센서를 GeoDataFrame(4326) → 32652 로 변환
    gdf_wgs = gpd.GeoDataFrame([{"geometry": geom}], geometry="geometry", crs="EPSG:4326")
    poly_m  = gdf_wgs.to_crs(epsg=32652).geometry.iloc[0]

    if sensors_df is None or not len(sensors_df):
        sensors_gdf_m = gpd.GeoDataFrame([], geometry=[], crs="EPSG:32652")
    else:
        sensors_gdf = gpd.GeoDataFrame(
            sensors_df.dropna(subset=["lat","lon"]).copy(),
            geometry=gpd.points_from_xy(sensors_df["lon"], sensors_df["lat"]),
            crs="EPSG:4326"
        )
        sensors_gdf_m = sensors_gdf.to_crs(epsg=32652)

    # 1) 기본 큰 셀 그리드 (고정 셀 크기)
    area_m2 = float(poly_m.area)
    base_target_cells = max(1, int(base_target_cells))
    cell_area = area_m2 / base_target_cells
    cell_size = math.sqrt(cell_area)
    cell_size = max(300.0, min(cell_size, 3000.0))  # 300m~3km 가드

    minx, miny, maxx, maxy = poly_m.bounds
    ncols = int(math.ceil((maxx - minx) / cell_size))
    nrows = int(math.ceil((maxy - miny) / cell_size))

    # 2) 센서 버퍼(핫스팟) 합성
    if len(sensors_gdf_m):
        hotspot = sensors_gdf_m.buffer(sensor_buffer_m).unary_union
    else:
        hotspot = None

    cells_out = []
    gid = 0
    for r in range(nrows):
        y0 = miny + r * cell_size
        y1 = y0 + cell_size
        for c in range(ncols):
            x0 = minx + c * cell_size
            x1 = x0 + cell_size
            base_cell = box(x0, y0, x1, y1)
            inter = poly_m.intersection(base_cell)
            if inter.is_empty:
                continue

            # 3) 핫스팟과 겹치면 세분화, 아니면 큰 셀 유지
            need_refine = False
            if hotspot is not None and inter.intersects(hotspot):
                need_refine = True

            if not need_refine:
                cells_out.append({
                    "grid_id": f"g{gid}",
                    "row": r, "col": c, "sub_row": -1, "sub_col": -1,
                    "geometry": inter
                })
                gid += 1
            else:
                sub_size = cell_size / float(refine_factor)
                for sr in range(refine_factor):
                    sy0 = y0 + sr * sub_size
                    sy1 = sy0 + sub_size
                    for sc in range(refine_factor):
                        sx0 = x0 + sc * sub_size
                        sx1 = sx0 + sub_size
                        sub_cell = box(sx0, sy0, sx1, sy1)
                        inter2 = inter.intersection(sub_cell)
                        if inter2.is_empty:
                            continue
                        cells_out.append({
                            "grid_id": f"g{gid}",
                            "row": r, "col": c, "sub_row": sr, "sub_col": sc,
                            "geometry": inter2
                        })
                        gid += 1

    grid_m = gpd.GeoDataFrame(cells_out, geometry="geometry", crs="EPSG:32652")
    grid_wgs = grid_m.to_crs(epsg=4326)
    return grid_wgs

def make_uniform_grid_over_geom(geom, target_cells=80, min_cell_m=300.0, max_cell_m=3000.0):
    """
    천안시 폴리곤 전체를 균일한 큰 격자로 자릅니다(세분화 없음).
    - target_cells: 대략 이 개수에 맞는 셀 크기를 자동 산정
    - 결과 컬럼: grid_id, row, col, sub_row=-1, sub_col=-1, geometry(4326)
    """
    import math
    from shapely.geometry import box

    # WGS84 -> UTM(32652)로 투영해 '미터' 단위로 자르기
    gdf_wgs = gpd.GeoDataFrame([{"geometry": geom}], geometry="geometry", crs="EPSG:4326")
    poly_m  = gdf_wgs.to_crs(epsg=32652).geometry.iloc[0]

    area_m2 = float(poly_m.area)
    target_cells = max(1, int(target_cells))
    # 타깃 개수에 맞춰 셀 한 변 길이 추정
    cell_size = math.sqrt(area_m2 / target_cells)
    cell_size = max(min_cell_m, min(cell_size, max_cell_m))  # 가드

    minx, miny, maxx, maxy = poly_m.bounds
    ncols = int(math.ceil((maxx - minx) / cell_size))
    nrows = int(math.ceil((maxy - miny) / cell_size))

    cells = []
    gid = 0
    for r in range(nrows):
        y0 = miny + r * cell_size
        y1 = y0 + cell_size
        for c in range(ncols):
            x0 = minx + c * cell_size
            x1 = x0 + cell_size
            base = box(x0, y0, x1, y1)
            inter = poly_m.intersection(base)
            if inter.is_empty:
                continue
            cells.append({
                "grid_id": f"g{gid}",
                "row": r, "col": c,
                "sub_row": -1, "sub_col": -1,
                "geometry": inter
            })
            gid += 1

    grid_m  = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:32652")
    grid_wgs = grid_m.to_crs(epsg=4326)
    return grid_wgs

def _to_gdf_points(df, lon_col="lon", lat_col="lat"):
    g = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    return g

def aggregate_metrics_by_grid(
    grid_gdf,
    df_cat,
    df_sensors,
    df_enf,
    df_pub=None,
    df_pri=None,
    traffic_value_col_candidates=("july_mean","traffic","value")
):
    """
    반환: GeoDataFrame
      - grid_id,row,col,sub_row,sub_col, 경계/중심 좌표
      - facilities_count, public_count, private_count, traffic_sum, enforcement_sum
      - score_facilities, score_traffic, score_enforcement
      - congestion_score_raw (세 점수 평균)
      - congestion_score (전체 격자 기준 0~10 재표준화 값)
    """
    import numpy as np
    import pandas as pd
    import geopandas as gpd

    grid = grid_gdf.copy()
    for col in ["grid_id","row","col","sub_row","sub_col"]:
        if col not in grid.columns:
            grid[col] = -1 if col in ("row","col","sub_row","sub_col") else None

    def _to_points_gdf(df, lat="lat", lon="lon"):
        if df is None or len(df) == 0:
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")
        d = df.dropna(subset=[lat, lon]).copy()
        d[lat] = pd.to_numeric(d[lat], errors="coerce")
        d[lon] = pd.to_numeric(d[lon], errors="coerce")
        d = d.dropna(subset=[lat, lon])
        if len(d) == 0:
            return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")
        return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d[lon], d[lat]), crs="EPSG:4326")

    g_cat = _to_points_gdf(df_cat)
    g_enf = _to_points_gdf(df_enf)
    g_sns = _to_points_gdf(df_sensors)
    g_pub = _to_points_gdf(df_pub) if df_pub is not None else gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")
    g_pri = _to_points_gdf(df_pri) if df_pri is not None else gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

    # 교통량 컬럼 결정
    traffic_col = None
    for c in traffic_value_col_candidates:
        if c in g_sns.columns:
            traffic_col = c
            break
    if traffic_col is None:
        g_sns["__traffic__"] = 0.0
        traffic_col = "__traffic__"
    g_sns[traffic_col] = pd.to_numeric(g_sns[traffic_col], errors="coerce").fillna(0.0)

    # 공간 인덱스
    sidx_cat = getattr(g_cat, "sindex", None)
    sidx_enf = getattr(g_enf, "sindex", None)
    sidx_sns = getattr(g_sns, "sindex", None)
    sidx_pub = getattr(g_pub, "sindex", None)
    sidx_pri = getattr(g_pri, "sindex", None)

    out_rows = []
    for g in grid.itertuples(index=False):
        poly = g.geometry
        if poly is None or poly.is_empty:
            continue

        # 주변시설 개수
        if len(g_cat):
            cand = list(sidx_cat.query(poly)) if sidx_cat is not None else g_cat.index
            fac_cnt = int(g_cat.iloc[cand].within(poly).sum())
        else:
            fac_cnt = 0

        # 공영/민영 주차장 개수
        if len(g_pub):
            cand = list(sidx_pub.query(poly)) if sidx_pub is not None else g_pub.index
            public_cnt = int(g_pub.iloc[cand].within(poly).sum())
        else:
            public_cnt = 0

        if len(g_pri):
            cand = list(sidx_pri.query(poly)) if sidx_pri is not None else g_pri.index
            private_cnt = int(g_pri.iloc[cand].within(poly).sum())
        else:
            private_cnt = 0

        # 단속 합계
        if len(g_enf):
            cand = list(sidx_enf.query(poly)) if sidx_enf is not None else g_enf.index
            enf_in = g_enf.iloc[cand][g_enf.iloc[cand].within(poly)]
            enf_sum = float(pd.to_numeric(enf_in.get("count", 0), errors="coerce").fillna(0.0).sum())
        else:
            enf_sum = 0.0

        # 교통량 합계
        if len(g_sns):
            cand = list(sidx_sns.query(poly)) if sidx_sns is not None else g_sns.index
            sns_in = g_sns.iloc[cand][g_sns.iloc[cand].within(poly)]
            traffic_sum = float(pd.to_numeric(sns_in[traffic_col], errors="coerce").fillna(0.0).sum())
        else:
            traffic_sum = 0.0

        minx, miny, maxx, maxy = poly.bounds
        cent = poly.centroid

        out_rows.append({
            "grid_id": getattr(g, "grid_id"),
            "row": getattr(g, "row", -1),
            "col": getattr(g, "col", -1),
            "sub_row": getattr(g, "sub_row", -1),
            "sub_col": getattr(g, "sub_col", -1),
            "min_lon": float(minx),
            "min_lat": float(miny),
            "max_lon": float(maxx),
            "max_lat": float(maxy),
            "centroid_lon": float(cent.x),
            "centroid_lat": float(cent.y),
            "facilities_count": int(fac_cnt),
            "public_count": int(public_cnt),
            "private_count": int(private_cnt),
            "traffic_sum": float(traffic_sum),
            "enforcement_sum": float(enf_sum),
            "geometry": poly
        })

    out = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=grid.crs)

    # --- 점수화: 0~10 스케일 ---
    def _scale_0_10(series, vmin=None, vmax=None):
        s = pd.to_numeric(series, errors="coerce").astype(float).fillna(0.0)
        mn = float(s.min()) if vmin is None else float(vmin)
        mx = float(s.max()) if vmax is None else float(vmax)
        if mx <= mn:
            return pd.Series(np.zeros(len(s)), index=series.index, dtype=float)
        return (s - mn) * (10.0 / (mx - mn))

    # 시설 점수
    out["score_facilities"] = _scale_0_10(out["facilities_count"])

    # 교통량 점수 (없음=관측 최솟값 간주)
    traffic_obs = out.loc[out["traffic_sum"] > 0, "traffic_sum"]
    traffic_min_obs = float(traffic_obs.min()) if len(traffic_obs) else 0.0
    traffic_for_score = np.where(out["traffic_sum"] > 0, out["traffic_sum"], traffic_min_obs)
    out["score_traffic"] = _scale_0_10(pd.Series(traffic_for_score, index=out.index))
    out["traffic_sum_imputed"] = traffic_for_score

    # 단속 점수
    out["score_enforcement"] = _scale_0_10(out["enforcement_sum"])

    # 종합(원시) 점수: 동일 가중 평균
    out["congestion_score_raw"] = (
        out["score_facilities"] + out["score_traffic"] + out["score_enforcement"]
    ) / 3.0

    # (유지) 0~10 스케일
    out["congestion_score"] = _scale_0_10(out["congestion_score_raw"])

    # (신규) 0~100 스케일
    mn = float(out["congestion_score_raw"].min())
    mx = float(out["congestion_score_raw"].max())
    if mx <= mn:
        out["congestion_score_100"] = 0.0
    else:
        out["congestion_score_100"] = (out["congestion_score_raw"] - mn) * (100.0 / (mx - mn))


    # 보기 좋게 숫자형 정돈
    float_cols = [
        "traffic_sum","enforcement_sum",
        "score_facilities","score_traffic","score_enforcement",
        "congestion_score_raw","congestion_score"
    ]
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    return out


def add_congestion_grid_layer(
    m,
    grid_scores_gdf,
    *,
    value_col="congestion_score_100",
    layer_name="[격자] 혼잡도",
    caption="혼잡도 점수 (0=낮음, 100=높음)",
    vmin=None, vmax=None
):
    import branca

    df = grid_scores_gdf.copy()

    # 값 컬럼 보정
    if value_col not in df.columns:
        if "congestion_score" in df.columns:
            df["__value__"] = pd.to_numeric(df["congestion_score"], errors="coerce").fillna(0.0) * 10.0
            value_col = "__value__"
            vmin, vmax = 0.0, 100.0
        else:
            raise ValueError(f"'{value_col}' 컬럼이 없습니다.")

    # 컬러 스케일
    vals = pd.to_numeric(df[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(vals):
        vmin, vmax = 0.0, 100.0
    else:
        vmin = 0.0 if vmin is None else vmin
        vmax = float(vals.max()) if vmax is None else vmax

    cmap = branca.colormap.LinearColormap(
        colors=["#2ECC71","#F1C40F","#E67E22","#E74C3C"],
        vmin=vmin, vmax=vmax
    )
    cmap.caption = caption
    # cmap.add_to(m)

    # 팝업 HTML을 '문자열'로 만들어 속성에 저장
    def _make_popup_row(r):
        # 표시는 100점만, 10점은 제거
        score100   = float(r.get(value_col, 0.0) or 0.0)
        facilities = int(r.get("facilities_count", 0) or 0)
        public_cnt = int(r.get("public_count", 0) or 0)
        private_cnt= int(r.get("private_count", 0) or 0)
        # ✅ 팝업 표시에 '최솟값 대입' 반영된 교통값 사용
        traffic    = float(r.get("traffic_sum_imputed", r.get("traffic_sum", 0.0)) or 0.0)
        enf_sum    = float(r.get("enforcement_sum", 0.0) or 0.0)

        return build_popup_html_str(
            title=f"격자 {r.get('grid_id','')}",
            rows=[
                ("혼잡도 (0~100)", f"{score100:.1f}"),
                ("주변시설 개수",   f"{facilities:,}"),
                ("공영 주차장 개수", f"{public_cnt:,}"),
                ("민영 주차장 개수", f"{private_cnt:,}"),
                # ✅ 라벨을 '일 평균 유동량(7월)'로 교체
                ("일 평균 유동량(7월)", f"{int(round(traffic)):,}"),
                ("단속건수 합계",   f"{int(round(enf_sum)):,}")
            ],
            # 넓은 팝업 레이아웃 보장
            width=500
        )


    df["__popup__"] = df.apply(_make_popup_row, axis=1)

    # 스타일
    def style_fn(feat):
        try:
            v = float(feat["properties"].get(value_col, 0.0) or 0.0)
        except Exception:
            v = 0.0
        return {"color": "#555555", "weight": 1, "fillColor": cmap(v), "fillOpacity": 0.5}

    def highlight_fn(_):
        return {"weight": 2, "color": "#000000"}

    gj = folium.GeoJson(
        data=json.loads(df.to_json()),
        name=layer_name,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["grid_id", value_col],
            aliases=["격자", "혼잡도"],
            localize=True
        ),
        popup=folium.features.GeoJsonPopup(
            fields=["__popup__"],
            labels=False,
            parse_html=True
        )
    )
    m.add_child(gj)
    return m



def export_grid_scores_csv(grid_scores_gdf, path):
    cols = [
        "grid_id","row","col","sub_row","sub_col",
        "min_lon","min_lat","max_lon","max_lat",
        "public_count","private_count",
        "centroid_lon","centroid_lat",
        "facilities_count","traffic_sum","traffic_sum_imputed","enforcement_sum",
        "score_facilities","score_traffic","score_enforcement",
        "congestion_score_raw",          # 원시 평균 점수(0~10 범위)
        "congestion_score",              # 재표준화 0~10
        "congestion_score_100"           # ✅ 재표준화 0~100 (신규)
    ]
    df = pd.DataFrame(grid_scores_gdf.drop(columns="geometry"))
    # 누락된 컬럼이 있으면 생성(에러 방지)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

# =========================================
# (NEW) 선택된 폴리곤들을 고정 크기 소격자로 쪼개기
# =========================================
def make_fixed_subgrid_over_polygons(selected_grid_gdf, sub_rows=10, sub_cols=10):
    """
    선택된 격자(폴리곤) 각각을 sub_rows x sub_cols 로 균등 분할합니다.
    - 부모 격자의 row/col은 그대로 상속
    - sub_row/sub_col 은 0..sub_rows-1 / 0..sub_cols-1 로 부여
    - grid_id 는 '부모ID_sr{sr}sc{sc}' 형태
    반환: EPSG:4326 GeoDataFrame
    """
    if selected_grid_gdf is None or not len(selected_grid_gdf):
        return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

    parent_wgs = selected_grid_gdf.copy()
    if parent_wgs.crs is None:
        parent_wgs.set_crs(epsg=4326, inplace=True)

    parent_m = parent_wgs.to_crs(epsg=32652)  # meter 단위로 쪼개야 정확

    out_rows = []
    for p in parent_m.itertuples(index=False):
        poly = p.geometry
        if poly is None or poly.is_empty:
            continue

        minx, miny, maxx, maxy = poly.bounds
        dx = (maxx - minx) / float(sub_cols)
        dy = (maxy - miny) / float(sub_rows)

        for sr in range(sub_rows):
            y0 = miny + sr * dy
            y1 = y0 + dy
            for sc in range(sub_cols):
                x0 = minx + sc * dx
                x1 = x0 + dx
                cell = box(x0, y0, x1, y1)
                inter = poly.intersection(cell)
                if inter.is_empty:
                    continue
                out_rows.append({
                    "grid_id": f"{getattr(p, 'grid_id')}_sr{sr}sc{sc}",
                    "row": getattr(p, "row", -1),
                    "col": getattr(p, "col", -1),
                    "sub_row": sr,
                    "sub_col": sc,
                    "geometry": inter
                })

    sub_m = gpd.GeoDataFrame(out_rows, geometry="geometry", crs="EPSG:32652")
    sub_wgs = sub_m.to_crs(epsg=4326)
    return sub_wgs


# =========================================
# (NEW) 0~100 재표준화 유틸
# =========================================
def _rescale_0_100(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vmin = float(s.min())
    vmax = float(s.max())
    if vmax <= vmin:
        return pd.Series(np.zeros(len(s)), index=series.index, dtype=float)
    return (s - vmin) * (100.0 / (vmax - vmin))



# =========================
# Helpers
# =========================
def _safe_float(x):
    try: return float(x)
    except Exception: return np.nan

def _inside(poly, lon, lat):
    try: return poly.contains(Point(float(lon), float(lat)))
    except Exception: return False

# =========================
# 메인
# =========================
if __name__ == "__main__":
    # 1) 천안 경계 (+구별 GDF)
    cheonan_gdf, cheonan_geom, gu_gdf_map = load_cheonan_boundary_shp(SHP_PATH)
    minX, minY, maxX, maxY = cheonan_geom.bounds
    print(f"[INFO] Cheonan bbox: ({minX:.6f}, {minY:.6f}) ~ ({maxX:.6f}, {maxY:.6f})")

    # 2) 카테고리별 수집 (천안 폴리곤 교차 셀만)
    GRID_X, GRID_Y = 6, 4
    by_id = {}
    for gc in TARGET_GROUPS:
        print(f"[INFO] Collecting category: {gc}")
        raw_docs = overlapped_category_in_polygon(gc, (minX, minY, maxX, maxY), GRID_X, GRID_Y, cheonan_geom,
                                                  headers=HEADERS, page_size=PAGE_SIZE)
        for d in raw_docs:
            pid = d.get("id")
            if not pid: continue
            if not category_passes_filter(gc, d):  # 대학/대형병원 필터 등
                continue
            lon = _safe_float(d.get("x")); lat = _safe_float(d.get("y"))
            if np.isnan(lon) or np.isnan(lat): continue
            if not _inside(cheonan_geom, lon, lat): continue
            if pid not in by_id:
                by_id[pid] = {**d, "_groups": {gc}}
            else:
                by_id[pid]["_groups"].add(gc)

    # 2.1) 공공기관(PO3) 키워드 보강: 우체국/보건지소/보건진료소를 PO3로 묶기
    print("[INFO] Keyword backfill for PO3: 우체국/보건지소/보건진료소")

    # 포함/제외 규칙 (편의점 택배취급점 등 잡음 제거)
    _re_post_office = re.compile(r"(우체국|Post\s*Office)", re.IGNORECASE)
    _re_health_post = re.compile(r"(보건지소|보건진료소|보건소)", re.IGNORECASE)
    _re_exclude = re.compile(r"(택배|편의점|CU|GS25|세븐일레븐|7\-?Eleven|이마트24|무인|대리점|편의)", re.IGNORECASE)

    KEYWORDS_PO3_EXTRA = ["우체국", "보건지소", "보건진료소", "보건소"]

    extra_docs_all = []
    for kw in KEYWORDS_PO3_EXTRA:
        extra_docs_all.extend(
            overlapped_keyword_in_polygon(
                kw, (minX, minY, maxX, maxY), GRID_X, GRID_Y, cheonan_geom,
                headers=HEADERS, page_size=PAGE_SIZE
            )
        )

    added_cnt = 0
    for d in extra_docs_all:
        pid = d.get("id")
        if not pid:
            continue

        name = str(d.get("place_name", "") or "")
        catname = str(d.get("category_name", "") or "")
        lon = _safe_float(d.get("x")); lat = _safe_float(d.get("y"))

        # 좌표/경계 체크
        if np.isnan(lon) or np.isnan(lat):
            continue
        if not _inside(cheonan_geom, lon, lat):
            continue

        # 포함/제외 필터
        is_post = bool(_re_post_office.search(name) or _re_post_office.search(catname))
        is_health = bool(_re_health_post.search(name) or _re_health_post.search(catname))
        if not (is_post or is_health):
            continue
        if _re_exclude.search(name):
            continue

        # by_id에 주입: 그룹을 PO3로 강제 태깅해 공공기관 레이어로 표시
        if pid not in by_id:
            by_id[pid] = {**d, "_groups": {"PO3"}}
            added_cnt += 1
        else:
            prev = len(by_id[pid].get("_groups", set()))
            by_id[pid].setdefault("_groups", set()).add("PO3")
            if len(by_id[pid]["_groups"]) > prev:
                added_cnt += 1

    print(f"[INFO] PO3 keyword backfill merged: +{added_cnt}")

    # 2.2) HP8 보강: '요양병원' + '재활병원' 키워드
    print("[INFO] Keyword backfill for HP8: 요양/재활병원")

    _re_care_hospital = re.compile(r"(요양병원|재활병원)", re.IGNORECASE)
    # 제외 규칙: 원치 않는 병원 유형 필터링 (선택)
    _re_exclude_med = re.compile(r"(치과|한의원|동물|의원)", re.IGNORECASE)

    KEYWORDS_HP8_EXTRA = ["요양병원", "재활병원"]

    extra_hp8_docs = []
    for kw in KEYWORDS_HP8_EXTRA:
        extra_hp8_docs.extend(
            overlapped_keyword_in_polygon(
                kw, (minX, minY, maxX, maxY), GRID_X, GRID_Y, cheonan_geom,
                headers=HEADERS, page_size=PAGE_SIZE
            )
        )

    added_hp8 = 0
    for d in extra_hp8_docs:
        pid = d.get("id")
        if not pid:
            continue
        name = str(d.get("place_name", "") or "")
        catname = str(d.get("category_name", "") or "")
        lon = _safe_float(d.get("x")); lat = _safe_float(d.get("y"))

        if np.isnan(lon) or np.isnan(lat):
            continue
        if not _inside(cheonan_geom, lon, lat):
            continue

        # 포함/제외 규칙 적용
        if not (_re_care_hospital.search(name) or _re_care_hospital.search(catname)):
            continue
        if _re_exclude_med.search(name):
            continue

        if pid not in by_id:
            by_id[pid] = {**d, "_groups": {"HP8"}}
            added_hp8 += 1
        else:
            prev = len(by_id[pid].get("_groups", set()))
            by_id[pid].setdefault("_groups", set()).add("HP8")
            if len(by_id[pid]["_groups"]) > prev:
                added_hp8 += 1

    print(f"[INFO] HP8 keyword backfill merged: +{added_hp8}")


    results = list(by_id.values())

#    # 2.5) 대학 누락 보정: 키워드 기반 수집(“대학” + 사용자 특정명)
#    print("[INFO] Keyword backfill for universities...")
#    KEYWORDS_UNIV = ["대학", "대학교", "University", "연암대학교", "성남평생교육원 성남대학"]
#    extra_docs = []
#    for kw in KEYWORDS_UNIV:
#        extra_docs.extend(
#            overlapped_keyword_in_polygon(kw, (minX, minY, maxX, maxY), GRID_X, GRID_Y, cheonan_geom,
#                                          headers=HEADERS, page_size=PAGE_SIZE)
#        )
#
#    # 필터링(와플대학 제외) + 천안 폴리곤 내부만 + by_id에 주입(그룹은 'SC4'로 태깅)
#    for d in extra_docs:
#        pid = d.get("id")
#        if not pid:
#            continue
#        name = str(d.get("place_name", "") or "")
#        catname = str(d.get("category_name", "") or "")
#        # '대학' 계열 키워드인지 확인 (와플대학 제외)
#        if not (_re_univ.search(name) or _re_univ.search(catname)):
#            continue
#        if _re_waffle.search(name):
#            continue
#
#        lon = _safe_float(d.get("x")); lat = _safe_float(d.get("y"))
#        if np.isnan(lon) or np.isnan(lat):
#            continue
#        if not _inside(cheonan_geom, lon, lat):
#            continue
#
#        # 기존에 있으면 그룹만 보강, 없으면 신규 삽입
#        if pid not in by_id:
#            by_id[pid] = {**d, "_groups": {"SC4"}}
#        else:
#            by_id[pid].setdefault("_groups", set()).add("SC4")


    # 3) DF 변환
    df_cat = pd.DataFrame({
        "id":   [p.get("id","") for p in results],
        "name": [p.get("place_name","") for p in results],
        "lon":  [_safe_float(p.get("x")) for p in results],
        "lat":  [_safe_float(p.get("y")) for p in results],
        "road_address": [p.get("road_address_name","") for p in results],
        "jibun_address": [p.get("address_name","") for p in results],
        "url":  [p.get("place_url","") for p in results],
        "category_name": [p.get("category_name","") for p in results],
        "group_code": [sorted(list(p.get("_groups", [])))[0] if p.get("_groups") else None for p in results],
    })
    df_cat = df_cat.dropna(subset=["lat","lon"]).reset_index(drop=True)
    print(f"[INFO] POI (category-based, filtered): {len(df_cat)}")

    # 3.1) 공공기관 정규화/중복 제거
    df_cat = dedup_po3_public_institutions(df_cat)
    print(f"[INFO] POI after PO3 normalization/dedup: {len(df_cat)}")

    # 3.2) DF 변환 직후, 캠퍼스 중복 제거 (CT1: 문화시설)
    df_cat = dedup_campus_pois(
        df_cat,
        categories=("CT1",),        # 필요시 ("CT1","PO3") 등 확장
        merge_radius_m=400,         # 독립기념관 같은 단지 합치기 반경
        min_cluster_size=3,         # 최소 3개 이상 모이면 합침
        name_guard=True
    )


    # 3.3) 수동 보강 POI(누락 대학 직접 주입)
    custom_rows = [
        {
            "id": "custom_univ_yeonam",
            "name": "연암대학",
            "lat": 36.9469889,
            "lon": 127.15545,
            "road_address": "충남 천안시 서북구 성환읍 연암로 313",  # ← 추가
            "jibun_address": "충남 천안시 서북구 성환읍 수향리 72-3",   # (지번 임시값이면 공란 가능)
            "url": "https://place.map.kakao.com/8088750",
            "category_name": "학교 > 대학교",
            "group_code": "SC4",
        },
    ]

    if len(custom_rows):
        df_custom = pd.DataFrame(custom_rows)
        # 기존 스키마와 컬럼 일치
        needed = ["id","name","lat","lon","road_address","jibun_address","url","category_name","group_code"]
        for col in needed:
            if col not in df_custom.columns:
                df_custom[col] = ""
        # 기존 df_cat과 병합(이름+좌표 기준 중복 제거)
        before = len(df_cat)
        df_cat = pd.concat([df_cat, df_custom[needed]], ignore_index=True)
        df_cat = df_cat.drop_duplicates(subset=["name","lat","lon"]).reset_index(drop=True)
        print(f"[INFO] Custom POIs merged: +{len(df_cat)-before}")


    # 4) CSV 저장(옵션)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    cat_csv = os.path.join(SAVE_DIR, f"cheonan_POI_category_{ts}.csv")
    df_cat.to_csv(cat_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved CSV (category): {cat_csv}")

    # 5) 주차장/수집기 로드 (+천안 내부 필터)
    df_pub = load_public_parking(PUBLIC_PARKING_CSV)
    df_pri = load_private_parking(PRIVATE_PARKING_XLSX)
    if len(df_pub):
        df_pub = df_pub[df_pub.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)
    if len(df_pri):
        df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

    # (A) 수집기 좌표
    df_sensors = load_traffic_sensors_exact(SENSORS_CSV)
    if len(df_sensors):
        df_sensors = df_sensors[df_sensors.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

    # (B) 7월 통계 불러와 '원본명(=join_key)'과 '교차로명'으로 조인
    df_stats = load_traffic_stats(TRAFFIC_STATS_CSV)   # 교차로명, july_mean, july_sum, days
    if len(df_sensors) and len(df_stats):
        df_sensors = df_sensors.merge(df_stats, left_on="join_key", right_on="교차로명", how="left")

    # (C) 불법주정차 단속 포인트 로드 (23년 + 24년)
    df_enf_23 = load_enforcement_points(ENFORCEMENT_CSV_23)
    df_enf_24 = load_enforcement_points(ENFORCEMENT_CSV_24)

    # 연도 구분
    if len(df_enf_23): df_enf_23["year"] = 2023
    if len(df_enf_24): df_enf_24["year"] = 2024

    # 두 파일 합치기
    df_enf = pd.concat([df_enf_23, df_enf_24], ignore_index=True) if (len(df_enf_23) or len(df_enf_24)) else pd.DataFrame()

    # 천안시 내부만 남기기
    if len(df_enf):
        df_enf = df_enf[df_enf.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)
    print(f"[INFO] Enforcement points: {len(df_enf)}")

    # 6) 지도 생성 + VWorld 베이스 + 라벨
    m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM, tiles=None)
    add_vworld_base_layers(m)
    m.add_child(MiniMap())

    # 🔥 불법주정차 히트맵 추가 (지도 만든 후!)
    if len(df_enf):
        add_enforcement_heatmap_layer(m, df_enf)
    else:
        print("[INFO] Enforcement dataframe is empty. No heatmap added.")

    # 7) 카테고리 레이어
    add_category_layers(m, df_cat)

    # 8) 주차장 레이어
    if len(df_pub) or len(df_pri):
        add_parking_layers_to_map(m, df_pub, df_pri)

    # 9) 수집기 레이어
    if len(df_sensors):
        add_traffic_sensors_layer(m, df_sensors)

#    # 10) 불법주정차 단속 레이어 추가
#    if len(df_enf):
#        add_enforcement_layer(m, df_enf)
#    else:
#        print("[INFO] Enforcement dataframe is empty. No layer added.")

    # 10) 격자 생성 + 혼잡도 계산
    # === 혼잡도 격자 생성 → 집계 → CSV 저장 → 지도 표시 ===
    # === (NEW) 적응형 격자 생성 → 집계 → CSV → 레이어 추가 ===
    # 1단계: 큰(균일) 격자만 생성해서 점수/지도로 확인
    grid_gdf = make_uniform_grid_over_geom(
        cheonan_geom,
        target_cells=80,    # 필요 시 60~120 등으로 조절
        min_cell_m=300.0,
        max_cell_m=3000.0
    )


    # 점수 계산: '스마트교통량 없음 → 최솟값 간주' 규칙 포함된 버전이어야 합니다
    grid_scores = aggregate_metrics_by_grid(
        grid_gdf, df_cat, df_sensors, df_enf, df_pub=df_pub, df_pri=df_pri
    )

    # =========================
    # (NEW) 대격자 중 총점 5점 이상만 다시 10x10 소격자로 분할 → 집계 → 0~100 재표준화 → CSV/지도
    # =========================

    # 1) 5점(100점 만점 기준) 이상인 대격자만 선별
    coarse_hot = grid_scores.loc[grid_scores["congestion_score_100"] >= 5.0].copy()
    print(f"[INFO] Refine 대상 대격자 수: {len(coarse_hot)}")

    # 2) 각 대격자를 10x10 = 100개 소격자로 균등 분할
    subgrid = make_fixed_subgrid_over_polygons(coarse_hot, sub_rows=10, sub_cols=10)
    print(f"[INFO] 생성된 소격자 수: {len(subgrid)}")

    # 3) 소격자 집계 (같은 함수 재사용)
    sub_scores = aggregate_metrics_by_grid(
        subgrid,
        df_cat=df_cat,
        df_sensors=df_sensors,
        df_enf=df_enf,
        df_pub=df_pub,
        df_pri=df_pri
    )

    # 4) 소격자끼리만 0~100 재표준화 (원시 평균 점수 기준)
    #    - aggregate_metrics_by_grid 안에서 이미 congestion_score_raw(0~10 평균),
    #      congestion_score(0~10 재표준화), congestion_score_100(전체 0~100)가 있을 수 있으므로
    #      '소집단 내' 재표준화를 별도 컬럼으로 둡니다.
    sub_scores["congestion_score_100_local"] = _rescale_0_100(sub_scores["congestion_score_raw"])
    # (원하시면 기존 100점 컬럼을 덮어써도 됩니다)
    # sub_scores["congestion_score_100"] = _rescale_0_100(sub_scores["congestion_score_raw"])

    # 5) CSV 내보내기 (소격자 전용)
    ts_ref = datetime.now().strftime("%Y%m%d_%H%M")
    sub_csv = os.path.join(SAVE_DIR, f"cheonan_congestion_subgrid_{ts_ref}.csv")
    # 소격자 CSV에는 지역 재표준화 점수 컬럼도 같이 저장
    sub_export_cols = [
        "grid_id","row","col","sub_row","sub_col",
        "min_lon","min_lat","max_lon","max_lat",
        "centroid_lon","centroid_lat",
        "public_count","private_count",
        "facilities_count","traffic_sum","traffic_sum_imputed","enforcement_sum",
        "score_facilities","score_traffic","score_enforcement",
        "congestion_score_raw","congestion_score","congestion_score_100",
        "congestion_score_100_local"  # ✅ 소집단 재표준화
    ]
    tmp = pd.DataFrame(sub_scores.drop(columns="geometry"))
    for c in sub_export_cols:
        if c not in tmp.columns:
            tmp[c] = np.nan
    tmp[sub_export_cols].to_csv(sub_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved refined subgrid scores to {sub_csv}")

    # 6) 지도에 소격자 레이어 추가 (체크박스로 구분)
    #    - 대격자: 기존 호출 유지
    #    - 소격자: 레이어명/값컬럼만 다르게
    add_congestion_grid_layer(m, grid_scores, value_col="congestion_score_100",
                              layer_name="[격자-대] 혼잡도(전체 표준화)",
                              caption="혼잡도(대격자, 전체 표준화) 0=낮음 100=높음",
                              vmin=0, vmax=100)

    # 🔥 소격자 레이어: 소격자끼리 0~100 재표준화한 값으로 표시
    if len(sub_scores):
        add_congestion_grid_layer(
            m,
            sub_scores,
            value_col="congestion_score_100_local",   # ← 소격자 전용 컬럼
            layer_name="[격자-소] 혼잡도(재분할/재표준화)",
            caption="혼잡도(소격자, 지역 재표준화) 0=낮음 100=높음",
            vmin=0, vmax=100
        )
    else:
        print("[INFO] 소격자 점수 데이터가 비어 있어 레이어를 추가하지 않습니다.")


    # CSV 내보내기
    ts2 = datetime.now().strftime("%Y%m%d_%H%M")
    grid_csv = os.path.join(SAVE_DIR, f"cheonan_congestion_grid_{ts2}.csv")
    export_grid_scores_csv(grid_scores, grid_csv)
    print(f"[INFO] Saved grid scores to {grid_csv}")

    # 11) 경계(halo + 전체 라인)
    folium.GeoJson(
        cheonan_gdf, name="[경계] 천안시 (halo)",
        style_function=lambda x: {"color": BOUNDARY_HALO_COLOR, "weight": BOUNDARY_HALO_WEIGHT, "opacity": 0.9},
        control=False
    ).add_to(m)
    folium.GeoJson(
        cheonan_gdf, name="[경계] 천안시",
        style_function=lambda x: {"color": BOUNDARY_COLOR, "weight": BOUNDARY_LINE_WEIGHT, "opacity": 1.0},
        highlight_function=lambda x: {"weight": BOUNDARY_LINE_WEIGHT + 1.5, "color": BOUNDARY_HALO_COLOR}
    ).add_to(m)

    # 11-1) 구별 경계 (체크박스 개별 on/off)
    dn_gdf = gu_gdf_map.get("동남구")
    if dn_gdf is not None and len(dn_gdf):
        folium.GeoJson(
            dn_gdf,
            name="[경계] 동남구",
            style_function=lambda x: {"color": DN_COLOR, "weight": 3, "opacity": 1.0},
            highlight_function=lambda x: {"weight": 4, "color": "#FFFFFF"}
        ).add_to(m)

    sb_gdf = gu_gdf_map.get("서북구")
    if sb_gdf is not None and len(sb_gdf):
        folium.GeoJson(
            sb_gdf,
            name="[경계] 서북구",
            style_function=lambda x: {"color": SB_COLOR, "weight": 3, "opacity": 1.0},
            highlight_function=lambda x: {"weight": 4, "color": "#FFFFFF"}
        ).add_to(m)

    # 12) LayerControl + 저장 + 열기
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(SAVE_HTML)
    print(f"[INFO] Saved map to {SAVE_HTML}")
    webbrowser.open('file://' + os.path.realpath(SAVE_HTML))
