# 천안시 MT1(대형마트), SC4(학교) 중 '대학', CT1(문화시설), PO3(공공기관)
# AT4(관광명소), HP8 중 '종합병원' 또는 '대학병원' 또는 '대형병원' 만 필터링
# 공영/민영 주차장 정보 지도에 함께 표시
# 천안시 경계면 정보 가져오기
# 천안시 교통량 수집기 위치 함께 표시
# 배경 위성사진으로 변경, 밝은 테마, 라벨 오버레이
# html 따로 저장
# 위성사진 VWorld에서 끌어옴 (VWORLD API KEY 사용)
# 천안시 경계 선으로 표시
# 서북구, 동남구 나눠서 경계 따로 표시
# 컬러 잘 보이도록 변환

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
    raise RuntimeError("REBUILD_MAP=True인데 환경변수 VWORLD_API_KEY가 비어있습니다.")

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
    <div style="font-size:14px; line-height:1.5; white-space: normal; word-break: keep-all; max-width:{width-10}px;">
        <div style="font-weight:700; margin-bottom:6px;">{safe_title}</div>
        {rows_html}
        {link_html}
    </div>
    """
    iframe = folium.IFrame(html=html, width=width, height=height)
    return folium.Popup(iframe, max_width=width + 10)

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
    .leaflet-popup-content { white-space: normal !important; line-height: 1.5 !important; }
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

    # 3.x) 수동 보강 POI(누락 대학 직접 주입)
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

    # 6) 지도 생성 + VWorld 베이스 + 라벨
    m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM, tiles=None)
    add_vworld_base_layers(m)
    m.add_child(MiniMap())

    # 7) 카테고리 레이어
    add_category_layers(m, df_cat)

    # 8) 주차장 레이어
    if len(df_pub) or len(df_pri):
        add_parking_layers_to_map(m, df_pub, df_pri)

    # 9) 수집기 레이어
    if len(df_sensors):
        add_traffic_sensors_layer(m, df_sensors)

    # 10) 경계(halo + 전체 라인)
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

    # 10-1) 구별 경계 (체크박스 개별 on/off)
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

    # 11) LayerControl + 저장 + 열기
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(SAVE_HTML)
    print(f"[INFO] Saved map to {SAVE_HTML}")
    webbrowser.open('file://' + os.path.realpath(SAVE_HTML))
