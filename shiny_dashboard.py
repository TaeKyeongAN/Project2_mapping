# app.py — Cheonan Parking Need Dashboard (Python Shiny)
# 가중치 슬라이더로 격자/소격자 점수 재계산 + Folium 지도 실시간 갱신

import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from pathlib import Path

from shiny import App, ui, render, reactive

from shiny.types import SilentException

# ---- 지도 레이어 제어 헬퍼들 ----
import folium

def _turn_off_all_overlays(m: folium.Map):
    """베이스 타일 제외 모든 오버레이를 초기 OFF로"""
    def _walk(layer):
        overlay_attr = getattr(layer, "overlay", None)
        # TileLayer(overlay=False)는 베이스맵이므로 제외
        if (overlay_attr is True) or (overlay_attr is None and not isinstance(layer, folium.raster_layers.TileLayer)):
            if hasattr(layer, "show"):
                layer.show = False
        for ch in getattr(layer, "_children", {}).values():
            _walk(ch)
    for ch in list(getattr(m, "_children", {}).values()):
        _walk(ch)

def _select_basemap(m: folium.Map, prefer_names=("Satellite", "위성")):
    """베이스맵들 중 prefer_names가 이름에 포함된 타일만 ON, 나머지 OFF"""
    bases = []
    for ch in list(m._children.values()):
        if isinstance(ch, folium.raster_layers.TileLayer) and getattr(ch, "overlay", False) is False:
            bases.append(ch)
    # 모두 OFF
    for t in bases:
        if hasattr(t, "show"):
            t.show = False
    # 선호 타일 한 개만 ON (없으면 첫 번째)
    for t in bases:
        nm = (getattr(t, "layer_name", "") or "").lower()
        if any(p.lower() in nm for p in prefer_names):
            t.show = True
            break
    else:
        if bases:
            bases[0].show = True

def _set_layer_show_by_name(m: folium.Map, name_substrings, show: bool = True):
    """레이어 이름에 부분일치하는 오버레이들의 show 토글"""
    def _walk(layer):
        nm = getattr(layer, "layer_name", None)
        if nm and any(s in nm for s in name_substrings) and hasattr(layer, "show"):
            layer.show = show
        for ch in getattr(layer, "_children", {}).values():
            _walk(ch)
    for ch in list(m._children.values()):
        _walk(ch)

def _wire_enf_legend_behavior_by_name(m, overlay_name="[단속] 히트맵(불법주정차)", legend_id="enf-legend"):
    """히트맵 오버레이가 켜질 때만 #enf-legend 보이기"""
    from branca.element import Element
    js = f"""
    <script>
    (function() {{
      function norm(s) {{ return (s||'').replace(/\\s+/g,'').trim(); }}
      var target = "{overlay_name}";
      var legendId = "{legend_id}";
      var map = window.{m.get_name()};
      function setVisible(on) {{
        var el = document.getElementById(legendId);
        if (el) el.style.display = on ? 'block' : 'none';
      }}
      map.on('overlayadd', function(e) {{ if (norm(e.name) === norm(target)) setVisible(true); }});
      map.on('overlayremove', function(e) {{ if (norm(e.name) === norm(target)) setVisible(false); }});
      // 초기 동기화
      setTimeout(function() {{
        var labels = document.querySelectorAll('.leaflet-control-layers-overlays label');
        var on = false;
        labels.forEach(function(label) {{
          var txt = (label.textContent || label.innerText || "").trim();
          if (norm(txt) === norm(target)) {{
            var input = label.querySelector('input[type="checkbox"]');
            on = input && input.checked;
          }}
        }});
        setVisible(on);
      }}, 0);
    }})();
    </script>
    """
    m.get_root().html.add_child(Element(js))


# === 코어 함수 ===
from cheonan_mapping_core import (
    # 데이터/레이어 로딩
    load_cheonan_boundary_shp, load_public_parking, load_private_parking,
    load_traffic_sensors_exact, load_traffic_stats, load_enforcement_points,
    add_vworld_base_layers, add_category_layers, add_parking_layers_to_map,
    add_enforcement_heatmap_layer, add_traffic_sensors_layer,
    # 격자/집계/표준화
    make_uniform_grid_over_geom, make_fixed_subgrid_over_polygons,
    aggregate_metrics_by_grid, add_congestion_grid_layer, _rescale_0_100,
    # 유틸
    _inside
)

from folium.plugins import MiniMap

# =========================
# 경로/파일 상수
# =========================
BASE_DIR = Path(__file__).resolve().parent          # .../project2_shiny/dashboard
DATA_DIR = BASE_DIR / "cheonan_data"

SAVE_DIR = str(DATA_DIR)
SHP_PATH = str(DATA_DIR / "N3A_G0100000" / "N3A_G0100000.shp")
PUBLIC_PARKING_CSV   = str(DATA_DIR / "천안도시공사_주차장 현황_20250716.csv")
PRIVATE_PARKING_XLSX = str(DATA_DIR / "충청남도_천안시_민영주차장정보.xlsx")   # (지오코딩 경로 보존)
PRIVATE_PARKING_GEO_CSV = str(DATA_DIR / "민영주차장_geocoded.csv")            # ★ 캐시 CSV 사용
SENSORS_CSV          = str(DATA_DIR / "천안_교차로_행정동_정확매핑.csv")
TRAFFIC_STATS_CSV    = str(DATA_DIR / "스마트교차로_통계.csv")
ENFORCEMENT_CSV_23   = str(DATA_DIR / "천안시_단속장소_위도경도_23년.csv")
ENFORCEMENT_CSV_24   = str(DATA_DIR / "천안시_단속장소_위도경도_24년.csv")
POI_CAT_CSV          = str(DATA_DIR / "cheonan_POI_category_20250820_0949.csv") # ★ 최신 POI CSV

MAP_CENTER_LAT, MAP_CENTER_LON = 36.815, 127.147
MAP_ZOOM = 12
TARGET_GRID_CELLS = 80   # ★ 요구사항 1: 대격자 개수 고정(조절 기능 제거)

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# UI
# =========================
# =========================
# UI (사이드바 제거, 상단 카드에 컨트롤 배치)
# =========================
# =========================
# UI (콤팩트 레이아웃: 3개 카드로 분리 + 작은 컨트롤)
# =========================
# =========================
# UI (콤팩트 레이아웃: 좌측에 '기타' 이동 + 라벨 줄바꿈 방지)
# =========================
app_ui = ui.page_fluid(
    # 컴팩트 + 라벨 한 줄 스타일
    ui.tags.style("""
    .compact-card .card-body { padding: 10px 12px; }
    .compact-card .form-label { margin-bottom: 4px; font-size: 0.92rem; }
    .compact-card .form-select, 
    .compact-card .form-control { padding: 4px 8px; height: 32px; font-size: 0.92rem; }
    .compact-card .btn-sm { padding: 4px 8px; font-size: 0.9rem; }
    .compact-card .form-check { margin-right: 8px; }
    .compact-card .form-check-label { font-size: 0.92rem; }
    .compact-row { row-gap: 8px; }
    /* ▼ 이 카드들에서는 라벨이 절대 줄바꿈되지 않도록 */
    .no-wrap-card .form-label { white-space: nowrap; }
    """),

    ui.h3("천안시 공영주차장 후보지 대시보드 (가중치 조정)"),
    ui.br(),

    # 상단: 좌(지표 가중치 + 기타), 우(격자 / 추천)
    ui.row(
        # ---- 왼쪽: 지표 가중치 + 기타(이동) ----
        ui.column(
            6,
            ui.card(
                ui.card_header("지표 가중치"),
                ui.row(
                    {"class": "compact-row"},
                    ui.column(12, ui.input_slider("w_fac", "주변시설(수요)", min=0, max=100, value=33, step=1, width="100%")),
                    ui.column(12, ui.input_slider("w_trf", "교통량(유동)",   min=0, max=100, value=33, step=1, width="100%")),
                    ui.column(12, ui.input_slider("w_enf", "불법주정차(압력)", min=0, max=100, value=34, step=1, width="100%")),
                ),
                class_="compact-card"
            ),
            ui.br(style="margin:6px 0;"),
            # ← 기타 카드 이동
            ui.card(
                ui.card_header("기타"),
                ui.row(
                    {"class": "compact-row"},
                    ui.column(6, ui.input_checkbox("fast_mode", "빠른 로드", value=True)),
                    ui.column(6, ui.download_button("dl_scores", "점수 CSV 다운로드", class_="btn-outline-secondary btn-sm w-100")),
                    ui.column(12, ui.input_action_button("recalc", "지도 재계산", class_="btn-primary btn-sm w-100")),
                ),
                class_="compact-card"
            ),
        ),

        # ---- 오른쪽: (1) 격자 설정, (2) 추천(고혼잡) ----
        ui.column(
            6,
            # (1) 격자 설정 — 라벨 줄바꿈 방지 클래스 추가
            ui.card(
                ui.card_header("격자 설정"),
                ui.row(
                    {"class": "compact-row"},
                    ui.column(
                        12,
                        ui.input_radio_buttons(
                            "subgrid_n", "소격자(행 = 열)",
                            choices={"10": "10×10", "5": "5×5"},
                            selected="10", inline=True
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_select(
                            "refine_thr_select",
                            "분할 기준(대격자점수)",
                            choices=[str(x) for x in range(5, 51, 5)],
                            selected="5", width="120px"
                        ),
                    ),
                    ui.column(
                        6,
                        ui.input_checkbox("local_norm", "소집단 내 0~100 재표준화", value=True),
                    ),
                ),
                class_="compact-card no-wrap-card"
            ),

            ui.br(style="margin:6px 0;"),

            # (2) 추천(고혼잡) 설정 — 라벨 줄바꿈 방지 클래스 추가
            ui.card(
                ui.card_header("추천(고혼잡) 설정"),
                ui.row(
                    {"class": "compact-row"},
                    ui.column(
                        4,
                        ui.input_select(
                            "rec_thr_select",
                            "추천 기준(점수)",
                            choices=["100","95","90","85","80","75","70"],
                            selected="90", width="100px"
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_select(
                            "penalty_pub",
                            "공영 1개당 차감",
                            choices=["5","10","15","20"],
                            selected="10", width="100px"
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_select(
                            "penalty_pri",
                            "민영 1개당 차감",
                            choices=[str(x) for x in range(1, 11)],
                            selected="5", width="100px"
                        ),
                    ),
                ),
                class_="compact-card no-wrap-card"
            ),
        ),
    ),

    ui.br(),

    # 본문: 지도 + 추천표
    ui.row(
        ui.column(
            8,
            ui.card(
                ui.card_header("지도"),
                ui.output_ui("map_ui"),
                full_screen=True
            )
        ),
        ui.column(
            4,
            ui.card(
                ui.card_header("추천 격자(조정점수 기준)"),
                ui.output_table("recommend_table")
            )
        )
    ),
)



# =========================
# 서버 로직
# =========================
def server(input, output, session):

    def _safe_input_int(getter, default: int) -> int:
        #"""Shiny 입력이 아직 준비 전(SilentException)이어도 기본값을 반환해 안전하게 읽기"""
        try:
            v = getter()           # 예: input.recommend_thr()
            if v is None or str(v).strip() == "":
                return default
            return int(v)
        except SilentException:
            # 초기 렌더링 중 입력값이 아직 없을 때
            return default
        except Exception:
            return default


    # === server() 안, 상단에 역지오코딩 보조 ===
    import os, time, requests
    from functools import lru_cache

    KAKAO_KEY = (os.getenv("KAKAO_REST_KEY") or "").strip()

    @lru_cache(maxsize=2000)
    def _reverse_geocode(lat: float, lon: float) -> str:
        if not KAKAO_KEY:
            return ""  # 키 없으면 주소 생략
        try:
            url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
            headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
            params = {"x": f"{float(lon)}", "y": f"{float(lat)}", "input_coord": "WGS84"}
            resp = requests.get(url, headers=headers, params=params, timeout=(5, 12))
            if resp.status_code != 200:
                return ""
            docs = resp.json().get("documents", [])
            if not docs:
                return ""
            addr = docs[0].get("address") or docs[0].get("road_address") or {}
            # 행정동/읍면 + 구/시 정도만 간략히
            region_3depth = addr.get("region_3depth_name", "")
            region_2depth = addr.get("region_2depth_name", "")
            # 예) "부성1동, 서북구"
            label = ", ".join([p for p in [region_3depth, region_2depth] if p])
            return label
        except Exception:
            return ""


    # ---- (유틸) 최신 POI CSV 불러오기: 고정 경로 사용 ----
    def _load_poi_csv():
        try:
            df = pd.read_csv(POI_CAT_CSV)
            # lat/lon 정리 (컬럼명 방어)
            rename = {}
            if "위도" in df.columns and "lat" not in df.columns: rename["위도"] = "lat"
            if "경도" in df.columns and "lon" not in df.columns: rename["경도"] = "lon"
            df = df.rename(columns=rename)
            df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
            df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
            return df.dropna(subset=["lat","lon"]).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
        
    def _safe_int_input(val, default: int) -> int:
    # """라디오/셀렉트 초기 None 방지: None/''이면 default 반환, 그 외는 int 캐스팅"""
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return int(default)
        try:
            return int(val)
        except Exception:
            return int(default)


    # ---- (유틸) 민영주차장: 캐시 CSV 최우선 ----
    def _load_private_parking_fast(cheonan_geom):
        geo_csv = Path(PRIVATE_PARKING_GEO_CSV)
        if geo_csv.exists():
            try:
                df_pri = pd.read_csv(geo_csv)
                # 보정
                df_pri["lat"] = pd.to_numeric(df_pri["lat"], errors="coerce")
                df_pri["lon"] = pd.to_numeric(df_pri["lon"], errors="coerce")
                df_pri = df_pri.dropna(subset=["lat","lon"])
                df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)]
                return df_pri.reset_index(drop=True)
            except Exception:
                pass
        # 캐시 없으면(또는 실패) 기존 함수(느림)로
        try:
            df_pri = load_private_parking(PRIVATE_PARKING_XLSX)
            df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)
            return df_pri
        except Exception:
            return pd.DataFrame(columns=["id","name","lat","lon","road_address","jibun_address","category","source"])

    # --- 무거운 원본 데이터: 앱 기동 시 1회 로드 ---
    @reactive.calc
    def base_data():
        # 경계
        cheonan_gdf, cheonan_geom, gu_map = load_cheonan_boundary_shp(SHP_PATH)

        fast = bool(input.fast_mode())

        # POI: 고정 CSV에서
        df_cat = _load_poi_csv()
        if len(df_cat):
            df_cat = df_cat[df_cat.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        # 공영
        df_pub = load_public_parking(PUBLIC_PARKING_CSV)
        if len(df_pub):
            df_pub = df_pub[df_pub.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        # 민영: fast면 캐시 CSV 우선
        if fast:
            df_pri = _load_private_parking_fast(cheonan_geom)
        else:
            # 느린 경로(필요 시)
            df_pri = load_private_parking(PRIVATE_PARKING_XLSX)
            if len(df_pri):
                df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        # 수집기 + 7월 통계
        df_sensors = load_traffic_sensors_exact(SENSORS_CSV)
        df_stats   = load_traffic_stats(TRAFFIC_STATS_CSV)
        if len(df_sensors) and len(df_stats):
            df_sensors = df_sensors.merge(df_stats, left_on="join_key", right_on="교차로명", how="left")

        # 단속 (23/24)
        df_enf_23 = load_enforcement_points(ENFORCEMENT_CSV_23)
        df_enf_24 = load_enforcement_points(ENFORCEMENT_CSV_24)
        if len(df_enf_23): df_enf_23["year"] = 2023
        if len(df_enf_24): df_enf_24["year"] = 2024
        df_enf = pd.concat([df_enf_23, df_enf_24], ignore_index=True) if (len(df_enf_23) or len(df_enf_24)) else pd.DataFrame()
        if len(df_enf):
            df_enf = df_enf[df_enf.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        return dict(
            cheonan_geom=cheonan_geom,
            cheonan_gdf=cheonan_gdf,
            gu_map=gu_map,
            df_cat=df_cat,
            df_pub=df_pub,
            df_pri=df_pri,
            df_sensors=df_sensors,
            df_enf=df_enf
        )

    # 점수 결합
    def _apply_weights(df: pd.DataFrame, w_fac: float, w_trf: float, w_enf: float, local_norm: bool):
        s_fac = pd.to_numeric(df.get("score_facilities", 0), errors="coerce").fillna(0.0)
        s_trf = pd.to_numeric(df.get("score_traffic", 0), errors="coerce").fillna(0.0)
        s_enf = pd.to_numeric(df.get("score_enforcement", 0), errors="coerce").fillna(0.0)
        wsum = max(1e-9, (w_fac + w_trf + w_enf))
        df["score_weighted_raw"] = (s_fac * w_fac + s_trf * w_trf + s_enf * w_enf) / wsum
        if local_norm:
            df["score_weighted_100"] = _rescale_0_100(df["score_weighted_raw"])
        else:
            mn, mx = float(df["score_weighted_raw"].min()), float(df["score_weighted_raw"].max())
            df["score_weighted_100"] = 0.0 if mx <= mn else (df["score_weighted_raw"] - mn) * (100.0 / (mx - mn))
        return df

    # --- 계산 블록: 시작 시 1회 실행 + 입력 변화에 따라 재실행 ---
    @reactive.calc
    @reactive.event(input.recalc,
                    input.subgrid_n,
                    input.refine_thr_select,
                    input.rec_thr_select,
                    input.penalty_pub,
                    input.penalty_pri,
                    ignore_init=False)
    def map_and_scores():
        try:
            bd = base_data()
            cheonan_geom = bd["cheonan_geom"]

            # 요구사항 1: 대격자 타겟 개수 '고정 80'
            grid_gdf = make_uniform_grid_over_geom(
                cheonan_geom,
                target_cells=TARGET_GRID_CELLS,
                min_cell_m=300.0, max_cell_m=3000.0
            )

            # 집계
            grid_scores = aggregate_metrics_by_grid(
                grid_gdf,
                df_cat=bd["df_cat"],
                df_sensors=bd["df_sensors"],
                df_enf=bd["df_enf"],
                df_pub=bd["df_pub"], df_pri=bd["df_pri"]
            )

            # 가중치
            w_fac, w_trf, w_enf = float(input.w_fac()), float(input.w_trf()), float(input.w_enf())
            grid_scores = _apply_weights(grid_scores, w_fac, w_trf, w_enf, local_norm=False)

            # 요구사항 2/3: 소격자 라디오 + 기준점 Select
            # 항상 문자열→정수로 안전 변환 (None 방지)
            subgrid_raw = input.subgrid_n()            # "5" 또는 "10" 또는 None(초기 미정 방지용)
            n = _safe_input_int(input.subgrid_n, 10)   # "5" 또는 "10" → int

            thr_raw = input.refine_thr_select()        # "5" ..."50"
            refine_thr = float(_safe_input_int(input.refine_threshold_select, 5))



            hot = grid_scores.loc[grid_scores["score_weighted_100"] >= refine_thr].copy()

            if len(hot):
                subgrid = make_fixed_subgrid_over_polygons(
                    hot, sub_rows=n, sub_cols=n
                )
                sub_scores = aggregate_metrics_by_grid(
                    subgrid,
                    df_cat=bd["df_cat"],
                    df_sensors=bd["df_sensors"],
                    df_enf=bd["df_enf"],
                    df_pub=bd["df_pub"], df_pri=bd["df_pri"]
                )
                sub_scores = _apply_weights(sub_scores, w_fac, w_trf, w_enf, local_norm=bool(input.local_norm()))
            else:
                sub_scores = pd.DataFrame()

            # --- 기존 sub_scores 계산 직후에 [추가] ---
            # 1) 사용자 설정 읽기
            thr_raw = input.rec_thr_select()      # "100".."70"
            rec_thr = float(str(thr_raw or "90"))

            pen_pub_raw = input.penalty_pub()     # "5","10","15","20"
            penalty_pub = float(str(pen_pub_raw or "10"))

            pen_pri_raw = input.penalty_pri()     # "1".."10"
            penalty_pri = float(str(pen_pri_raw or "5"))

            # 2) 소격자가 있을 때만 조정점수 계산
            recommend_df = pd.DataFrame()
            if isinstance(sub_scores, pd.DataFrame) and len(sub_scores):
                df = sub_scores.copy()

                # 안전 변환
                for c in ["public_count", "private_count", "score_weighted_100"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

                # 조정 점수 = 가중치 100점 - (공영 수 × penalty_pub) - (민영 수 × penalty_pri)
                df["adjusted_score"] = (
                    df.get("score_weighted_100", 0)
                    - df.get("public_count", 0) * penalty_pub
                    - df.get("private_count", 0) * penalty_pri
                )

                # 추천 기준 필터
                rec = df.loc[df["adjusted_score"] >= rec_thr].copy()

                # 중심점 (이미 grid_scores에 centroid가 없을 수 있어 geometry에서 다시 산출)
                if "centroid_lat" not in rec.columns or "centroid_lon" not in rec.columns:
                    rec["centroid_lat"] = rec["geometry"].centroid.y
                    rec["centroid_lon"] = rec["geometry"].centroid.x

                # (선택) 주소 붙이기 (키 있으면)
                if KAKAO_KEY and len(rec):
                    # 너무 잦은 호출 방지: 행 기준 최대 300개 정도만
                    _limit = min(300, len(rec))
                    addrs = []
                    for i, r in enumerate(rec.itertuples(index=False)):
                        if i >= _limit:
                            addrs.append("")
                            continue
                        lat = float(getattr(r, "centroid_lat", float("nan")) or float("nan"))
                        lon = float(getattr(r, "centroid_lon", float("nan")) or float("nan"))
                        if np.isnan(lat) or np.isnan(lon):
                            addrs.append("")
                            continue
                        addrs.append(_reverse_geocode(lat, lon))
                        time.sleep(0.12)  # 아주 소량 딜레이
                    rec["address_hint"] = addrs
                else:
                    rec["address_hint"] = ""

                # 표로 쓸 최소 컬럼만 골라 정렬
                recommend_df = rec[[
                    "grid_id", "centroid_lat", "centroid_lon",
                    "score_weighted_100", "public_count", "private_count",
                    "adjusted_score", "address_hint"
                ]].sort_values("adjusted_score", ascending=False).reset_index(drop=True)


            # --- 지도 ---
            m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM, tiles=None)
            try:
                add_vworld_base_layers(m)  # 베이스(위성 기본 / 밝은 지도 선택 / 라벨 오버레이)
            except Exception:
                folium.TileLayer("OpenStreetMap", name="OSM 기본", show=True).add_to(m)

            # 미니맵
            m.add_child(MiniMap(position="bottomright", toggle_display=True))

            # ---- 레이어들 (모두 개별 FeatureGroup으로) ----
            # 경계 (천안시/동남/서북) — 기본 ON: 천안시 경계, 라벨은 add_vworld_base_layers에서 ON
            if bd.get("cheonan_gdf") is not None:
                # 컨트롤 항목으로는 "[경계] 천안시" 하나만 올리되, 표시선/헤일로는 control=False로 숨김
                fg_city = folium.FeatureGroup(name="[경계] 천안시", show=True)
                folium.GeoJson(
                    bd["cheonan_gdf"], name="[경계] 천안시(halo)",
                    style_function=lambda x: {"color": "#FFFFFF", "weight": 7, "opacity": 0.9},
                    control=False
                ).add_to(fg_city)
                folium.GeoJson(
                    bd["cheonan_gdf"], name="[경계] 천안시(선)",
                    style_function=lambda x: {"color": "#00E5FF", "weight": 3.5, "opacity": 1.0},
                    control=False
                ).add_to(fg_city)
                fg_city.add_to(m)

            gm = bd.get("gu_map") or {}
            fg_gu = folium.FeatureGroup(name="[경계] 자치구(동남/서북)", show=False)
            dn = gm.get("동남구"); sb = gm.get("서북구")
            if dn is not None and len(dn):
                folium.GeoJson(dn, name="[경계] 동남구",
                               style_function=lambda x: {"color": "#BA2FE5", "weight": 3, "opacity": 1.0}).add_to(fg_gu)
            if sb is not None and len(sb):
                folium.GeoJson(sb, name="[경계] 서북구",
                               style_function=lambda x: {"color": "#FF5722", "weight": 3, "opacity": 1.0}).add_to(fg_gu)
            fg_gu.add_to(m)

            # 단속 히트맵(OFF 기본)
            if len(bd["df_enf"]):
                add_enforcement_heatmap_layer(m, bd["df_enf"])

            # 주차장(공영/민영) (OFF 기본)
            add_parking_layers_to_map(m, bd["df_pub"], bd["df_pri"])

            # 교통 수집기 (OFF 기본)
            if len(bd["df_sensors"]):
                add_traffic_sensors_layer(m, bd["df_sensors"])

            # 주변시설 POI + 좌상단 범례(OFF 기본)
            if len(bd["df_cat"]):
                add_category_layers(m, bd["df_cat"])

            # 격자(대/소) (OFF 기본)
            if len(grid_scores):
                add_congestion_grid_layer(
                    m, grid_scores,
                    value_col="score_weighted_100",
                    layer_name="[격자-대] 가중치 점수(0~100)",
                    caption="격자(대) — 가중치 반영 0~100",
                    vmin=0, vmax=100
                )
            if len(sub_scores):
                add_congestion_grid_layer(
                    m, sub_scores,
                    value_col="score_weighted_100",
                    layer_name="[격자-소] 가중치 점수(0~100)",
                    caption="격자(소) — 가중치 반영 0~100",
                    vmin=0, vmax=100
                )

            # 레이어 컨트롤(우상단)
            _turn_off_all_overlays(m)  # 베이스 제외 전부 OFF

            # 베이스: 위성 True, 밝은 지도 False
            _select_basemap(m, prefer_names=("Satellite", "위성"))

            # 라벨(VWorld Hybrid) ON
            _set_layer_show_by_name(m, ["라벨 (VWorld Hybrid)", "라벨(VWorld Hybrid)", "Hybrid"], show=True)

            # [경계] 천안시 ON (자치구는 기본 OFF)
            _set_layer_show_by_name(m, ["[경계] 천안시"], show=True)

            # === [추천 하이라이트] 소격자 외곽선만 표시 (LayerControl 전에 추가) ===
            # 현재 입력값으로 다시 계산해 항상 최신 조건 반영
            thr = _safe_input_int(input.rec_thr_select, 90)  # 100/95/90/... 중 선택, 기본 90

            if isinstance(sub_scores, pd.DataFrame) and len(sub_scores) and "geometry" in sub_scores.columns:
                import json

                # 최신 임계값으로 필터링 (패널티 반영된 최종점수 사용)
                score_col = "score_final_100" if "score_final_100" in sub_scores.columns else "score_weighted_100"
                rec_gdf = sub_scores.loc[
                    pd.to_numeric(sub_scores[score_col], errors="coerce").fillna(-1) >= thr
                ].copy()

                if len(rec_gdf):
                    def _rec_style(_):
                        # 외곽선만, 채우기 없음
                        return {"color": "#00E5FF", "weight": 4, "opacity": 1.0, "fillOpacity": 0.0}

                    def _rec_hover(_):
                        return {"color": "#FFFFFF", "weight": 5, "opacity": 1.0, "fillOpacity": 0.0}
                    
                    fg_rec = folium.FeatureGroup(name="[격자-소] 추천 하이라이트", show=False)

                    # 표시용 점수 컬럼: score_final_100(있으면 그대로), 없으면 패널티 반영
                    disp_score_col = None
                    if "score_final_100" in rec_gdf.columns:
                        rec_gdf["disp_score"] = pd.to_numeric(rec_gdf["score_final_100"], errors="coerce")
                        disp_score_col = "disp_score"
                    else:
                        # score_weighted_100 에서 패널티 적용
                        _pen_pub = _safe_input_int(input.penalty_pub, 10)
                        _pen_pri = _safe_input_int(input.penalty_pri, 5)
                        rec_gdf["disp_score"] = (
                            pd.to_numeric(rec_gdf.get("score_weighted_100", 0), errors="coerce").fillna(0)
                            - pd.to_numeric(rec_gdf.get("public_count", 0), errors="coerce").fillna(0) * _pen_pub
                            - pd.to_numeric(rec_gdf.get("private_count", 0), errors="coerce").fillna(0) * _pen_pri
                        )
                        disp_score_col = "disp_score"

                    # 주소 힌트가 추천 표 쪽에 이미 있으면 매핑해서 붙임(없으면 공란)
                    try:
                        if isinstance(recommend_df, pd.DataFrame) and "grid_id" in recommend_df.columns:
                            addr_map = dict(zip(recommend_df["grid_id"], recommend_df.get("address_hint", "")))
                            rec_gdf["address_hint"] = rec_gdf["grid_id"].map(addr_map).fillna("")
                        else:
                            if "address_hint" not in rec_gdf.columns:
                                rec_gdf["address_hint"] = ""
                    except Exception:
                        if "address_hint" not in rec_gdf.columns:
                            rec_gdf["address_hint"] = ""

                    # 툴팁/팝업 필드 준비(없을 수도 있는 컬럼은 안전 처리)
                    for col in ["public_count", "private_count", "centroid_lat", "centroid_lon"]:
                        if col not in rec_gdf.columns:
                            rec_gdf[col] = np.nan

                    gj = folium.GeoJson(
                        data=json.loads(rec_gdf.to_json()),
                        name="[격자-소] 추천 하이라이트",
                        style_function=_rec_style,
                        highlight_function=_rec_hover,
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=["grid_id", disp_score_col, "public_count", "private_count"],
                            aliases=["격자", "조정점수", "공영수", "민영수"],
                            localize=True,
                            sticky=False
                        ),
                        popup=folium.features.GeoJsonPopup(
                            fields=["grid_id", "centroid_lat", "centroid_lon", "address_hint"],
                            aliases=["격자", "위도", "경도", "행정동(힌트)"],
                            labels=False,
                            localize=True
                        ),
                    )
                    gj.add_to(fg_rec)
                    fg_rec.add_to(m)


            # 추천 하이라이트만 다시 ON (초기 전체 OFF 뒤에 덮어씀)
            _set_layer_show_by_name(m, ["[격자-소] 추천 하이라이트"], show=True)


            folium.LayerControl(collapsed=False, position="topright").add_to(m)
            # 컨트롤 추가 직후
            _wire_enf_legend_behavior_by_name(
                m,
                overlay_name="[단속] 히트맵(불법주정차)",  # cheonan_mapping_core에서 만든 오버레이 이름과 정확히 일치
                legend_id="enf-legend"                      # 커스텀 범례 div id
            )

            return {"map": m, "grid_scores": grid_scores, "sub_scores": sub_scores, "recommend": recommend_df, "error": ""}

        except Exception:
            err = traceback.format_exc()
            print("[DEBUG] map_and_scores ERROR:\n", err)
            return {"map": None, "grid_scores": pd.DataFrame(), "sub_scores": pd.DataFrame(), "error": err}

    # --- 출력부 ---
    @output
    @render.ui
    def map_ui():
        data = map_and_scores()
        if data.get("error"):
            return ui.pre(data["error"])
        html = (data["map"]._repr_html_()
                if data.get("map") is not None
                else folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM)._repr_html_())
        return ui.div(ui.HTML(html), style="height: 78vh; min-height: 600px; border-radius: 8px; overflow: hidden;")


    @render.download(filename=lambda: f"cheonan_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")
    def dl_scores():
        import io, zipfile
        data = map_and_scores()
        g = data["grid_scores"]; s = data["sub_scores"]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if isinstance(g, pd.DataFrame) and len(g):
                g2 = g.drop(columns=["geometry"], errors="ignore")
                zf.writestr("grid_scores.csv", g2.to_csv(index=False, encoding="utf-8-sig"))
            if isinstance(s, pd.DataFrame) and len(s):
                s2 = s.drop(columns=["geometry"], errors="ignore")
                zf.writestr("subgrid_scores.csv", s2.to_csv(index=False, encoding="utf-8-sig"))
        buf.seek(0)
        return buf
    
    @output
    @render.table
    def recommend_table():
        data = map_and_scores()
        df = data.get("recommend", pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or not len(df):
            return pd.DataFrame({"안내": ["추천 기준을 만족하는 소격자가 없습니다."]})
        # 사용자 친화적 컬럼명으로 바꿔서 보여주기
        out = df.rename(columns={
            "grid_id": "격자",
            "centroid_lat": "위도",
            "centroid_lon": "경도",
            "score_weighted_100": "원점수(0~100)",
            "public_count": "공영수",
            "private_count": "민영수",
            "adjusted_score": "조정점수",
            "address_hint": "행정동(힌트)"
        }).copy()
        # 소수점 정리
        for c in ["위도","경도","원점수(0~100)","조정점수"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
        return out


app = App(app_ui, server)
