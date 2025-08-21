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

from shinywidgets import output_widget, render_widget

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---- 지도 레이어 제어 헬퍼들 ----
import folium

def _remove_existing_layer_controls(m: folium.Map):
    """지도에 이미 올라간 LayerControl이 있으면 제거(중복 방지)"""
    for k, ch in list(m._children.items()):
        if isinstance(ch, folium.map.LayerControl):
            del m._children[k]

def _debug_dump_layers(m: folium.Map, banner="[DEBUG]"):
    """현재 지도 레이어 목록을 콘솔에 출력(이름/overlay/show 상태 확인용)"""
    try:
        print(f"{banner} Layer list:")
        for ch in m._children.values():
            nm = getattr(ch, "layer_name", None)
            ov = getattr(ch, "overlay", None)
            sh = getattr(ch, "show", None)
            print(f" - {type(ch).__name__}: name={nm} overlay={ov} show={sh}")
    except Exception as e:
        print("[DEBUG] dump failed:", e)


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

def _select_basemap(m: folium.Map, prefer_names=("Imagery","Satellite","위성","항공","WorldImagery","Esri","VWorld","영상")):
    """
    베이스 후보: overlay=False 이거나, 이름상 'Base/Basic/Positron/Imagery/Satellite/위성/VWorld' 등인 타일/WMS.
    prefer_names에 해당하는 것을 1개만 ON, 나머지는 OFF.
    또한 'Hybrid/라벨'은 베이스와 별개로 항상 켭니다(라벨만 켜는 동작이 종종 overlay=True로 구현되기 때문).
    """
    bases = []
    labels = []

    def is_label_name(nm: str) -> bool:
        nm = (nm or "").lower()
        return any(k in nm for k in ("hybrid", "label", "라벨"))

    def is_base_name(nm: str) -> bool:
        nm = (nm or "").lower()
        return any(k in nm for k in (
            "imagery","satellite","위성","항공","worldimagery","esri","vworld","base","basic","positron","osm"
        ))

    for ch in list(m._children.values()):
        nm = getattr(ch, "layer_name", "") or ""
        ov = getattr(ch, "overlay", None)
        if hasattr(ch, "show"):
            if is_label_name(nm):
                labels.append(ch)
            # overlay가 False 이거나, 이름상 베이스로 보이면 베이스 후보에 포함
            if (ov is False) or is_base_name(nm):
                bases.append(ch)

    # 모든 베이스 OFF
    for t in bases:
        t.show = False

    # 선호 베이스 ON
    chosen = None
    for t in bases:
        nm = (getattr(t, "layer_name", "") or "").lower()
        if any(p.lower() in nm for p in prefer_names):
            t.show = True
            chosen = t
            break
    if chosen is None and bases:
        bases[0].show = True  # 마지막 안전장치

    # 라벨(Hybrid)은 항상 ON
    for lab in labels:
        lab.show = True


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

# ---- 주차장 현황 탭에서 쓰는 헬퍼들 ----
import geopandas as gpd
from branca.element import Element

def _inject_map_css(m):
    css = Element("""
    <style>
      .leaflet-control-container .leaflet-top.leaflet-right { right: 12px !important; left: auto !important; }
      .leaflet-control-layers { margin-top: 10px; }
      /* 컨트롤이 카드/iframe 등에 가려지지 않게 */
      .leaflet-control { z-index: 10000 !important; }
      .leaflet-top, .leaflet-bottom { z-index: 10000 !important; }
    </style>
    """)
    m.get_root().html.add_child(css)

SCORE_DOC_HTML = """
<div class="score-doc">
  <b>부족도 산정식</b>
  <div><code>부족도 = 표준화(인구밀도) − 표준화(면적당 공영주차장 수)</code></div>
  <div><code>표준화: 0 ~ 100점</code></div>
  <div><code>값이 클수록 <b>부족</b></code></div>
</div>
<style>
.score-doc{font-size:.92rem;background:#f7f9ff;border:1px solid #e4ecff;border-radius:8px;padding:8px 10px;margin-top:8px}
.score-doc code{background:#eff4ff;padding:1px 4px;border-radius:4px}
.score-doc small{color:#6b7280}
</style>
"""

def _as_iframe(m: folium.Map, height_css: str = "78vh"):
    """
    Folium Map을 외부 파일로 저장하지 않고,
    iframe src=data:text/html;base64,... 방식으로 '격리'하여 렌더.
    """
    import base64
    html = m.get_root().render()  # folium 전체 HTML (head+body 포함)
    b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return ui.HTML(
        f'<iframe sandbox="allow-scripts allow-same-origin" '
        f'src="data:text/html;base64,{b64}" '
        f'style="width:100%; height:{height_css}; min-height:600px; '
        f'border:0; border-radius:8px; display:block;"></iframe>'
    )

def _load_emd_cheonan(emd_path: Path, cheonan_geom):
    """읍·면·동 경계를 천안시 영역으로 필터링해 GeoDataFrame으로 반환(없으면 빈 gdf)."""
    if not Path(emd_path).exists():
        return gpd.GeoDataFrame(columns=["EMD_NAME", "geometry"], geometry="geometry", crs="EPSG:4326")
    try:
        gdf = gpd.read_file(emd_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        # 이름 컬럼 추정
        name_col = next((c for c in ["EMD_KOR_NM","EMD_NM","법정동명","adm_nm","emd_nm"] if c in gdf.columns), None)
        gdf["EMD_NAME"] = gdf[name_col] if name_col else gdf.index.astype(str)
        out = gdf[["EMD_NAME","geometry"]].copy()
        # 천안시 영역으로 제한
        if cheonan_geom is not None and len(out):
            out = out.loc[out.geometry.centroid.within(cheonan_geom)].copy()
        return out
    except Exception:
        return gpd.GeoDataFrame(columns=["EMD_NAME", "geometry"], geometry="geometry", crs="EPSG:4326")



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

# Folium 지도를 iframe으로 격리해서 서빙할 폴더
MAP_HTML_DIR = BASE_DIR / "__maps__"
os.makedirs(MAP_HTML_DIR, exist_ok=True)

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
EMD_SHP_PATH = DATA_DIR / "BND_ADM_DONG_PG" / "BND_ADM_DONG_PG.shp"
STATUS_CSV = DATA_DIR / "2023_충남현황.csv"


MAP_CENTER_LAT, MAP_CENTER_LON = 36.815, 127.147
MAP_ZOOM = 12
TARGET_GRID_CELLS = 80   # ★ 요구사항 1: 대격자 개수 고정(조절 기능 제거)

os.makedirs(SAVE_DIR, exist_ok=True)

# 부록1(배경) 탭에서 쓰는 자산/데이터 경로
ASSETS_PREFIX = "/assets"          # 현재 app.py가 있는 폴더를 /assets 로 서빙
DATA_DIR2 = BASE_DIR / "data"      # 시각화용 CSV 폴더 (전달받은 코드 기준)

# ── 부록2(설명) Appendix(설명) 표 데이터 ─────────────────────────────
df_vars = pd.DataFrame(
    [
        ["주변시설(수요)", "주변시설·문화시설·관광명소 등 수요 지표의 가중치", "33", "범위 0–100, 클수록 수요 영향 ↑"],
        ["주변시설-1 대형마트/백화점", "대형마트, 백화점", "", ""],
        ["주변시설-2 학교(대학)", "대학교, 대학원", "", ""],
        ["주변시설-3 문화시설", "도서관, 공연장, 미술관, 박물관 등", "", ""],
        ["주변시설-4 공공기관", "시청, 구청, 주민센터 등", "", ""],
        ["주변시설-5 병원", "종합병원, 대학병원, 재활병원 등", "", ""],
        ["교통량(유동)", "스마트교통 수집기 기반 유동 지표의 가중치", "33", "범위 0–100, 클수록 유동 영향 ↑"],
        ["불법주정차(단속)", "단속 히트맵 지표의 가중치", "34", "범위 0–100, 클수록 단속 영향 ↑"],
        ["대격자(소격자 분할 기준 점수)", "대격자 점수가 임계 이상인 칸만 소격자로 분할", "5", "단위: 0–100 (대격자 가중점수)"],
        ["소격자", "대격자 중 핫셀을 세분화할 때 쪼개는 격자 수", "10×10", "선택: 10×10(정밀) / 5×5(속도)"],
        ["공영 주차장", "소격자 내 공영 주차장 개수만큼 감점", "10", "선택: 0/5/10/15/20"],
        ["민영 주차장", "소격자 내 민영 주차장 개수만큼 감점", "5", "선택: 0–10"],
    ],
    columns=["변수명", "설명", "예시값", "비고"],
)

df_sources = pd.DataFrame(
    [
        ["천안_교차로_행정동_매핑", "천안시 교통정보센터", "⭕"],
        ["전국_시군구_차량등록대수", "KOSIS(국가통계포털)", "⭕"],
        ["스마트교차로_통계", "천안시 교통정보센터", "⭕"],
        ["천안시_공영주차장 정보", "공공데이터포털", "⭕"],
        ["천안시_민영주차장 정보", "네이버지도", "❌"],
        ["충청남도 천안시_불법주정차단속현황", "공공데이터포털", "⭕"],
        ["지도 API", "카카오맵", "❌"],
        ["충남_지역별_인구수_인구밀도", "행정안전부", "⭕"],
    ],
    columns=["공공 데이터명", "출처", "국가중점"],
)


# ── 부록1 시각화 보조 함수들 ─────────────────────────
def _read_csv_smart(path, encodings=("utf-8-sig", "cp949", "euc-kr")):
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def _to_short_sido(name: str) -> str:
    if not isinstance(name, str): return name
    name = name.strip()
    rep = {
        "강원특별자치도":"강원","세종특별자치시":"세종","전북특별자치도":"전북","제주특별자치도":"제주",
        "서울특별시":"서울","부산광역시":"부산","대구광역시":"대구","인천광역시":"인천","광주광역시":"광주",
        "대전광역시":"대전","울산광역시":"울산","경기도":"경기","강원도":"강원","충청북도":"충북",
        "충청남도":"충남","전라북도":"전북","전라남도":"전남","경상북도":"경북","경상남도":"경남",
    }
    return rep.get(name, name)

def _get_latest_numeric_col(df: pd.DataFrame, exclude_cols: list[str]) -> str:
    cand = [c for c in df.columns if c not in exclude_cols]
    for c in reversed(cand):
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return cand[-1] if cand else df.columns[-1]

def _parking_sido_counts(df_parking: pd.DataFrame) -> pd.DataFrame:
    col = "시도명" if "시도명" in df_parking.columns else ("CTPRVN_NM" if "CTPRVN_NM" in df_parking.columns else None)
    if col is None:
        raise ValueError("주차장 데이터에 '시도명' 또는 'CTPRVN_NM' 컬럼이 필요합니다.")
    out = df_parking[[col]].copy()
    out.rename(columns={col: "시도명"}, inplace=True)
    out["시도명"] = out["시도명"].astype(str).str.strip().replace({
        "강원특별자치도":"강원","세종특별자치시":"세종","전북특별자치도":"전북","제주특별자치도":"제주"
    }).map(_to_short_sido)
    g = out.groupby("시도명", as_index=False).size().rename(columns={"size":"공영주차장수"})
    return g


# =========================
# UI  (사이드바 제거, 한 장짜리 설정 카드로 통합)
# =========================
# =========================
# UI (탭 네비게이션으로 상단 구성)
# =========================
# =========================
# UI  (사이드바 제거, 상단 탭 구성)
# =========================
app_ui = ui.page_fluid(
    # --- 컴팩트 스타일 (레이블 줄바꿈 방지/여백 축소 + 추천표 가로 스크롤) ---
    ui.tags.style("""
    .compact-controls .card-body{padding:12px;}
    .compact-controls .shiny-input-container{margin-bottom:8px;}
    .compact-controls .form-label{margin-bottom:2px; white-space:nowrap;}
    .compact-controls .form-check{margin-bottom:0;}
    .compact-controls .form-select,
    .compact-controls .form-control{height:calc(1.6rem + 2px); padding:2px 6px; font-size:0.9rem;}
    .compact-controls .irs--shiny .irs-bar,
    .compact-controls .irs--shiny .irs-handle{transform:scale(0.9);}

    #recommend_table { overflow-x: auto !important; }
    #recommend_table table {
        table-layout: auto !important;
        width: max-content !important;
    }
    #recommend_table th, #recommend_table td {
        white-space: nowrap !important;
        word-break: keep-all !important;
    }
    """),

    # 제목
    ui.h2("천안시 주차장 추가 설치 후보지 추천"),

    # 상단 탭 네비게이션 (Main / 주차장 현황 / 부록1 / 부록2)
    ui.navset_tab(
        

        # ---- Main 탭: (기존 메인 화면 전부) ----
        ui.nav_panel(
            "Main",
            # 사용자 설정 카드
            ui.card(
                ui.card_header("사용자 가중치/설정"),
                ui.div({"class": "compact-controls"},
                    ui.row(
                        # 지표 가중치
                        ui.column(4,
                            ui.h6("지표 가중치"),
                            ui.input_slider("w_fac", "주변시설(수요)", min=0, max=100, value=33, step=1),
                            ui.input_slider("w_trf", "교통량(유동)", min=0, max=100, value=33, step=1),
                            ui.input_slider("w_enf", "불법주정차(단속)", min=0, max=100, value=34, step=1),
                        ),
                        # 격자 설정
                        ui.column(4,
                            ui.h6("격자 설정"),
                            ui.input_radio_buttons(
                                "subgrid_n", "소격자 개수",
                                choices={"10": "10x10", "5": "5x5"},
                                selected="10", inline=True
                            ),
                            ui.input_select(
                                "refine_thr_select", "소격자로 분할하는 기준 점수 (대격자)",
                                choices=[str(x) for x in range(5, 51, 5)], selected="5"
                            ),
                            ui.input_checkbox("local_norm", "소격자 내 0~100으로 혼잡도 점수 재표준화", value=True),
                        ),
                        # 추천(고혼잡) 설정 + [지도 재계산] 버튼
                        ui.column(4,
                            ui.h6("지역 추천 기준 설정"),
                            ui.row(
                                ui.column(6,
                                    ui.input_select(
                                        "rec_thr_select", "추천 기준(점수)",
                                        choices=["100", "95", "90", "85", "80", "75", "70", "65", "60", "55", "50"], selected="90"
                                    ),
                                ),
                                ui.column(6,
                                    ui.input_select(
                                        "penalty_pub", "공영 주차장 1개당 차감 점수",
                                        choices=["0", "5", "10", "15", "20"], selected="10"
                                    ),
                                ),
                            ),
                            ui.row(
                                ui.column(6,
                                    ui.input_select(
                                        "penalty_pri", "민영 주차장 1개당 차감 점수",
                                        choices=[str(x) for x in range(0, 11)], selected="5"
                                    ),
                                ),
                            ),
                            ui.div({"class": "mt-2"},
                                ui.input_action_button("recalc", "지도 재계산", class_="btn-primary w-100")
                            ),
                        ),
                    ),
                ),
            ),

            ui.br(),

            # 지도: 전체 폭(12)
            ui.card(
                ui.card_header("지도"),
                ui.output_ui("map_ui"),
                full_screen=True
            ),

            ui.br(),

            # 추천표: 지도 아래 전체 폭(12) + CSV 다운로드 버튼
            ui.card(
                ui.card_header(
                    ui.div({"class": "d-flex justify-content-between align-items-center"},
                        ui.span("추천 격자(조정점수 기준)"),
                        ui.download_button("dl_recommend", "CSV 파일 다운로드", class_="btn btn-outline-secondary btn-sm")
                    )
                ),
                ui.output_table("recommend_table")
            ),
        ),


        # ---- 주차장 현황 탭 ----
        ui.nav_panel(
            "주차장 현황",
            ui.card(
                ui.card_header("천안시 공영/민영 주차장 현황"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("천안시 지도"),
                        ui.output_ui("parking_map_ui"),
                        full_screen=True
                    ),
                    ui.card(
                        ui.card_header("천안시 공영주차장 부족도 Top 10"),
                        ui.div(
                            output_widget("cheonan_top10"),   # 우측 그래프
                            ui.HTML(SCORE_DOC_HTML),          # 설명
                            style="display:flex; flex-direction:column; gap:8px;"
                        ),
                        full_screen=True
                    ),
                    col_widths=(7, 5)
                )
            )
        ),


        # ---- 부록1(배경) 탭 ----
        ui.nav_panel(
            "부록1(배경)",
            # (부록1 전용) 여백/그리드 살짝 타이트하게
            ui.tags.style("""
              #appendix1 .card-body{padding:12px}
              #appendix1 .g-2{--bs-gutter-x:.5rem; --bs-gutter-y:.5rem;}
            """),
            ui.div({"id": "appendix1"},  # ← 범위를 부록 탭 내부로만 한정
                ui.card(
                    ui.card_header("천안시 주차난 관련 기사"),
                    ui.row(
                        ui.column(
                            6,
                            ui.card(
                                ui.card_header("기사 1"),
                                ui.img(
                                    src=f"{ASSETS_PREFIX}/cheonan_parking_1.png",
                                    alt="천안 주차 관련 기사 이미지 1",
                                    style="width:100%; height:420px; object-fit:contain; background:#f8f9fa; border-radius:12px;"
                                )
                            )
                        ),
                        ui.column(
                            6,
                            ui.card(
                                ui.card_header("기사 2"),
                                ui.img(
                                    src=f"{ASSETS_PREFIX}/cheonan_parking_2.png",
                                    alt="천안 주차 관련 기사 이미지 2",
                                    style="width:100%; height:420px; object-fit:contain; background:#f8f9fa; border-radius:12px;"
                                )
                            )
                        ),
                    ),
                ),
                ui.br(),
                ui.card(
                    ui.card_header("시각화"),
                    ui.row(
                        # 좌측(7): 2개 그래프 세로 스택
                        ui.column(
                            7,
                            ui.card(
                                ui.card_header("시도별 공영주차장 수 & 인구 대비 보급률"),
                                output_widget("fig_sido_supply"),
                            ),
                            ui.br(),
                            ui.card(
                                ui.card_header("충남 시군구별 인구 1만명당 공영주차장"),
                                output_widget("fig_chungnam_bar"),
                            ),
                            class_="g-2"
                        ),
                        # 우측(5): Top20 그래프(라벨 안 잘리게)
                        ui.column(
                            5,
                            ui.card(
                                ui.card_header("충남 공영주차장 부족도 Top 20 (읍면동 기준)"),
                                output_widget("fig_chungnam_top20"),
                            ),
                            class_="g-2"
                        ),
                        class_="g-2"
                    ),
                ),
            ),
        ),


        # ---- 부록2(설명) 탭 ----
        ui.nav_panel(
            "부록2(설명)",
            # 이 탭 안에서만 적용될 스타일(스코프: #appendix2)
            ui.tags.style("""
                #appendix2 .card { border-radius: 10px; }
                #appendix2 .card-header { font-weight: 600; }

                /* 공통 테이블 스타일 (탭 내부 스코프) */
                #appendix2 #vars_tbl table, 
                #appendix2 #data_tbl table { width: 100%; border-collapse: collapse; }
                #appendix2 #vars_tbl th, #appendix2 #vars_tbl td, 
                #appendix2 #data_tbl th, #appendix2 #data_tbl td {
                    border: 1px solid #e5e7eb; padding: 10px; vertical-align: middle;
                    white-space: pre-line; word-break: keep-all; font-size: 0.95rem;
                }
                #appendix2 #vars_tbl thead th, 
                #appendix2 #data_tbl thead th {
                    background: #f8fafc; position: sticky; top: 0; z-index: 1;
                    text-align: center;                 /* 헤더 가운데 정렬 */
                }

                /* 기본은 본문 좌측 정렬 */
                #appendix2 #vars_tbl td, 
                #appendix2 #data_tbl td { text-align: left; }

                /* '변수명' 컬럼(첫 번째 컬럼)만 가운데 정렬 + 강조 */
                #appendix2 #vars_tbl td:first-child {
                    text-align: center;                 /* 첫 번째 컬럼 가운데 정렬 */
                    font-weight: 600;
                }

                /* 데이터 설명 표의 마지막 컬럼(국가중점)만 가운데 정렬 */
                #appendix2 #data_tbl td:last-child, 
                #appendix2 #data_tbl th:last-child {
                    text-align: center; width: 90px;
                }
            """),
            ui.div({"id": "appendix2"},
                ui.h3("변수 정의 / 데이터 설명"),
                ui.row(
                    ui.column(
                        8,
                        ui.card(
                            ui.card_header("변수 정의"),
                            ui.output_table("vars_tbl")
                        )
                    ),
                    ui.column(
                        4,
                        ui.card(
                            ui.card_header("데이터 설명"),
                            ui.output_table("data_tbl")
                        )
                    ),
                )
            ),
        ),
    )
)




# =========================
# 서버 로직
# =========================
def server(input, output, session):
    # --- 첫 렌더링에 지도/표를 한 번 계산하기 위한 부트스트랩 트리거 ---
    boot_trigger = reactive.value(False)

    @reactive.effect
    def _kickoff_once():
        # 서버가 기동되고 첫 reactive 사이클에서 단 1회만 True로 전환
        if not boot_trigger.get():
            boot_trigger.set(True)

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
    
    def _safe_input_float(getter, default: float) -> float:
        try:
            v = getter()
            if v is None or str(v).strip() == "":
                return float(default)
            return float(v)
        except SilentException:
            return float(default)
        except Exception:
            return float(default)

    def _safe_input_bool(getter, default: bool) -> bool:
        try:
            v = getter()
            if isinstance(v, bool):
                return v
            if v in (None, ""):
                return bool(default)
            # 셀렉트/문자 케이스
            s = str(v).strip().lower()
            if s in ("true","1","yes","y","on"):
                return True
            if s in ("false","0","no","n","off"):
                return False
            return bool(default)
        except SilentException:
            return bool(default)
        except Exception:
            return bool(default)


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

    # --- 무거운 원본 데이터: 앱 기동 시 1회 로드 + 지도 재계산 버튼 눌렀을 시에만 가동 ---
    @reactive.calc
    @reactive.event(boot_trigger, ignore_init=False)
    def base_data():
        # 경계
        cheonan_gdf, cheonan_geom, gu_map = load_cheonan_boundary_shp(SHP_PATH)

        fast = True

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

        # ... base_data() 안, df_enf 처리 아래에 이어서 추가 ...
        # 읍·면·동 (없으면 빈 gdf)
        try:
            emd_gdf = _load_emd_cheonan(EMD_SHP_PATH, cheonan_geom)
        except Exception:
            emd_gdf = gpd.GeoDataFrame(columns=["EMD_NAME","geometry"], geometry="geometry", crs="EPSG:4326")

        return dict(
            cheonan_geom=cheonan_geom,
            cheonan_gdf=cheonan_gdf,
            gu_map=gu_map,
            df_cat=df_cat,
            df_pub=df_pub,
            df_pri=df_pri,
            df_sensors=df_sensors,
            df_enf=df_enf,
            emd_gdf=emd_gdf
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

    # --- 계산 블록: 시작 시 1회 실행 + 버튼 눌릴 때만 재실행 ---
    @reactive.calc
    @reactive.event(boot_trigger, input.recalc, ignore_init=False)   # ← 버튼만 이벤트
    def map_and_scores():
        try:
            bd = base_data()                           # base_data()는 그대로 사용

            # ▼▼ 버튼 외에는 반응하지 않도록 입력값 모두 isolate에서 읽습니다 ▼▼
            with reactive.isolate():
                w_fac = _safe_input_float(input.w_fac, 33.0)
                w_trf = _safe_input_float(input.w_trf, 33.0)
                w_enf = _safe_input_float(input.w_enf, 34.0)

                n = _safe_input_int(input.subgrid_n, 10)

                refine_thr = _safe_input_float(input.refine_thr_select, 5.0)

                local_norm = _safe_input_bool(input.local_norm, True)

                rec_thr     = _safe_input_int(input.rec_thr_select, 90)
                penalty_pub = _safe_input_int(input.penalty_pub, 10)
                penalty_pri = _safe_input_int(input.penalty_pri, 5)
            # ▲▲ 여기서 읽은 값들만 아래에서 사용합니다 ▲▲

            cheonan_geom = bd["cheonan_geom"]

            # 1) 대격자 생성(고정 80)
            grid_gdf = make_uniform_grid_over_geom(
                cheonan_geom, target_cells=TARGET_GRID_CELLS,
                min_cell_m=300.0, max_cell_m=3000.0
            )

            # 2) 대격자 집계 → 가중치
            grid_scores = aggregate_metrics_by_grid(
                grid_gdf,
                df_cat=bd["df_cat"], df_sensors=bd["df_sensors"], df_enf=bd["df_enf"],
                df_pub=bd["df_pub"], df_pri=bd["df_pri"]
            )
            grid_scores = _apply_weights(grid_scores, w_fac, w_trf, w_enf, local_norm=False)

            # 3) 소격자(핫셀만 분할)
            hot = grid_scores.loc[grid_scores["score_weighted_100"] >= refine_thr].copy()
            if len(hot):
                subgrid = make_fixed_subgrid_over_polygons(hot, sub_rows=n, sub_cols=n)
                sub_scores = aggregate_metrics_by_grid(
                    subgrid,
                    df_cat=bd["df_cat"], df_sensors=bd["df_sensors"], df_enf=bd["df_enf"],
                    df_pub=bd["df_pub"], df_pri=bd["df_pri"]
                )
                sub_scores = _apply_weights(sub_scores, w_fac, w_trf, w_enf, local_norm=local_norm)
            else:
                sub_scores = pd.DataFrame()

            # 4) 추천(조정점수=원점수-패널티) 테이블
            recommend_df = pd.DataFrame()
            if isinstance(sub_scores, pd.DataFrame) and len(sub_scores):
                df = sub_scores.copy()
                for c in ["public_count", "private_count", "score_weighted_100"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                df["adjusted_score"] = (
                    df.get("score_weighted_100", 0)
                    - df.get("public_count", 0) * penalty_pub
                    - df.get("private_count", 0) * penalty_pri
                )
                rec = df.loc[df["adjusted_score"] >= rec_thr].copy()
                if "centroid_lat" not in rec.columns or "centroid_lon" not in rec.columns:
                    rec["centroid_lat"] = rec["geometry"].centroid.y
                    rec["centroid_lon"] = rec["geometry"].centroid.x
                if KAKAO_KEY and len(rec):
                    import time
                    _limit = min(300, len(rec))
                    addrs = []
                    for i, r in enumerate(rec.itertuples(index=False)):
                        if i >= _limit: addrs.append(""); continue
                        lat = float(getattr(r, "centroid_lat", float("nan")) or float("nan"))
                        lon = float(getattr(r, "centroid_lon", float("nan")) or float("nan"))
                        addrs.append("" if (np.isnan(lat) or np.isnan(lon)) else _reverse_geocode(lat, lon))
                        time.sleep(0.12)
                    rec["address_hint"] = addrs
                else:
                    rec["address_hint"] = ""
                recommend_df = rec[[
                    "grid_id","centroid_lat","centroid_lon",
                    "score_weighted_100","public_count","private_count",
                    "adjusted_score","address_hint"
                ]].sort_values("adjusted_score", ascending=False).reset_index(drop=True)

            # 5) 지도 생성(생략 없음: 기존 코드 그대로)
            m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM, tiles=None)
            try:
                add_vworld_base_layers(m)
            except Exception:
                folium.TileLayer("OpenStreetMap", name="OSM 기본", show=True).add_to(m)
            m.add_child(MiniMap(position="bottomright", toggle_display=True))

            # ---- 레이어들 (모두 개별 FeatureGroup으로) ----
            # [경계] 천안시 — 컨트롤 항목 하나로 묶고 내부 선/헤일로는 control=False
            if bd.get("cheonan_gdf") is not None:
                fg_city = folium.FeatureGroup(name="[경계] 천안시", show=True)
                folium.GeoJson(
                    bd["cheonan_gdf"], name="[경계] 천안시(halo)",
                    style_function=lambda x: {"color": "#000000", "weight": 7, "opacity": 0.9},
                    control=False,
                ).add_to(fg_city)
                folium.GeoJson(
                    bd["cheonan_gdf"], name="[경계] 천안시(선)",
                    style_function=lambda x: {"color": "#FFFFFF", "weight": 3.5, "opacity": 1.0},
                    control=False,
                ).add_to(fg_city)
                fg_city.add_to(m)

            # [경계] 동남/서북
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

            # [단속] 히트맵(OFF 기본)
            if len(bd["df_enf"]):
                add_enforcement_heatmap_layer(m, bd["df_enf"])

            # [주차장] 공영/민영 (OFF 기본)
            add_parking_layers_to_map(m, bd["df_pub"], bd["df_pri"])

            # [교통 수집기] (OFF 기본)
            if len(bd["df_sensors"]):
                add_traffic_sensors_layer(m, bd["df_sensors"])

            # [주변시설] + 범례 (OFF 기본)
            if len(bd["df_cat"]):
                add_category_layers(m, bd["df_cat"])

            # [격자] 대/소 (OFF 기본)
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


            # === 추천 하이라이트 레이어 (표와 동기화) ===
            if (
                isinstance(sub_scores, pd.DataFrame) and len(sub_scores) and "geometry" in sub_scores.columns
                and isinstance(recommend_df, pd.DataFrame) and len(recommend_df)
            ):
                import json

                # 1) 표에 나온 추천 grid_id만 선택
                rec_ids = set(recommend_df["grid_id"].astype(str))
                rec_gdf = sub_scores.loc[sub_scores["grid_id"].astype(str).isin(rec_ids)].copy()

                # 2) 메타 결합
                meta_cols = ["grid_id", "adjusted_score", "address_hint", "centroid_lat", "centroid_lon"]
                rec_gdf = rec_gdf.merge(
                    recommend_df[meta_cols],
                    on="grid_id",
                    how="left",
                    suffixes=("","_rec")
                )
                rec_gdf["disp_score"] = pd.to_numeric(rec_gdf["adjusted_score"], errors="coerce")

                for col in ["public_count","private_count","centroid_lat","centroid_lon","address_hint"]:
                    if col not in rec_gdf.columns:
                        rec_gdf[col] = ""

                def _rec_style(_):
                    return {"color": "#00E5FF", "weight": 4, "opacity": 1.0, "fillOpacity": 0.0}
                def _rec_hover(_):
                    return {"color": "#FFFFFF", "weight": 5, "opacity": 1.0, "fillOpacity": 0.0}

                fg_rec = folium.FeatureGroup(name="[격자-소] 추천 하이라이트", show=False)
                gj = folium.GeoJson(
                    data=json.loads(rec_gdf.to_json()),
                    name="[격자-소] 추천 하이라이트",
                    style_function=_rec_style,
                    highlight_function=_rec_hover,
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=["grid_id", "disp_score", "public_count", "private_count"],
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
                    control=False,   # ★ 컨트롤러 중복 방지: 그룹만 컨트롤에 노출
                )
                gj.add_to(fg_rec)
                fg_rec.add_to(m)


            # ---- (하이라이트 FeatureGroup을 m에 add 한 직후부터) ----

            # ---- 기본 가시성 강제 (모든 레이어 add 끝난 뒤에 실행) ----

            # (A) 혹시 내부에서 LayerControl을 추가했으면 제거하고 새로 1회만 추가
            _remove_existing_layer_controls(m)

            # (B) 베이스 외 모든 오버레이 OFF
            _turn_off_all_overlays(m)

            # (C) 위성 베이스 강제 ON (여러 이름 대응)
            _select_basemap(m, prefer_names=("Imagery","Satellite","위성","항공","WorldImagery","Esri"))

            # (D) 기본 ON 레이어들
            _set_layer_show_by_name(m, ["Hybrid", "라벨"], show=True)            # 라벨
            _set_layer_show_by_name(m, ["[경계] 천안시"], show=True)             # 시 경계
            _set_layer_show_by_name(m, ["추천", "하이라이트"], show=True)        # 추천 하이라이트

            # (E) 반드시 OFF로 시작시킬 레이어들(겹침/혼란 방지)
            _set_layer_show_by_name(m, ["자치구", "동남구", "서북구"], show=False)        # 구 경계
            _set_layer_show_by_name(m, ["히트맵", "단속"], show=False)                    # 불법주정차 히트맵
            _set_layer_show_by_name(m, ["격자-대", "격자-소", "가중치"], show=False)      # 점수 컬러맵
            # 필요 시 주차장/센서도 OFF
            # _set_layer_show_by_name(m, ["[주차장] 공영","[주차장] 민영"], show=False)
            # _set_layer_show_by_name(m, ["수집기","센서","교통"], show=False)

            _inject_map_css(m)  # LayerControl 위치/여백 보정

            # (F) LayerControl은 마지막에 1회만
            folium.LayerControl(collapsed=False, position="topright").add_to(m)

            # (G) 디버그(콘솔에서 레이어 상태 확인)
            _debug_dump_layers(m, "[DEBUG MAIN]")


            # (옵션) 불법주정차 범례 토글 스크립트 연결
            _wire_enf_legend_behavior_by_name(
                m, overlay_name="[단속] 히트맵(불법주정차)", legend_id="enf-legend"
            )


            return {
                "map": m,
                "grid_scores": grid_scores,
                "sub_scores": sub_scores,
                "recommend": recommend_df,
                "error": ""
            }

        except Exception:
                    err = traceback.format_exc()
                    print("[DEBUG] map_and_scores ERROR:\n", err)
                    return {"map": None, "grid_scores": pd.DataFrame(), "sub_scores": pd.DataFrame(), "error": err}
        
    # -------------------------
    # 주차장 현황 탭 전용 지도
    # -------------------------
    @reactive.calc
    @reactive.event(boot_trigger, ignore_init=False)
    def build_parking_map():
        bd = base_data()  # 기존 로딩 재사용

        m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM, tiles=None)

        # 베이스맵: VWorld → 실패 시 OSM/Carto/Esri
        try:
            add_vworld_base_layers(m)  # 위성 + 라벨 (VWorld Hybrid)
            # 위성을 기본으로 ON
            folium.TileLayer(
                "CartoDB positron", 
                name="밝은 지도 (Carto Positron)", 
                show=False
            ).add_to(m)
        except Exception:
            folium.TileLayer(
                tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                name="OSM 기본", attr="&copy;OSM", show=False
            ).add_to(m)
            # 여기서 위성을 True로
            folium.TileLayer("Esri.WorldImagery", name="위성 (Esri)", show=True).add_to(m)
            folium.TileLayer("CartoDB positron", name="밝은 지도 (Carto Positron)", show=False).add_to(m)


        _inject_map_css(m)
        m.add_child(MiniMap(position="bottomright", toggle_display=True))

        # [경계] 천안시 (halo + 본선)
        try:
            if bd["cheonan_gdf"] is not None:
                folium.GeoJson(
                    bd["cheonan_gdf"],
                    name="[경계] 천안시 (halo)",
                    style_function=lambda x: {"color": "#FFFFFF", "weight": 7, "opacity": 0.9},
                    control=False,
                ).add_to(m)
                folium.GeoJson(
                    bd["cheonan_gdf"],
                    name="[경계] 천안시",
                    style_function=lambda x: {"color": "#00E5FF", "weight": 3.5, "opacity": 1.0},
                    highlight_function=lambda x: {"weight": 5, "color": "#FFFFFF"},
                ).add_to(m)
        except Exception as e:
            print("[WARN] draw city:", e)

        # [경계] 동남구/서북구
        try:
            gm = bd.get("gu_map") or {}
            if gm.get("동남구") is not None and len(gm["동남구"]):
                folium.GeoJson(
                    gm["동남구"], name="[경계] 동남구",
                    style_function=lambda x: {"color":"#BA2FE5","weight":3,"opacity":1.0},
                    highlight_function=lambda x: {"weight":4,"color":"#FFFFFF"},
                ).add_to(m)
            if gm.get("서북구") is not None and len(gm["서북구"]):
                folium.GeoJson(
                    gm["서북구"], name="[경계] 서북구",
                    style_function=lambda x: {"color":"#FF5722","weight":3,"opacity":1.0},
                    highlight_function=lambda x: {"weight":4,"color":"#FFFFFF"},
                ).add_to(m)
        except Exception as e:
            print("[WARN] draw gus:", e)

        # [경계] 읍·면·동
        try:
            emd = bd.get("emd_gdf")
            if emd is not None and len(emd):
                emd_group = folium.FeatureGroup(name="[경계] 읍·면·동", show=True)
                folium.GeoJson(
                    emd,
                    style_function=lambda x: {"color":"#FFFFFF","weight":6.5,"opacity":1.0,"fill":False},
                    control=False,
                ).add_to(emd_group)
                folium.GeoJson(
                    emd,
                    style_function=lambda x: {"color":"#222222","weight":3.0,"opacity":1.0,"fill":False},
                    highlight_function=lambda x: {"weight":3.6,"color":"#000000"},
                    tooltip=folium.GeoJsonTooltip(fields=["EMD_NAME"], aliases=["읍·면·동:"], sticky=True),
                    control=False,
                ).add_to(emd_group)
                emd_group.add_to(m)
        except Exception as e:
            print("[WARN] draw emd:", e)

        # [주차장] 공영(파랑)
        try:
            df_pub = bd["df_pub"]
            if isinstance(df_pub, pd.DataFrame) and len(df_pub):
                fg_pub = folium.FeatureGroup(name="[주차장] 공영 (파랑)", show=True)
                for _, r in df_pub.iterrows():
                    lat = float(r.get("lat", np.nan)); lon = float(r.get("lon", np.nan))
                    if np.isnan(lat) or np.isnan(lon): continue
                    name = str(r.get("name", r.get("주차장명", "공영주차장")))
                    folium.Marker(
                        [lat, lon], tooltip=name,
                        popup=folium.Popup(f"<b>{name}</b>", max_width=300),
                        icon=folium.Icon(color="blue", icon="car", prefix="fa")
                    ).add_to(fg_pub)
                fg_pub.add_to(m)
        except Exception as e:
            print("[WARN] draw pub:", e)

        # [주차장] 민영(빨강)
        try:
            df_pri = bd["df_pri"]
            if isinstance(df_pri, pd.DataFrame) and len(df_pri):
                fg_pri = folium.FeatureGroup(name="[주차장] 민영 (빨강)", show=True)
                for _, r in df_pri.iterrows():
                    lat = float(r.get("lat", np.nan)); lon = float(r.get("lon", np.nan))
                    if np.isnan(lat) or np.isnan(lon): continue
                    name = str(r.get("name", r.get("주차장명", "민영주차장")))
                    folium.Marker(
                        [lat, lon], tooltip=name,
                        popup=folium.Popup(f"<b>{name}</b>", max_width=300),
                        icon=folium.Icon(color="red", icon="car", prefix="fa")
                    ).add_to(fg_pri)
                fg_pri.add_to(m)
        except Exception as e:
            print("[WARN] draw pri:", e)

        # --- 베이스/레이어 가시성 강제: 위성을 기본으로, 라벨 ON ---
        _remove_existing_layer_controls(m)
        _turn_off_all_overlays(m)
        _select_basemap(m, prefer_names=("Imagery","Satellite","위성","항공","WorldImagery","Esri","VWorld","영상"))

        # 기본 ON
        _set_layer_show_by_name(m, ["Hybrid", "라벨"], show=True)
        _set_layer_show_by_name(m, ["[경계] 읍·면·동"], show=True)

        folium.LayerControl(collapsed=False, position="topright").add_to(m)
        _debug_dump_layers(m, "[DEBUG PARKING]")


        return m
    
    @reactive.calc
    def cheonan_top10_df():
        """
        STATUS_CSV에서 '부족도점수' 계산 후, 천안만 필터해 Top10 반환.
        부족도점수 = 표준화(인구밀도) - 표준화(면적당 공영주차장수)
        """
        # 파일 체크
        if not Path(STATUS_CSV).exists():
            return pd.DataFrame({"__error__":[f"현황 CSV가 없습니다: {STATUS_CSV}"]})

        # 로드(인코딩 폴백)
        df = None
        for enc in ("utf-8-sig","cp949","utf-8","euc-kr","latin1"):
            try:
                df = pd.read_csv(STATUS_CSV, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            return pd.DataFrame({"__error__":[f"CSV 읽기 실패: {STATUS_CSV}"]})

        # 필요한 컬럼
        need = {"시군구","읍면동","면적","인구수","인구밀도","공영주차장 수"}
        missing = sorted(list(need - set(df.columns)))
        if missing:
            return pd.DataFrame({"__error__":[f"필수 컬럼 누락: {', '.join(missing)}"]})

        # 계산
        df["면적당_주차장수"] = pd.to_numeric(df["공영주차장 수"], errors="coerce") / pd.to_numeric(df["면적"], errors="coerce")

        def _rescale_0_100_local(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            vmin, vmax = s.min(), s.max()
            if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
                return pd.Series([50.0] * len(s), index=s.index)
            return (s - vmin) / (vmax - vmin) * 100.0

        df["인구밀도_점수"] = _rescale_0_100_local(df["인구밀도"])
        df["면적당주차장_점수"] = _rescale_0_100_local(df["면적당_주차장수"])
        df["부족도점수"] = df["인구밀도_점수"] - df["면적당주차장_점수"]

        # 천안만 필터 → Top10
        cheonan = df[df["시군구"].astype(str).str.contains("천안", na=False)].copy()
        if cheonan.empty:
            return pd.DataFrame({"__error__":[f"천안 데이터가 없습니다. (시군구 예: 천안시 동남구/서북구)"]})

        cheonan["지역"] = cheonan["시군구"].astype(str) + " " + cheonan["읍면동"].astype(str)
        cheonan = cheonan.sort_values("부족도점수", ascending=False).head(10)
        cheonan = cheonan.sort_values("부족도점수", ascending=True).reset_index(drop=True)

        out_cols = ["시군구","읍면동","지역","인구수","인구밀도","공영주차장 수","면적당_주차장수","부족도점수"]
        return cheonan[out_cols]



    # --- 출력부 ---
    @output
    @render.ui
    def map_ui():
        data = map_and_scores()
        if data.get("error"):
            return ui.pre(data["error"])

        m = data.get("map")
        if m is None:
            m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM)

        # 인라인으로 바로 렌더 (data: iframe 금지)
        html = m._repr_html_()
        return ui.div(
            ui.HTML(html),
            style="height:78vh; min-height:600px; border-radius:8px; overflow:visible; position:relative;"
        )


    @output
    @render.ui
    def parking_map_ui():
        try:
            m = build_parking_map()
        except SilentException:
            # base_data() 준비 전엔 잠깐 로딩 메시지
            return ui.div("지도를 준비 중입니다…", 
                          style="padding:8px; color:#6b7280;")
        if m is None:
            return ui.div("지도를 불러오지 못했습니다.", 
                          style="padding:8px; color:#6b7280;")
    
        # iframe 대신 인라인으로 그대로 렌더 (Main 탭과 동일한 방식)
        html = m._repr_html_()
        return ui.div(
            ui.HTML(html),
            style="height:78vh; min-height:600px; border-radius:8px; "
                  "overflow:visible; position:relative;"
        )

    
    @output
    @render_widget
    def cheonan_top10():
        df10 = cheonan_top10_df()
    
        # 에러 메시지면 텍스트 안내
        if "__error__" in df10.columns:
            import plotly.graph_objects as go
            msg = df10["__error__"].iloc[0]
            fig = go.Figure()
            fig.add_annotation(text=f"⚠️ {msg}", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            fig.update_layout(height=320, margin=dict(l=20,r=20,t=20,b=20))
            return fig
    
        if df10 is None or len(df10) == 0:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(text="표시할 데이터가 없습니다.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            fig.update_layout(height=320, margin=dict(l=20,r=20,t=20,b=20))
            return fig
    
        fig = px.bar(
            df10,
            x="부족도점수",
            y="지역",
            orientation="h",
            hover_data=["시군구","읍면동","인구수","인구밀도","공영주차장 수","면적당_주차장수","부족도점수"],
            title=None,
            labels={"지역":"읍면동","부족도점수":"부족도 점수"}
        )
        fig.update_layout(
            height=650,
            margin=dict(l=150, r=30, t=10, b=30),
            xaxis_title="부족도 점수",
            yaxis_title="읍면동"
        )
        fig.update_traces(text=df10["부족도점수"].round(1), textposition="outside", cliponaxis=False)
        return fig
    



    # ── 부록1 Plotly 그래프 ①: 시도별 보급률 ─────────────────
    @output
    @render_widget
    def fig_sido_supply():
        try:
            pop_csv = DATA_DIR2 / "전국_시군구_성별인구수.csv"
            park_csv = DATA_DIR2 / "전국_시군구_공영주차장.csv"

            df_pop = _read_csv_smart(pop_csv, encodings=("cp949","utf-8-sig","euc-kr"))
            try:
                df_parking = _read_csv_smart(park_csv, encodings=("utf-8-sig","cp949"))
            except Exception:
                df_parking = pd.read_csv(park_csv)

            # 인구: 지역/최신 컬럼
            region_col = next((c for c in ["행정구역(시군구)","행정구역별(읍면동)","행정구역(시도)","시도명","지역","행정구역"] if c in df_pop.columns), df_pop.columns[0])
            latest_col = _get_latest_numeric_col(df_pop, exclude_cols=[region_col])

            pop = df_pop[[region_col, latest_col]].copy()
            pop.rename(columns={region_col:"지역", latest_col:"인구"}, inplace=True)
            pop = pop[~pop["지역"].astype(str).isin({"전국","합계","소계"})]
            pop = pop[~pop["지역"].astype(str).str.contains(" ", na=False)]
            pop["시도명"] = pop["지역"].map(_to_short_sido)
            pop["인구"] = pd.to_numeric(pop["인구"], errors="coerce")
            pop_sido = pop.groupby("시도명", as_index=False)["인구"].sum()

            parking_sido = _parking_sido_counts(df_parking)

            df_sido = pop_sido.merge(parking_sido, on="시도명", how="left")
            df_sido["공영주차장수"] = pd.to_numeric(df_sido["공영주차장수"], errors="coerce").fillna(0).astype(int)
            df_sido["보급률_10만명당"] = (df_sido["공영주차장수"] / df_sido["인구"]) * 100_000

            x_order = df_sido.sort_values("보급률_10만명당", ascending=False)["시도명"].tolist()
            df_plot = df_sido.set_index("시도명").loc[x_order].reset_index()
            mean_rate = df_plot["보급률_10만명당"].mean()
            highlight = "충남"
            bar_colors = ["#E45756" if s == highlight else "#4C78A8" for s in df_plot["시도명"]]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=df_plot["시도명"], y=df_plot["공영주차장수"],
                textposition="outside", marker=dict(color=bar_colors), name="공영주차장 수"
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=df_plot["시도명"], y=df_plot["보급률_10만명당"],
                mode="lines+markers", name="보급률(10만명당)"
            ), secondary_y=True)
            fig.add_trace(go.Scatter(
                x=df_plot["시도명"], y=[mean_rate]*len(df_plot),
                mode="lines", line=dict(dash="dash"), name="평균 보급률"
            ), secondary_y=True)
            fig.update_layout(
                title="시도별 공영주차장 수 & 인구 대비 보급률(10만명당)",
                xaxis=dict(categoryorder="array", categoryarray=x_order, tickangle=-45, tickfont=dict(size=10)),
                bargap=0.20, margin=dict(b=150)
            )
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"그래프 생성 중 오류: {e}", x=0.5, y=0.5, showarrow=False)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            fig.update_layout(height=400, title="시도별 공영주차장/보급률")
            return fig

    # ── 부록1 Plotly 그래프 ②: 충남 시군구별 ────────────────
    @output
    @render_widget
    def fig_chungnam_bar():
        try:
            path = DATA_DIR2 / "chungnam_metrics.csv"
            df = pd.read_csv(path, encoding="utf-8-sig")
            df["is_cheonan"] = df["시군구명"].astype(str).str.contains("천안")

            y_col = "인구1만명당_주차시설_r" if "인구1만명당_주차시설_r" in df.columns else (
                "인구1만명당_주차시설" if "인구1만명당_주차시설" in df.columns else None
            )
            if y_col is None:
                raise ValueError("필요 컬럼(인구1만명당_주차시설_r)이 없습니다.")

            df["label"] = np.where(df["is_cheonan"], df["시군구명"] + " ⭐", df["시군구명"])
            df_plot = df.sort_values(y_col, ascending=False).copy()
            colors = np.where(df_plot["is_cheonan"], "crimson", "steelblue")

            fig = px.bar(
                df_plot, x="label", y=y_col,
                title="충남 시군구별 인구 1만명당 공영주차장 시설 수",
                labels={"label":"시군구", y_col:"시설/1만명"},
            )
            fig.update_traces(marker_color=colors, hovertemplate="<b>%{x}</b><br>인구 1만명당: %{y}<extra></extra>")
            fig.update_layout(
                xaxis={"categoryorder":"array","categoryarray":list(df_plot["label"])},
                yaxis_title="시설/1만명", xaxis_title="시군구", title_x=0.5
            )
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"그래프 생성 중 오류: {e}", x=0.5, y=0.5, showarrow=False)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            fig.update_layout(height=400, title="충남 시군구별 인구 1만명당 공영주차장")
            return fig
        
    @output
    @render_widget
    def fig_chungnam_top20():
        try:
            path = DATA_DIR2 / "2023_충남현황.csv"
            df = pd.read_csv(path, encoding="utf-8-sig")

            # ── 지표 계산 ─────────────────────────────
            if "공영주차장 수" not in df.columns or "면적" not in df.columns:
                raise ValueError("필수 컬럼('공영주차장 수','면적')이 누락되었습니다.")
            df["면적당_주차장수"] = df["공영주차장 수"] / df["면적"]

            def rescale(s: pd.Series) -> pd.Series:
                s = pd.to_numeric(s, errors="coerce")
                rng = s.max() - s.min()
                return (s - s.min()) / rng * 100 if rng != 0 else pd.Series([50]*len(s), index=s.index)

            if "인구밀도" not in df.columns:
                raise ValueError("필수 컬럼('인구밀도')이 누락되었습니다.")

            df["인구밀도_점수"] = rescale(df["인구밀도"])
            df["면적당주차장_점수"] = rescale(df["면적당_주차장수"])
            df["부족도점수"] = df["인구밀도_점수"] - df["면적당주차장_점수"]

            # 표기 라벨
            if not {"시군구","읍면동"}.issubset(df.columns):
                raise ValueError("필수 컬럼('시군구','읍면동')이 누락되었습니다.")
            df["지역"] = df["시군구"].astype(str) + " " + df["읍면동"].astype(str)

            # ── Top 20 선택 및 정렬 ─────────────
            top20 = df.sort_values("부족도점수", ascending=False).head(20).copy()
            top20 = top20.sort_values("부족도점수", ascending=True)

            # 색상: 시군구에 '천안' 포함만 빨간색
            top20["is_cheonan"] = top20["시군구"].astype(str).str.contains("천안")
            colors = np.where(top20["is_cheonan"], "crimson", "steelblue")

            hover_cols_all = ["시군구","읍면동","인구수","인구밀도","공영주차장 수","면적당_주차장수","부족도점수"]
            hover_cols = [c for c in hover_cols_all if c in top20.columns]

            fig = px.bar(
                top20,
                x="부족도점수",
                y="지역",
                orientation="h",
                hover_data=hover_cols,
                title="충남 공영주차장 부족도 Top 20 (읍면동 기준)",
                labels={"지역":"읍면동", "부족도점수":"부족도 점수"}
            )

            # 오른쪽 라벨이 안 잘리도록: x축 여유 + 클리핑 해제
            xmax = float(pd.to_numeric(top20["부족도점수"], errors="coerce").max())
            pad = max(8.0, xmax * 0.15)  # 최소 8, 또는 최대값의 15%
            fig.update_xaxes(range=[0, xmax + pad])

            fig.update_traces(
                marker_color=colors,
                texttemplate="%{x:.1f}",
                textposition="outside",
                cliponaxis=False
            )

            fig.update_layout(
                xaxis_title="부족도 점수",
                yaxis_title="읍면동",
                height=900,
                margin=dict(l=160, r=220, t=70, b=40),
                yaxis=dict(tickfont=dict(size=12)),
                showlegend=False
            )
            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"그래프 생성 중 오류: {e}", x=0.5, y=0.5, showarrow=False)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            fig.update_layout(height=460, title="충남 공영주차장 부족도 Top 20")
            return fig
            
    # ── 부록2 데이터프레임 ────────────────
    @output
    @render.table
    def vars_tbl():
        return df_vars

    @output
    @render.table
    def data_tbl():
        return df_sources




    @render.download(filename=lambda: f"cheonan_recommend_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    def dl_recommend():
        data = map_and_scores()
        df = data.get("recommend", pd.DataFrame())

        if not isinstance(df, pd.DataFrame) or not len(df):
            # 추천이 없으면 안내 한 줄짜리 CSV
            import io
            buf = io.BytesIO()
            pd.DataFrame({"안내": ["추천 기준을 만족하는 소격자가 없습니다."]}).to_csv(
                buf, index=False, encoding="utf-8-sig"
            )
            buf.seek(0)
            return buf

        # 테이블과 동일한 사용자 친화적 컬럼명으로 저장
        out = df.rename(columns={
            "grid_id": "격자",
            "centroid_lat": "위도",
            "centroid_lon": "경도",
            "score_weighted_100": "원점수(0~100)",
            "public_count": "공영 주차장 수",
            "private_count": "민영 주차장 수",
            "adjusted_score": "조정점수",
            "address_hint": "행정동",
        }).copy()

        # '행정동' 한 줄 처리(테이블과 동일)
        if "행정동" in out.columns:
            out["행정동"] = (
                out["행정동"].astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)
                .str.replace(r"\s*,\s*", " · ", regex=True)
                .str.strip()
            )

        # 숫자 반올림(보기 좋게)
        for c in ["위도", "경도", "원점수(0~100)", "조정점수"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

        import io
        buf = io.BytesIO()
        out.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)
        return buf

    
    # --- (교체) 추천 표 출력 ---
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
            "public_count": "공영 주차장 수",
            "private_count": "민영 주차장 수",
            "adjusted_score": "조정점수",
            "address_hint": "행정동",
        }).copy()
    
        # '행정동'을 한 줄로 보이도록 정리
        if "행정동" in out.columns:
            out["행정동"] = (
                out["행정동"].astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)      # 줄바꿈 제거
                .str.replace(r"\s*,\s*", " · ", regex=True)    # 콤마 → 중점
                .str.strip()
            )
            # (선택) 완전한 no-wrap을 원하면 공백을 비분리 공백으로 바꿔주세요
            # out["행정동"] = out["행정동"].str.replace(" ", "\u00A0", regex=False)
    
        # 숫자 표시 정리
        for c in ["위도", "경도", "원점수(0~100)", "조정점수"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    
        return out



app = App(app_ui, server, static_assets={ASSETS_PREFIX: BASE_DIR})
