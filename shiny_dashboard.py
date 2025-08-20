# app.py — Cheonan Parking Need Dashboard (Python Shiny)
# 가중치 슬라이더로 격자/소격자 점수 재계산 + Folium 지도 실시간 갱신

import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
# 파일 맨 위 import 근처에 추가
from pathlib import Path

from shiny import App, ui, render, reactive, req

# === 여러분의 기존 함수들을 모듈로 가져옵니다 ===
from cheonan_mapping_core import (
    # 데이터/레이어 로딩
    load_cheonan_boundary_shp, load_public_parking, load_private_parking,
    load_traffic_sensors_exact, load_traffic_stats, load_enforcement_points,
    add_vworld_base_layers, add_category_layers, add_parking_layers_to_map,
    add_enforcement_heatmap_layer,
    # 격자/집계/표준화
    make_uniform_grid_over_geom, make_fixed_subgrid_over_polygons,
    aggregate_metrics_by_grid, add_congestion_grid_layer, _rescale_0_100,
    # 유틸
    _inside
)

import folium
from folium.plugins import MiniMap

# =========================
# 경로/파일 상수 (필요에 따라 수정)
# =========================

# === 여기를 기존 상수 정의 대신 사용 ===
BASE_DIR = Path(__file__).resolve().parent          # .../project2_shiny/dashboard
DATA_DIR = BASE_DIR / "cheonan_data"

SAVE_DIR = str(DATA_DIR)                            # "./cheonan_data" 대신
SHP_PATH = str(DATA_DIR / "N3A_G0100000" / "N3A_G0100000.shp")
PUBLIC_PARKING_CSV   = str(DATA_DIR / "천안도시공사_주차장 현황_20250716.csv")
PRIVATE_PARKING_XLSX = str(DATA_DIR / "충청남도_천안시_민영주차장정보.xlsx")
SENSORS_CSV          = str(DATA_DIR / "천안_교차로_행정동_정확매핑.csv")
TRAFFIC_STATS_CSV    = str(DATA_DIR / "스마트교차로_통계.csv")
ENFORCEMENT_CSV_23   = str(DATA_DIR / "천안시_단속장소_위도경도_23년.csv")
ENFORCEMENT_CSV_24   = str(DATA_DIR / "천안시_단속장소_위도경도_24년.csv")

MAP_CENTER_LAT, MAP_CENTER_LON = 36.815, 127.147
MAP_ZOOM = 12

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# UI
# =========================
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("지표 가중치"),
        ui.input_slider("w_fac", "주변시설(수요) 가중치", min=0, max=100, value=33, step=1),
        ui.input_slider("w_trf", "교통량(유동) 가중치", min=0, max=100, value=33, step=1),
        ui.input_slider("w_enf", "불법주정차(압력) 가중치", min=0, max=100, value=34, step=1),
        ui.hr(),
        ui.h4("격자 설정"),
        ui.input_slider("target_cells", "대격자 목표 개수(대략)", min=40, max=160, value=80, step=10),
        ui.input_slider("sub_rows", "소격자 행(격자당)", min=5, max=20, value=10, step=1),
        ui.input_slider("sub_cols", "소격자 열(격자당)", min=5, max=20, value=10, step=1),
        ui.input_slider("refine_threshold", "소격자 분할 기준(대격자 점수, 0~100)", min=0, max=50, value=5, step=1),
        ui.input_checkbox("local_norm", "소격자 점수는 소집단 내 0~100 재표준화", value=True),
        ui.hr(),
        ui.input_action_button("recalc", "지도 재계산", class_="btn-primary"),
        ui.hr(),
        ui.input_checkbox("fast_mode", "빠른 로드(지오코딩/대용량 건너뛰기)", value=True),
        ui.download_button("dl_scores", "대격자/소격자 점수 CSV 다운로드"),
        width=350
    ),
    # ---- 메인 영역(그냥 나열) ----
    ui.row(
        ui.column(6, ui.card(ui.card_header("요약"), ui.output_text("summary_text"))),
        ui.column(6, ui.card(ui.card_header("가중치"), ui.output_text("weights_text"))),
    ),
    ui.card(ui.card_header("지도"), ui.output_ui("map_ui")),
    title="천안시 공영주차장 후보지 대시보드 (가중치 조정)",
    fillable=True
)

# =========================
# 서버 로직
# =========================
def server(input, output, session):
    # --- 무거운 원본 데이터: 앱 기동 시 1회 로드 ---
    @reactive.calc
    def base_data():
        bd = {}
        cheonan_gdf, cheonan_geom, gu_map = load_cheonan_boundary_shp(SHP_PATH)

        # --- 빠른 모드: 지오코딩/대용량을 건너뛰어 즉시 렌더링 ---
        fast = bool(input.fast_mode())

        # 공영 주차장 (CSV는 lat/lon 있으니 OK)
        df_pub = load_public_parking(PUBLIC_PARKING_CSV)
        if len(df_pub):
            df_pub = df_pub[df_pub.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        # 민영 주차장: fast_mode면 '지오코딩 건너뛰기'
        if fast:
            # 엑셀에 위도/경도가 이미 있으면 그것만 사용, 없으면 아예 스킵
            try:
                dfm = pd.read_excel(PRIVATE_PARKING_XLSX)
                rename = {}
                if "위도" in dfm.columns: rename["위도"] = "lat"
                if "경도" in dfm.columns: rename["경도"] = "lon"
                if "소재지도로명주소" in dfm.columns: rename["소재지도로명주소"] = "road_address"
                if "소재지지번주소" in dfm.columns: rename["소재지지번주소"] = "jibun_address"
                if "주차장명" in dfm.columns: rename["주차장명"] = "name"
                dfm = dfm.rename(columns=rename)
                has_coords = {"lat","lon"}.issubset(dfm.columns)
                if has_coords:
                    df_pri = dfm[["name","lat","lon","road_address","jibun_address"]].copy()
                    df_pri["category"] = "민영주차장"; df_pri["source"] = "천안시/민영"; df_pri["id"] = "private_" + df_pri.index.astype(str)
                    df_pri["lat"] = pd.to_numeric(df_pri["lat"], errors="coerce")
                    df_pri["lon"] = pd.to_numeric(df_pri["lon"], errors="coerce")
                    df_pri = df_pri.dropna(subset=["lat","lon"])
                    df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)
                else:
                    # 좌표 없으면 빠른 모드에서는 민영주차장 레이어 생략
                    df_pri = pd.DataFrame(columns=["id","name","lat","lon","road_address","jibun_address","category","source"])
            except Exception:
                df_pri = pd.DataFrame(columns=["id","name","lat","lon","road_address","jibun_address","category","source"])
        else:
            # 기존(느린) 전체 지오코딩 경로
            df_pri = load_private_parking(PRIVATE_PARKING_XLSX)
            if len(df_pri):
                df_pri = df_pri[df_pri.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        # 교통 수집기 + 7월 통계
        df_sensors = load_traffic_sensors_exact(SENSORS_CSV)
        df_stats   = load_traffic_stats(TRAFFIC_STATS_CSV)
        if len(df_sensors) and len(df_stats):
            df_sensors = df_sensors.merge(df_stats, left_on="join_key", right_on="교차로명", how="left")

        # 불법주정차
        df_enf_23 = load_enforcement_points(ENFORCEMENT_CSV_23)
        df_enf_24 = load_enforcement_points(ENFORCEMENT_CSV_24)
        if len(df_enf_23): df_enf_23["year"] = 2023
        if len(df_enf_24): df_enf_24["year"] = 2024
        df_enf = pd.concat([df_enf_23, df_enf_24], ignore_index=True) if (len(df_enf_23) or len(df_enf_24)) else pd.DataFrame()
        if len(df_enf):
            df_enf = df_enf[df_enf.apply(lambda r: _inside(cheonan_geom, r["lon"], r["lat"]), axis=1)].reset_index(drop=True)

        return dict(
            cheonan_geom=cheonan_geom,
            df_pub=df_pub, df_pri=df_pri,
            df_sensors=df_sensors, df_enf=df_enf
        )


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

    # --- 계산 블록: 시작 시 1회 실행 + 버튼으로 재실행 ---
    @reactive.calc
#    @reactive.event(input.recalc, ignore_init=False)  # ✅ 앱 시작 시 1회 실행
    def map_and_scores():
        try:
            bd = base_data()  # ← 여기서도 터질 수 있음
            cheonan_geom = bd["cheonan_geom"]

            grid_gdf = make_uniform_grid_over_geom(
                cheonan_geom,
                target_cells=int(input.target_cells()),
                min_cell_m=300.0, max_cell_m=3000.0
            )

            grid_scores = aggregate_metrics_by_grid(
                grid_gdf,
                df_cat=None,
                df_sensors=bd["df_sensors"],
                df_enf=bd["df_enf"],
                df_pub=bd["df_pub"], df_pri=bd["df_pri"]
            )

            w_fac, w_trf, w_enf = float(input.w_fac()), float(input.w_trf()), float(input.w_enf())
            grid_scores = _apply_weights(grid_scores, w_fac, w_trf, w_enf, local_norm=False)

            refine_thr = float(input.refine_threshold())
            hot = grid_scores.loc[grid_scores["score_weighted_100"] >= refine_thr].copy()

            if len(hot):
                subgrid = make_fixed_subgrid_over_polygons(
                    hot, sub_rows=int(input.sub_rows()), sub_cols=int(input.sub_cols())
                )
                sub_scores = aggregate_metrics_by_grid(
                    subgrid,
                    df_cat=None,
                    df_sensors=bd["df_sensors"],
                    df_enf=bd["df_enf"],
                    df_pub=bd["df_pub"], df_pri=bd["df_pri"]
                )
                sub_scores = _apply_weights(sub_scores, w_fac, w_trf, w_enf, local_norm=bool(input.local_norm()))
            else:
                sub_scores = pd.DataFrame()

            # --- 지도 ---
            m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM, tiles=None)
            try:
                add_vworld_base_layers(m)  # 키 문제시 여기서만 실패할 수 있음
            except Exception as e:
                # 위성 타일 실패해도 기본 OSM 타일로 대체해 지도는 뜨게 함
                folium.TileLayer("OpenStreetMap", name="OSM 기본", show=True).add_to(m)

            m.add_child(MiniMap())

            if len(bd["df_enf"]):
                add_enforcement_heatmap_layer(m, bd["df_enf"])
            if len(bd["df_pub"]) or len(bd["df_pri"]):
                add_parking_layers_to_map(m, bd["df_pub"], bd["df_pri"])

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

            folium.LayerControl(collapsed=False).add_to(m)
            return {"map": m, "grid_scores": grid_scores, "sub_scores": sub_scores, "error": ""}

        except Exception:
            err = traceback.format_exc()
            # 콘솔 + UI 둘 다 노출
            print("[DEBUG] map_and_scores ERROR:\n", err)
            return {"map": None, "grid_scores": pd.DataFrame(), "sub_scores": pd.DataFrame(), "error": err}


    # --- 출력부 ---
    @output
    @render.ui
    def map_ui():
        data = map_and_scores()
        if data.get("error"):
            return ui.pre(data["error"])
        if data.get("map") is None:
            m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=MAP_ZOOM)
            return ui.HTML(m._repr_html_())
        return ui.HTML(data["map"]._repr_html_())   # ✅ 고정
    
    @output
    @render.text
    def summary_text():
        data = map_and_scores()
        if data.get("error"):
            return "계산 중 오류가 발생했습니다. 좌측 지도 영역의 에러 상세를 확인하세요."
        bd = base_data()
        pub_n = 0 if bd["df_pub"] is None else len(bd["df_pub"])
        pri_n = 0 if bd["df_pri"] is None else len(bd["df_pri"])
        enf_n = 0 if bd["df_enf"] is None else len(bd["df_enf"])
        sns_n = 0 if bd["df_sensors"] is None else len(bd["df_sensors"])
        g = data.get("grid_scores", pd.DataFrame()); s = data.get("sub_scores", pd.DataFrame())
        return (
            f"대격자 {len(g):,}개, 소격자 {len(s):,}개 (재분할 기준: {int(input.refine_threshold())}점) | "
            f"공영 {pub_n:,} / 민영 {pri_n:,} / 단속 {enf_n:,} / 수집기 {sns_n:,} | "
            f"빠른 로드={bool(input.fast_mode())}"
        )


    
    # ---- 가중치 텍스트 ----
    @output
    @render.text
    def weights_text():
        return f"시설 {int(input.w_fac())} / 교통 {int(input.w_trf())} / 단속 {int(input.w_enf())}"
    # ---- CSV 다운로드 ----
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

app = App(app_ui, server)
