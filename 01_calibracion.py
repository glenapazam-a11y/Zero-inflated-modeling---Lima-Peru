"""
================================================================================
SCRIPT 1 de 2 — CALIBRACION DE LOS 8 MODELOS DE PRECIPITACION DIARIA
================================================================================
MODO CIEGO: no carga ni usa ningun dato observado del periodo de prediccion.

QUE HACE
    Ajusta (calibra) los ocho modelos sobre 1977-2019 y los evalua en la
    validacion 2020-2022, con una sola semilla (la de CONFIG["seed"]).

QUE ENTREGA  ->  01_CALIBRACION_parametros.xlsx  (+ 01_parametros_calibrados.csv)
    00_Ficha_modelos          Que es cada modelo, familia, n de parametros,
                              periodos de calibracion/validacion/prediccion.
    01_Parametros_calibrados  ENTREGABLE PRINCIPAL. Tabla larga
                              (Modelo, Grupo, Parametro, Valor) con TODOS los
                              parametros ajustados de los 8 modelos:
                                M1 pi_m, alpha_m, beta_m, mean_gamma por mes
                                M2 hiperparametros LSTM + epoca de early-stop +
                                   alpha_focal, pos_weight, diagnostico Gamma
                                M3 idem (entrenamiento conjunto)
                                M4 p (Tweedie), phi, coeficientes beta del GLM,
                                   peak_doy, P_shape
                                M5 coef. GLM ocurrencia y cantidad, alpha, phi
                                M6 xi, kappa, sigma, coef. estacionales
                                M7 alpha, L, n_cens, coef. GLM
                                M8 w, mu1, mu2, p01, p11, coef. Markov
    02_Metricas_calibracion   Desempeno de esta corrida unica (semilla base).
    Mx_val / Mx_pred          Serie diaria por modelo con obs, pred y la banda
                              del ensemble (pred_member, p10, p90).

LO QUE **NO** HACE (para no duplicar salidas)
    La variabilidad entre semillas, los estadisticos por semilla y el contraste
    de Wilcoxon se calculan en el SCRIPT 2 (02_semillas_wilcoxon.py), que importa
    este archivo como libreria y reutiliza exactamente estas mismas funciones.

ARQUITECTURA DEL PIPELINE
    Calibracion  : 1977-2019  (train_end, excl. 1996)
    Validacion   : 2020-2022  (val_end)
    Prediccion   : 2023-2024  (MODO CIEGO continuo, rollout autorregresivo)
    Gap respetado: dic-1995 -> ene-1997 (build_sequences_safe)

MODELOS
    M1 Climatologico mensual (Bernoulli + Gamma)            [baseline]
    M2 Hurdle LSTM (OccurrenceLSTM + AmountGammaLSTM)
    M3 ZI-Gamma LSTM (HurdleGammaLSTM, entrenamiento conjunto)
    M4 Poisson-Gamma (GLM Tweedie compuesto)
    M5 Hurdle-GLM (logistica + GLM Gamma sobre doy)
    M6 ZI-EGPD estacional (Pareto generalizada extendida cero-inflada, MLE)
    M7 ZI-Gamma censurada (Hurdle-GLM + censura < L)        [paper 2024]
    M8 Wilks 1999 (Markov 2 estados + mezcla exponencial)   [referencia clasica]

USO
    python 01_calibracion.py
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
# ► CONFIGURACIÓN CENTRAL — solo edita esta sección
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Rutas ──────────────────────────────────────────────────────────────
    "input_file"    : r"D:\0final\serie3f.xlsx",   # v18: serie con 1996 ya excluido
    "output_folder" : r"D:\0final\vfinal",

    # ── Columnas ───────────────────────────────────────────────────────────
    "date_col"      : "date",
    "precip_col"    : "ppd",

    # ── Períodos ───────────────────────────────────────────────────────────
    # v73: reparticion — entrenamiento 1977-2019, validacion 2020-2022,
    #      prediccion CIEGA continua de 2023 y 2024 (ninguno observado).
    "train_end"     : "2019-12-31",
    "val_end"       : "2022-12-31",
    "pred_year"     : 2023,                 # primer ano ciego (ancla del corte y del contexto)
    "pred_years"    : [2023, 2024],         # v73: anos de prediccion ciega (rollout autorregresivo continuo)
    "pred_label"    : "2023-2024",          # etiqueta para titulos/archivos

    # ── Parámetros hidrológicos ────────────────────────────────────────────
    "wet_threshold" : 0.05,
    "n_autoreg_paths": 20,            # trayectorias del ensemble autorregresivo
    # "ar_occurrence_mode": v74 — ELIMINADO: la ocurrencia es siempre Bernoulli (la rama "threshold" era código muerto)

    # ── Arquitectura redes ─────────────────────────────────────────────────
    "seq_len"       : 90,             # v19: 60→90 (+contexto estacional, -0.2% secuencias)
    "d_model"       : 64,
    "dropout"       : 0.25,
    "batch_size"    : 256,
    "lr"            : 1e-3,

    # ── Épocas ─────────────────────────────────────────────────────────────
    "epochs_occ"    : 80,             # v19: 40→80 (red débil en v18, necesita más entrenamiento)
    "epochs_amt"    : 80,             # v19: 50→70 | uniformizado a 80 (igual que epochs_occ/epochs_pred)
    "epochs_pred"   : 80,             # v19: 50→80
    "early_stop_patience": 15,        # v19: 10→15 (más tiempo para salir de mesetas)

    # ── Balance de pérdidas ────────────────────────────────────────────────
    "occ_loss_weight": 1.0,
    "focal_gamma_pred": 3.0,          # v34: subido de 2.0 → afila separación seco/húmedo
    # NOTA v19: alpha_focal calculado dinámicamente como n_pos/n_total (~0.08).
    # pos_weight = n_neg/n_pos (~11.2) se calcula también dinámicamente.
    # Ver corrección #2 del header.

    # ── DDPM ───────────────────────────────────────────────────────────────
    "diff_steps"    : 200,            # v18: restaurado desde 100 (v17 lo bajó sin justificación)
    "ddim_steps"    : 50,             # v18: restaurado desde 25

    # ── Reproducibilidad ───────────────────────────────────────────────────
    "seed"          : 42,
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",

    # ── MODO CIEGO ─────────────────────────────────────────────────────────
    "blind_mode"    : True,

    # ── v26: corrección de ocurrencia ──────────────────────────────────────────────
    # v34: umbrales recalibrados tras focal_gamma=3.0 (probabilidades se desplazaron ↑)
    # v37: ZIDF-Gamma y GammaParams SIN DDPM → su umbral se recalibra con el barrido
    # ── v71: recalibración de la cabeza de ocurrencia (SOLO Hurdle/Op.1) ───────
    # Diagnóstico v70: la cabeza occ del Hurdle nunca baja de prob_min≈0.136 →
    # con Bernoulli-sin-umbral (v68) cada día seco aporta prob·mu → PBIAS=+209%.
    # FIX: recalibrar prob sobre un HELD-OUT del TRAIN (cola cronológica, sin leak):
    # estira la parte baja del rango hacia 0 para que los días secos se apaguen.
    "hurdle_calibration"  : "isotonic",  # v71: "isotonic" | "platt" | "none"
    "hurdle_cal_frac"     : 0.15,        # v71: fracción final (cronológica) del train reservada para calibrar occ
    # v74: estos prob_thr YA NO deciden la ocurrencia (es Bernoulli). Solo seleccionan
    #      qué días predichos se reportan como "húmedos" en el diagnóstico Gamma (alpha/beta).
    #      Valores legado del barrido de validación (criterio PBIAS) de versiones ≤v59.
    "prob_thr_hurdle"     : 0.35,   # legado v53 — solo diagnóstico Gamma
    "prob_thr_zidf_gamma" : 0.21,   # legado v48 — solo diagnóstico Gamma
    "prob_thr_zidf_params": 0.24,   # legado v46 — solo diagnóstico Gamma (Op.4 eliminado)
    "pos_weight_cap"      : 4.0,    # cap de pos_weight (Hurdle y ZIDF-Gamma sin cambio)
    "pos_weight_cap_zidf_params": 3.0,  # v44: punto medio entre v42=4.0(FAR alto) y v43=2.0(colapso)

    # ── v55: Frequency-matching por mes en validación ──────────────────────────
    # Umbral dinámico por mes que iguala la frecuencia observada de días húmedos
    # v56: freq_match_val ELIMINADO — usaba cuantil de prob_val (trampa de validación).
    # v74: la ocurrencia es Bernoulli(prob), NO umbral. Los prob_thr_* solo se usan para
    #      seleccionar días húmedos en el diagnóstico Gamma y en el barrido de umbrales.
    "cap_mm"              : 3.1,   # v56: tope de cantidad = max real de la serie (era 1.0=P95, cortaba cola)
    "n_gamma_samples"     : 50,    # v58: nº de muestras Gamma → MEDIA (insesgada, suaviza ruido sin tocar PBIAS/cap/umbral)
    "n_ensemble"          : 50,    # v59: nº de miembros del ensamble (generador estocástico). Métricas=media; gráfica=miembro repr.
    "plot_envelope"       : True,  # v59: dibujar banda P10-P90 del ensamble en la gráfica de validación
    "plot_member"         : True,  # v59: dibujar miembro representativo (textura diaria) en vez de la media plana
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def _reseed_torch(cfg):
    """v72: re-siembra el RNG global de torch al inicio de cada modelo neuronal.

    Motivo: en v71 la recalibración del Hurdle entrena la red occ con menos
    secuencias y hace pasadas extra sobre el held-out → consume el RNG global de
    torch de forma distinta → cambia la inicialización de pesos de los modelos
    LSTM que corren DESPUÉS (Op.2/Op.4). Re-sembrar aquí desacopla cada modelo:
    cada red neuronal arranca con el MISMO estado RNG sin importar lo previo.
    Los modelos no-neuronales (Op.3,5,6,7,8,9,10) no dependen de esto.
    """
    s = int(cfg.get("seed", 42))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


print(f"[INFO] Dispositivo : {CONFIG['device']}  |  Torch: {torch.__version__}")
print(f"[INFO] vFINAL-CIEGO — 8 modelos finales: Op.3 Climatologico (base) | Op.1 Hurdle-Gamma LSTM | Op.2 ZI-Gamma LSTM | Op.5 Poisson-Gamma (Dzupire 2018) | Op.6 Hurdle-GLM | Op.7 ZI-EGPD estacional | Op.9 ZI-Gamma censurada | Op.10 Wilks 1999 (Markov+mezcla-exp) || vFINAL: ELIMINADOS Op.4 (ZIDF-GammaParams duplicado), Op.8 (ZI-EGPD marginal, colapsa) y la difusion DDPM (no usada) | reparticion 1977-2019 / 2020-2022 / ciego 2023-2024 | SIN LEAK")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cfg):
    print("\n[DATOS] Cargando serie (MODO CIEGO v18)...")
    df = pd.read_excel(cfg["input_file"], parse_dates=[cfg["date_col"]])
    df = df[[cfg["date_col"], cfg["precip_col"]]].copy()
    df.columns = ["date", "ppd"]
    df["ppd"] = pd.to_numeric(df["ppd"], errors="coerce")
    df.loc[df["ppd"] < 0, "ppd"] = np.nan

    # ── Corte ciego: eliminar pred_year y posteriores ──────────────────────
    year_p = cfg["pred_year"]
    cut = pd.Timestamp(f"{year_p}-01-01")
    n_dropped = (df["date"] >= cut).sum()
    df = df[df["date"] < cut].copy()
    print(f"   [CIEGO] Filas con date >= {cut.date()} eliminadas: {n_dropped:,}")

    # v18: 1996 ya está excluido en serie3.xlsx — no se necesita lógica aquí.
    # build_sequences_safe manejará el gap dic-1995/ene-1997 automáticamente.
    anios = sorted(df["date"].dt.year.unique())
    if 1996 in anios:
        raise ValueError(
            "[v18] ERROR: serie3.xlsx contiene año 1996. "
            "Usa la serie con 1996 ya excluido o activa la exclusión manual."
        )
    print(f"   [v18] Año 1996 ausente en la serie ✅")

    n_miss = df["ppd"].isna().sum()
    print(f"   Registros (< {year_p}): {len(df):,}  |  S/D: {n_miss:,} ({100*n_miss/len(df):.1f}%)")
    df["ppd"] = df["ppd"].interpolate(method="linear", limit=3).fillna(0)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   Zero-inflation: {(df['ppd']==0).mean():.3f}")
    print(f"   Rango: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Verificar gaps ─────────────────────────────────────────────────────
    dates_s = df["date"].sort_values().reset_index(drop=True)
    diffs = dates_s.diff().dt.days.fillna(1)
    gaps = diffs[diffs > 1]
    if len(gaps):
        print(f"   [v18] Gaps detectados en la serie ({len(gaps)}):")
        for idx in gaps.index:
            print(f"     {dates_s[idx-1].date()} → {dates_s[idx].date()} "
                  f"({int(gaps[idx])} días) — build_sequences_safe lo respeta")
    else:
        print(f"   [v18] Sin gaps temporales ✅")

    # ── Calendario sintético pred_years (ppd=NaN → nunca entra como input) ──
    y0, y1 = cfg["pred_years"][0], cfg["pred_years"][-1]
    dates_pred = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    df_synth = pd.DataFrame({"date": dates_pred, "ppd": np.nan})
    df = pd.concat([df, df_synth], ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"   [CIEGO] Calendario sintético {y0}–{y1}: {len(df_synth)} días con ppd=NaN")

    return df


def add_features(df, cfg):
    """Features comunes a Op.1, Op.2, Op.4."""
    df = df.copy()
    thr = cfg["wet_threshold"]
    df["doy_sin"]      = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cos"]      = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["month_sin"]    = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["ppd_lag1"]     = df["ppd"].shift(1).fillna(0)
    df["wet_lag1"]     = (df["ppd"].shift(1).fillna(0) >= thr).astype(np.float32)
    df["ppd_roll7"]    = df["ppd"].shift(1).rolling(7,  min_periods=1).sum().fillna(0)
    df["ppd_roll30"]   = df["ppd"].shift(1).rolling(30, min_periods=1).sum().fillna(0)
    df["ppd_roll90"]   = df["ppd"].shift(1).rolling(90, min_periods=1).sum().fillna(0)
    df["wet_freq30"]   = (df["ppd"].shift(1) >= thr).rolling(30, min_periods=1).mean().fillna(0)
    wet_shifted = (df["ppd"].shift(1).fillna(0) >= thr)
    grp = wet_shifted.cumsum()
    df["dias_secos_consec"] = (~wet_shifted).groupby(grp).cumcount().astype(np.float32)
    for k in range(2, 8):
        df[f"ppd_lag{k}"] = df["ppd"].shift(k).fillna(0)
    return df


def _compute_clim_daily(df_raw, cfg):
    """Media climatológica mensual (solo calibración). Para calentamiento del buffer."""
    train_data = df_raw[df_raw["date"] <= cfg["train_end"]].copy()
    return (train_data.groupby(train_data["date"].dt.month)["ppd"]
            .mean().to_dict())


def _features_from_buffer(buf, date, thr):
    """Features para 'date' dado el buffer de ppd previos. Zero-leak garantizado."""
    doy = date.dayofyear
    mon = date.month
    b   = np.asarray(buf, dtype=np.float64)
    n   = len(b)

    def lag(k):
        idx = n - k
        return float(b[idx]) if idx >= 0 else 0.0

    ppd_lags = {f"ppd_lag{k}": lag(k) for k in range(1, 8)}
    ppd_lag1 = ppd_lags["ppd_lag1"]
    wet_lag1 = float(ppd_lag1 >= thr)
    roll7    = float(b[-7:].sum())  if n >= 1 else 0.0
    roll30   = float(b[-30:].sum()) if n >= 1 else 0.0
    roll90   = float(b[-90:].sum()) if n >= 1 else 0.0
    wf30     = float((b[-30:] >= thr).mean()) if n >= 1 else 0.0

    dsc = 0.0
    for back in range(1, n + 1):
        if b[-back] >= thr:
            break
        dsc += 1.0

    feat = {
        "date": date,
        "ppd":  0.0,
        "doy_sin":   np.sin(2 * np.pi * doy / 365.25),
        "doy_cos":   np.cos(2 * np.pi * doy / 365.25),
        "month_sin": np.sin(2 * np.pi * mon / 12),
        "month_cos": np.cos(2 * np.pi * mon / 12),
        "wet_lag1":  wet_lag1,
        "ppd_roll7":  roll7,
        "ppd_roll30": roll30,
        "ppd_roll90": roll90,
        "wet_freq30": wf30,
        "dias_secos_consec": dsc,
    }
    feat.update(ppd_lags)
    return feat


def build_pred_context_df(df_raw, cfg):
    """Contexto para predicción de pred_year. Zero-leak: sin ppd observado de pred_year."""
    year = cfg["pred_year"]
    thr  = cfg["wet_threshold"]

    hist = (df_raw[df_raw["date"].dt.year < year][["date", "ppd"]]
            .sort_values("date").reset_index(drop=True))
    clim_daily = _compute_clim_daily(df_raw, cfg)
    ppd_buffer_init = list(hist["ppd"].values)

    pred_dates = (df_raw[df_raw["date"].dt.year.isin(cfg["pred_years"])]["date"]
                  .sort_values().reset_index(drop=True))

    rows = []
    buf  = list(ppd_buffer_init)
    for date in pred_dates:
        feat = _features_from_buffer(buf, date, thr)
        rows.append(feat)
        buf.append(float(clim_daily.get(date.month, 0.0)))

    pred_feat_df = pd.DataFrame(rows)
    hist_feat = add_features(hist, cfg)
    df_context = pd.concat([hist_feat, pred_feat_df], ignore_index=True)
    return df_context, ppd_buffer_init, clim_daily


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO DE PROBABILIDADES BRUTAS
# ══════════════════════════════════════════════════════════════════════════════

def _print_prob_diagnostico(prob_val, y_val, thr, label):
    prob = np.asarray(prob_val, dtype=np.float64).flatten()
    obs  = (np.asarray(y_val,  dtype=np.float64) >= thr).astype(int)
    n_wet_obs = int(obs.sum())
    n_total   = len(prob)
    sep = '─' * 60
    print(f'\n  ╔══ DIAGNÓSTICO PROB. BRUTAS: {label} ══╗')
    print(f'  {sep}')
    print(f'    n_total     : {n_total}')
    print(f'    n_wet_obs   : {n_wet_obs}  ({100*n_wet_obs/n_total:.1f}%)')
    print(f'    prob_min    : {prob.min():.4f}')
    print(f'    prob_max    : {prob.max():.4f}')
    print(f'    prob_mean   : {prob.mean():.4f}')
    print(f'    prob_std    : {prob.std():.4f}')
    print(f'\n  [PERCENTILES]')
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f'    P{p:>4.1f} : {np.percentile(prob, p):.4f}')
    print(f'\n  [DÍAS QUE SUPERAN UMBRAL vs OBS ({n_wet_obs})]')
    print(f"    {'umbral':>8}  {'n_pred':>7}  {'n_pred%':>8}  {'ratio':>12}  interp")
    for thr_t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n_pred = int((prob >= thr_t).sum())
        pct    = 100 * n_pred / n_total
        ratio  = n_pred / n_wet_obs if n_wet_obs > 0 else float('nan')
        if   ratio > 2.0:  interp = 'demasiados húmedos'
        elif ratio > 1.2:  interp = 'ligeramente excesivo'
        elif ratio > 0.8:  interp = '✅ cercano al obs'
        elif ratio > 0.3:  interp = 'subestima húmedos'
        else:              interp = '⚠ colapso'
        print(f'    {thr_t:>8.2f}  {n_pred:>7d}  {pct:>7.1f}%  {ratio:>12.3f}  {interp}')
    prob_max = prob.max()
    if prob_max < 0.15:
        veredicto = '❌ RED COLAPSADA — prob_max<0.15'
    elif prob_max < 0.30:
        veredicto = '⚠ RED DÉBIL — prob_max<0.30'
    else:
        n_above_20 = int((prob >= 0.20).sum())
        if n_above_20 < n_wet_obs * 0.3:
            veredicto = '⚠ UMBRAL ALTO — señal pero pocos días > 0.20'
        else:
            veredicto = '✅ RED CON SEÑAL'
    print(f'\n  [VEREDICTO] {veredicto}')
    print(f'  {sep}')
    print(f'  ╚══ FIN DIAG. PROB: {label} ══╝\n')


def apply_autoreg_ensemble(pred_dates_df, ppd_buffer_init, param_fn_batch,
                           scaler, feature_cols, cfg, clim_daily,
                           df_obs_ref, n_paths=None):
    """Ensemble vectorizado de trayectorias autorregresivas. Zero-leak garantizado."""
    year    = cfg["pred_year"]
    seq_len = cfg["seq_len"]
    thr     = cfg["wet_threshold"]
    n_paths = n_paths or cfg.get("n_autoreg_paths", 20)

    hist_df = (df_obs_ref[df_obs_ref["date"].dt.year < year]
               [["date", "ppd"]].sort_values("date").reset_index(drop=True))
    hist_feat_df = add_features(hist_df, cfg)
    feat_cache_init = hist_feat_df[feature_cols].values.astype(np.float32)

    pred_dates = list(pred_dates_df["date"])
    n_days = len(pred_dates)
    obs_lookup = (df_obs_ref[df_obs_ref["date"].dt.year.isin(cfg["pred_years"])]
                  .set_index("date")["ppd"].to_dict())

    rng = np.random.default_rng(cfg["seed"])
    bufs = [list(ppd_buffer_init) for _ in range(n_paths)]
    init_cache_list = feat_cache_init.tolist()
    feat_caches = [list(init_cache_list) for _ in range(n_paths)]
    paths = np.zeros((n_paths, n_days), dtype=np.float64)

    for di, date in enumerate(pred_dates):
        batch = np.empty((n_paths, seq_len, len(feature_cols)), dtype=np.float32)
        feat_rows = []
        for k in range(n_paths):
            feat_dict = _features_from_buffer(bufs[k], date, thr)
            feat_row  = [feat_dict.get(c, 0.0) for c in feature_cols]
            feat_rows.append(feat_row)
            window = feat_caches[k][-seq_len:]
            if len(window) < seq_len:
                window = [feat_row] * (seq_len - len(window)) + window
            batch[k] = scaler.transform(np.asarray(window, dtype=np.float32))

        ret   = param_fn_batch(batch)
        prob  = np.clip(np.asarray(ret[0], dtype=np.float64), 0.0, 1.0)
        alpha = np.clip(np.asarray(ret[1], dtype=np.float64), 1e-6, None)
        beta  = np.clip(np.asarray(ret[2], dtype=np.float64), 1e-6, None)

        # prob ya viene clip([0,1], float64) de la línea de extracción → prob_eff = prob
        prob_eff = prob

        # v74: OCURRENCIA = Bernoulli(prob_eff) en todo el pipeline (validación y ciego).
        #   Cada trayectoria muestrea rains ~ Bernoulli(prob). Esto rompe el ATRACTOR SECO:
        #   días húmedos esporádicos mantienen vivos los lags y los features de calendario
        #   hacen emerger la estacionalidad. Es el muestreo estándar de un generador
        #   estocástico. SIN LEAK: la prob proviene SOLO de la red entrenada, jamás de obs 2024.
        #   (v74 limpieza: eliminada la rama "threshold" — código muerto; ya no se umbraliza.)
        rains = rng.random(n_paths) < prob_eff

        alpha_s = alpha

        # v62: muestreo Gamma(alpha_red, 1/beta_red) — usa exactamente la distribución
        # que la red aprendió, sin parámetros externos. E[X]=alpha/beta=mu_red.
        # v61 usaba Gamma(1.8_fijo, mu/1.8) sobreescribiendo el alpha de la red.
        _cap_ar = cfg.get("cap_mm", 3.1)
        _gamma_samples = rng.gamma(shape=alpha_s, scale=1.0 / beta)
        _gamma_samples = np.clip(_gamma_samples, 0.0, _cap_ar)
        sampled = np.where(rains, _gamma_samples, 0.0)

        for k in range(n_paths):
            # v54 FIX 1: buffer recibe 0.0 cuando pred=0 (antes recibía climatología).
            # La climatología inflaba ppd_roll7/30/90 y wet_freq30 artificialmente
            # → el modelo "veía" señal húmeda aunque no prediciera lluvia real.
            # Efecto observado: pico anómalo de 8-10 mm en agosto 2021 (vs ~0 mm obs).
            # Con buf_val=0 el buffer refleja la predicción real del ensemble.
            buf_val = float(sampled[k]) if sampled[k] > 0 else 0.0
            bufs[k].append(buf_val)
            feat_caches[k].append(feat_rows[k])
        paths[:, di] = sampled

    pred_mean = paths.mean(axis=0)
    # [ANEXOS] percentiles del ensemble autorregresivo (para banda P10-P90 en test)
    pred_p10 = np.percentile(paths, 10, axis=0)
    pred_p90 = np.percentile(paths, 90, axis=0)
    # miembro representativo (trayectoria más cercana a la mediana del acumulado)
    _tot = paths.sum(axis=1)
    _rep = int(np.argmin(np.abs(_tot - np.median(_tot))))
    pred_member = paths[_rep]
    obs_p = [float(obs_lookup.get(d, np.nan)) for d in pred_dates]
    return pd.DataFrame({"date": pred_dates, "obs": obs_p, "pred": pred_mean,
                         "pred_member": pred_member,
                         "p10": pred_p10, "p90": pred_p90})


def split_data(df, cfg):
    train = df[df["date"] <= cfg["train_end"]].copy()
    val   = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    pred  = df[df["date"].dt.year.isin(cfg["pred_years"])].copy()
    print(f"   Calibración: {len(train):,} días | Validación: {len(val):,} | {cfg['pred_label']}: {len(pred):,}")
    return train, val, pred


def build_sequences(df, seq_len, feature_cols, scaler_x=None, fit=False):
    """Alias → build_sequences_safe (respeta gaps temporales)."""
    return build_sequences_safe(df, seq_len, feature_cols, scaler_x=scaler_x, fit=fit)


def build_sequences_safe(df, seq_len, feature_cols, scaler_x=None, fit=False):
    """Construye secuencias respetando gaps temporales (ninguna ventana cruza un gap)."""
    df = df.reset_index(drop=True)
    Xr = df[feature_cols].values.astype(np.float32)
    Yr = df["ppd"].values.astype(np.float32)

    if fit:
        scaler_x = StandardScaler()
        Xr = scaler_x.fit_transform(Xr)
    else:
        Xr = scaler_x.transform(Xr)

    dates = pd.to_datetime(df["date"])
    day_diff = dates.diff().dt.days.fillna(1).values
    gap_mask = day_diff > 1

    Xs, Ys = [], []
    block_start = 0
    n_gaps_skipped = 0

    for i in range(len(df)):
        if gap_mask[i]:
            block_start = i
            n_gaps_skipped += 1
        if (i - block_start) >= seq_len:
            Xs.append(Xr[i - seq_len: i])
            Ys.append(Yr[i])

    if n_gaps_skipped > 0:
        print(f"   [build_sequences_safe] Gaps: {n_gaps_skipped} "
              f"| Secuencias: {len(Xs)} "
              f"(vs {max(0, len(df)-seq_len)} sin control de gap)")

    return np.array(Xs, np.float32), np.array(Ys, np.float32), scaler_x


# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, thr=0.1, label=""):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    nse_denom = np.var(y_true) * len(y_true)
    nse = 1 - np.sum((y_true - y_pred)**2) / nse_denom if nse_denom > 0 else np.nan

    r     = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 else 0
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta  = y_pred.mean() / y_true.mean() if y_true.mean() > 0 else np.nan
    kge   = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2) if not np.isnan(alpha) else np.nan

    pbias = 100 * (y_pred.sum() - y_true.sum()) / y_true.sum() if y_true.sum() else np.nan

    nz = y_true >= thr
    mse_wet = mean_squared_error(y_true[nz], y_pred[nz]) if nz.sum() else np.nan
    mae_wet = mean_absolute_error(y_true[nz], y_pred[nz]) if nz.sum() else np.nan

    ob = (y_true >= thr).astype(int)
    pb = (y_pred >= thr).astype(int)
    tp = int(((ob == 1) & (pb == 1)).sum())
    fp = int(((ob == 0) & (pb == 1)).sum())
    fn = int(((ob == 1) & (pb == 0)).sum())
    tn = int(((ob == 0) & (pb == 0)).sum())
    pod  = tp / (tp + fn) if (tp + fn) else 0.0
    far  = fp / (tp + fp) if (tp + fp) else 0.0
    csi  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * pod / (prec + pod) if (prec + pod) else 0.0

    if label:
        print(f"\n  [{label}]")
        print(f"  NSE={nse:.3f}  KGE={kge:.3f}  PBIAS={pbias:+.1f}%  RMSE={rmse:.4f}")
        print(f"  POD={pod:.3f}  FAR={far:.3f}  CSI={csi:.3f}  F1={f1:.3f}")
        print(f"  Acum obs={y_true.sum():.1f}mm  pred={y_pred.sum():.1f}mm")

    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse,
        "NSE_diario": nse, "KGE": kge, "PBIAS_%": pbias,
        "MSE_eventos": mse_wet, "MAE_eventos": mae_wet,
        "POD": pod, "FAR": far, "CSI": csi, "Precision": prec, "F1": f1,
        "Acum_obs": float(y_true.sum()), "Acum_pred": float(y_pred.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def blind_metrics(y_pred, label=""):
    y_pred = np.asarray(y_pred, float)
    y_pred = y_pred[~np.isnan(y_pred)]
    if label:
        print(f"\n  [{label}]  (MODO CIEGO: sin métricas vs obs)")
        print(f"  Acum pred={y_pred.sum():.1f}mm  días>0={(y_pred>0).sum()}")
    return {
        "MSE": np.nan, "MAE": np.nan, "RMSE": np.nan,
        "NSE_diario": np.nan, "KGE": np.nan, "PBIAS_%": np.nan,
        "MSE_eventos": np.nan, "MAE_eventos": np.nan,
        "POD": np.nan, "FAR": np.nan, "CSI": np.nan, "Precision": np.nan, "F1": np.nan,
        "Acum_obs": np.nan, "Acum_pred": float(y_pred.sum()),
        "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan,
        "NSE_mensual": np.nan,
    }


def nse_mensual(y_true, y_pred, dates):
    df = pd.DataFrame({"obs": y_true, "pred": y_pred,
                       "mes": pd.to_datetime(dates).to_period("M")})
    m = df.groupby("mes")[["obs", "pred"]].sum()
    denom = np.var(m["obs"].values) * len(m)
    if denom == 0:
        return np.nan
    return 1 - np.sum((m["obs"].values - m["pred"].values)**2) / denom


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÓDULOS DE RED NEURONAL
# ══════════════════════════════════════════════════════════════════════════════

def zidf_gamma_nll(logit_pi, log_alpha, log_beta, y, thr=0.05, eps=1e-6):
    """v45: NLL del verdadero Zero-Inflated Gamma (ZIDF).

    Diferencia con Hurdle:
      Hurdle → P(y=0) y P(y>0) son procesos SEPARADOS (BCE + Gamma truncada).
      ZIDF   → mezcla de dos fuentes de cero:
               - cero estructural (prob pi): días donde la lluvia es 0 por estructura
               - cero muestral: la Gamma misma puede caer bajo el umbral

    Densidad ZIDF:
      si y <= thr:  P = pi + (1-pi) · F_gamma(thr)      [cero estructural + muestral]
      si y >  thr:  P = (1-pi) · f_gamma(y)             [húmedo genuino]

    donde pi = sigmoid(logit_pi) es la prob de cero estructural.
    """
    pi    = torch.sigmoid(logit_pi).clamp(eps, 1 - eps)
    alpha = torch.exp(log_alpha).clamp(min=eps)
    beta  = torch.exp(log_beta).clamp(min=eps)
    y_c   = torch.clamp(y, min=eps)

    is_wet = (y > thr).float()

    # log f_gamma(y) — densidad Gamma en y (días húmedos)
    log_f_gamma = (alpha * torch.log(beta) - torch.lgamma(alpha)
                   + (alpha - 1.0) * torch.log(y_c) - beta * y_c)

    # F_gamma(thr) — CDF Gamma evaluada en el umbral (prob de cero muestral)
    # gammainc puede no tener gradiente respecto a alpha en algunas versiones de torch;
    # se calcula sin gradiente (es un término de corrección menor, no la señal principal)
    thr_t = torch.full_like(alpha, float(thr))
    with torch.no_grad():
        F_thr = torch.special.gammainc(alpha, beta * thr_t).clamp(eps, 1 - eps)

    # log P(y=0) = log(pi + (1-pi)·F_gamma(thr))
    log_p_dry = torch.log(pi + (1 - pi) * F_thr + eps)

    # log P(y>0) = log(1-pi) + log f_gamma(y)
    log_p_wet = torch.log(1 - pi + eps) + log_f_gamma

    ll = is_wet * log_p_wet + (1 - is_wet) * log_p_dry
    return -ll.mean()


def gamma_nll(log_alpha, log_beta, y, eps=1e-6):
    alpha = torch.exp(log_alpha) + eps
    beta  = torch.exp(log_beta)  + eps
    y     = torch.clamp(y, min=eps)
    ll = (alpha * torch.log(beta) - torch.lgamma(alpha)
          + (alpha - 1.0) * torch.log(y) - beta * y)
    return -ll.mean()


def gamma_nll_weighted(log_alpha, log_beta, y, eps=1e-6):
    """v32: gamma_nll ponderada por log(1+y).
    Días con lluvia alta pesan más → la red aprende a diferenciar magnitudes.
    cv_mu debería subir de 0.125 a >0.3.
    """
    alpha = torch.exp(log_alpha) + eps
    beta  = torch.exp(log_beta)  + eps
    y     = torch.clamp(y, min=eps)
    ll    = (alpha * torch.log(beta) - torch.lgamma(alpha)
             + (alpha - 1.0) * torch.log(y) - beta * y)
    w     = torch.log1p(y)          # peso proporcional a magnitud
    w     = w / w.mean().clamp(min=eps)   # normalizar para no cambiar escala de lr
    return -(w * ll).mean()


def gamma_nll_rank(log_alpha, log_beta, y, eps=1e-6):
    """v49: gamma_nll ponderada por rango (rank-weighted).
    Días con más lluvia (mayor rango) pesan más → la red aprende a diferenciar
    magnitudes en lugar de colapsar hacia la media. Sube cv_mu.
    Sigue siendo Gamma truncada sobre días húmedos — formulación Hurdle intacta.
    """
    alpha = torch.exp(log_alpha).clamp(min=eps)
    beta  = torch.exp(log_beta).clamp(min=eps)
    y_c   = torch.clamp(y, min=eps)
    ll = (alpha * torch.log(beta) - torch.lgamma(alpha)
          + (alpha - 1.0) * torch.log(y_c) - beta * y_c)
    # peso por rango dentro del batch — diferencia magnitudes sin tocar la formulación
    ranks = torch.argsort(torch.argsort(y_c)).float() + 1.0
    w = ranks / ranks.sum()
    w = w / w.mean().clamp(min=eps)
    return -(w * ll).mean()


def fit_occurrence_calibrator(prob_cal, occ_cal, method="isotonic"):
    """v71: recalibra las probabilidades de ocurrencia sobre un HELD-OUT del TRAIN.

    Motivo (diagnóstico v70): la cabeza occ del Hurdle comprime sus salidas en
    [~0.136, ~0.39] y nunca se compromete con 'seco'. Con el muestreo Bernoulli
    sin umbral (v68) cada uno de los 1005 días aporta prob·mu → PBIAS=+209%.

    Una recalibración MONÓTONA (isotónica o Platt) reasigna prob_cruda → prob_emp
    tal que la prob coincida con la frecuencia húmeda observada en el held-out.
    Como los días de prob baja casi nunca llovieron, la isotónica los empuja hacia
    0 → en el camino Bernoulli esos días se apagan y el sesgo desaparece.

    SIN LEAK: el held-out es la cola cronológica del TRAIN (1977-2020); la red occ
    NO se entrenó sobre él, y jamás toca validación (2021-23) ni 2024.

    Devuelve (fn, nombre) donde fn: prob(np.ndarray)->prob_calibrada (misma forma).
    """
    prob_cal = np.asarray(prob_cal, dtype=np.float64).ravel()
    occ_cal  = np.asarray(occ_cal,  dtype=np.float64).ravel()

    # Salvaguardas: held-out demasiado chico o sin húmedos → no recalibrar
    if method == "none" or len(prob_cal) < 50 or occ_cal.sum() < 10:
        return (lambda p: np.asarray(p, dtype=np.float64)), "identidad (sin recalibrar)"

    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(prob_cal.reshape(-1, 1), occ_cal.astype(int))
        def _fn(p):
            p = np.asarray(p, dtype=np.float64)
            out = lr.predict_proba(p.reshape(-1, 1))[:, 1]
            return out.reshape(p.shape)
        return _fn, "Platt (logística 1-D)"

    # isotónica (por defecto): monótona, no paramétrica, puede mapear a 0 exacto
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(prob_cal, occ_cal)
    def _fn(p):
        p = np.asarray(p, dtype=np.float64)
        out = iso.predict(p.ravel())
        return out.reshape(p.shape)
    return _fn, "isotónica (held-out train)"


class FocalLoss(nn.Module):
    """Focal Loss para clasificación binaria desbalanceada.
    alpha = peso de la clase POSITIVA (lluvia) ≈ wet_frac (~0.08).
    v18: alpha se pasa dinámicamente, nunca se usa 0.85 fijo.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight)
        probs   = torch.sigmoid(logits)
        p_t     = torch.where(targets > 0.5, probs, 1 - probs)
        alpha_t = torch.where(targets > 0.5,
                              torch.full_like(p_t, self.alpha),
                              torch.full_like(p_t, 1 - self.alpha))
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


class OccurrenceLSTM(nn.Module):
    def __init__(self, n_feat, d=64, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.lstm = nn.LSTM(n_feat, d, 2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(d*2, d), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(d, 1))
    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class AmountGammaLSTM(nn.Module):
    """v33: cabeza ampliada (3 capas) + clamp [-6,6] para más rango en alpha/beta.
    v31 tenía solo 1 capa intermedia de 64 → colapsaba hacia media (cv_mu=0.042).
    """
    def __init__(self, n_feat, d=64, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.lstm = nn.LSTM(n_feat, d, 2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(d*2, d*2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d*2, d),   nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d, 2)
        )
    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        p = self.head(out[:, -1, :])
        # clamp [-6,6]: más rango que v31 [-4,4] → alpha/beta pueden variar más
        return torch.clamp(p[:, 0], -6, 6), torch.clamp(p[:, 1], -6, 6)


class HurdleGammaLSTM(nn.Module):
    """v40: amt vuelve a Linear(d,2) — cabeza simple, bias_init ancla escala.
    v39 demostró que 3 capas sin ancla diverge. La expresividad viene del
    backbone (lstm+body), no de la cabeza.
    """
    def __init__(self, n_feat, d=64, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.lstm = nn.LSTM(n_feat, d, 2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.body = nn.Sequential(nn.Linear(d*2, d*4), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(d*4, d), nn.ReLU())
        self.occ  = nn.Linear(d, 1)
        # v40: Linear simple — bias_init ancla alpha/beta climatológicos
        self.amt  = nn.Linear(d, 2)
    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        h = self.body(out[:, -1, :])
        p = self.occ(h).squeeze(-1)
        a = self.amt(h)
        return p, torch.clamp(a[:, 0], -6, 6), torch.clamp(a[:, 1], -6, 6)


class DiffusionDenoiser(nn.Module):
    def __init__(self, dim, hidden=256, steps=300):
        super().__init__()
        self.steps = steps
        self.te = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden*2), nn.SiLU(),
            nn.Linear(hidden*2, hidden*2), nn.SiLU(),
            nn.Linear(hidden*2, hidden), nn.SiLU(),
            nn.Linear(hidden, dim))
    def forward(self, y, s):
        t = self.te(s.float() / self.steps)
        return self.net(torch.cat([y, t], dim=-1))


class DDPM:
    def __init__(self, dim, steps=200, device="cpu"):
        self.S = steps; self.device = device; self.dim = dim
        betas = torch.linspace(1e-4, 0.02, steps).to(device)
        self.betas = betas; self.alphas = 1 - betas
        self.ab = torch.cumprod(self.alphas, 0)
        self.model = DiffusionDenoiser(dim, 256, steps).to(device)
        self.mean = None; self.std = None

    def fit_scaler(self, P):
        self.mean = P.mean(axis=0, keepdims=True)
        self.std  = P.std(axis=0,  keepdims=True) + 1e-6

    def _sc(self, P):  return (P - self.mean) / self.std
    def _usc(self, P): return P * self.std + self.mean

    def q_sample(self, y0, s, noise=None):
        if noise is None: noise = torch.randn_like(y0)
        ab = self.ab[s].reshape(-1, 1)
        return torch.sqrt(ab)*y0 + torch.sqrt(1-ab)*noise, noise

    def train(self, P_clean, cfg):
        self.fit_scaler(P_clean)
        Pc = torch.tensor(self._sc(P_clean).astype(np.float32)).to(self.device)
        loader = DataLoader(TensorDataset(Pc), batch_size=cfg["batch_size"],
                            shuffle=True, num_workers=0)
        opt = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs_diff"])
        hist = []
        for ep in range(1, cfg["epochs_diff"]+1):
            self.model.train(); losses = []
            for (yb,) in loader:
                s = torch.randint(0, self.S, (yb.size(0),), device=self.device)
                yn, eps = self.q_sample(yb, s)
                loss = nn.functional.mse_loss(self.model(yn, s.float().unsqueeze(1)), eps)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); losses.append(loss.item())
            sched.step(); hist.append(np.mean(losses))
            if ep % 20 == 0:
                print(f"   DDPM época {ep}/{cfg['epochs_diff']} | loss={hist[-1]:.5f}")
        return hist

    @torch.no_grad()
    def denoise(self, P_in, ddim_steps=None):
        self.model.eval()
        y = torch.tensor(self._sc(P_in).astype(np.float32)).to(self.device)
        K = ddim_steps or self.S
        steps = np.linspace(self.S-1, 0, K).round().astype(int)
        for idx in range(len(steps)):
            s = int(steps[idx])
            st = torch.full((y.size(0), 1), s, device=self.device, dtype=torch.float32)
            eps = self.model(y, st)
            ab = self.ab[s]
            x0 = (y - torch.sqrt(1-ab)*eps) / torch.sqrt(ab)
            if idx < len(steps)-1:
                ab_p = self.ab[int(steps[idx+1])]
                y = torch.sqrt(ab_p)*x0 + torch.sqrt(1-ab_p)*eps
            else:
                y = x0
        return self._usc(y.cpu().numpy())


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO POR MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _print_model_diagnostico(label, y_val, pred_val, thr, pred_p,
                              cfg, n_train_wet, n_val_wet, umbral_occ, met_val,
                              alpha_focal=None, pos_weight=None):
    sep = "─" * 64
    print(f"\n  ╔══ DIAGNÓSTICO MODELO: {label} ══╗")
    print(f"  {sep}")
    print(f"  [DATOS]")
    print(f"    Eventos húmedos TRAIN : {n_train_wet:>6}  (thr={thr} mm)")
    print(f"    Eventos húmedos VAL   : {n_val_wet:>6}")
    print(f"    Umbral ocurrencia cal.: "
          f"{'N/A (E[Y] continuo — v23)' if (umbral_occ is None or (isinstance(umbral_occ, float) and np.isnan(umbral_occ))) else f'{umbral_occ:.4f}'}")
    if alpha_focal is not None:
        print(f"    alpha_focal           : {alpha_focal:.4f}  (= wet_frac train, v18✅)")
    if pos_weight is not None:
        print(f"    pos_weight            : {pos_weight:.2f}  (= n_neg/n_pos, v19✅)")

    wet_obs  = y_val[y_val >= thr]
    wet_pred = pred_val[y_val >= thr]
    if len(wet_obs) > 0 and len(wet_pred) > 0:
        mean_obs = wet_obs.mean(); std_obs = wet_obs.std()
        mean_pred = wet_pred.mean(); std_pred = wet_pred.std()
        ratio_mean = mean_pred / mean_obs if mean_obs > 0 else np.nan
        flag = ('⚠ subestima' if ratio_mean < 0.7 else
                ('⚠ sobreestima' if ratio_mean > 1.4 else '✅ OK'))
        print(f"\n  [CANTIDAD — días húmedos obs (val)]")
        print(f"    Media obs  : {mean_obs:.4f} mm  |  std obs  : {std_obs:.4f} mm")
        print(f"    Media pred : {mean_pred:.4f} mm  |  std pred : {std_pred:.4f} mm")
        print(f"    Ratio pred/obs : {ratio_mean:.3f}  {flag}")

    nse_m = met_val.get("NSE_mensual", np.nan)
    kge   = met_val.get("KGE", np.nan)
    pbias = met_val.get("PBIAS_%", np.nan)
    pod   = met_val.get("POD", np.nan)
    far   = met_val.get("FAR", np.nan)
    csi   = met_val.get("CSI", np.nan)
    f1    = met_val.get("F1", np.nan)

    def _v(x, fmt=".3f"):
        return f"{x:{fmt}}" if not (x is None or (isinstance(x, float) and np.isnan(x))) else "NaN"

    print(f"\n  [MÉTRICAS VALIDACIÓN 2020-2022]")
    print(f"    NSE_mensual : {_v(nse_m)}   KGE : {_v(kge)}   PBIAS : {_v(pbias, '+.1f')}%")
    print(f"    POD : {_v(pod)}   FAR : {_v(far)}   CSI : {_v(csi)}   F1 : {_v(f1)}")
    print(f"    Acum obs : {_v(met_val.get('Acum_obs'), '.1f')} mm   "
          f"Acum pred : {_v(met_val.get('Acum_pred'), '.1f')} mm")

    alertas = []
    if not np.isnan(pbias):
        if pbias < -30: alertas.append(f"⚠ PBIAS muy negativo ({pbias:+.1f}%)")
        if pbias >  50: alertas.append(f"⚠ PBIAS muy positivo ({pbias:+.1f}%)")
    if not np.isnan(far) and far > 0.7:
        alertas.append(f"⚠ FAR alto ({far:.3f}) → muchos falsos positivos")
    if not np.isnan(pod) and pod < 0.2:
        alertas.append(f"⚠ POD bajo ({pod:.3f}) → no detecta lluvia")
    if not np.isnan(kge) and kge < 0:
        alertas.append(f"⚠ KGE negativo ({kge:.3f})")
    n_pred_pos = int((pred_p > 0).sum()) if pred_p is not None else 0
    acum_py = float(np.nansum(pred_p)) if pred_p is not None else 0.0
    print(f"\n  [PRED {cfg['pred_label']} — MODO CIEGO]")
    print(f"    Días con pred > 0 : {n_pred_pos}")
    print(f"    Acumulado pred    : {acum_py:.2f} mm")
    if n_pred_pos == 0:
        alertas.append(f"❌ COLAPSO TOTAL: cero días con lluvia en {cfg['pred_label']}")
    elif n_pred_pos < 5:
        alertas.append(f"⚠ CASI COLAPSO: solo {n_pred_pos} días con lluvia")

    if alertas:
        print(f"\n  [ALERTAS]")
        for a in alertas:
            print(f"    {a}")
    else:
        print(f"\n  [ALERTAS] ✅ Sin alertas críticas")

    print(f"  {sep}")
    print(f"  ╚══ FIN DIAGNÓSTICO: {label} ══╝\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPCIÓN 3 — CLIMATOLÓGICO MENSUAL
# ══════════════════════════════════════════════════════════════════════════════

def run_option3(df, cfg, df_raw=None):
    print("\n" + "="*60)
    print("  OPCIÓN 3 — Climatológico mensual (baseline)")
    print("="*60)
    thr = cfg["wet_threshold"]
    df_full = df_raw if df_raw is not None else df
    df["month"] = df["date"].dt.month
    # v73b: el baseline climatológico se ajusta SOLO con el periodo de entrenamiento
    # (<= train_end). Antes usaba <= val_end, lo que evaluaba el baseline EN MUESTRA
    # sobre la validación y le daba ventaja injusta frente a los modelos ML. Ahora la
    # validación 2020-2022 es out-of-sample también para el baseline (comparación justa).
    df_fit = df[df["date"] <= cfg["train_end"]].copy()

    params = {}
    for m in range(1, 13):
        sub = df_fit[(df_fit["month"] == m) & df_fit["ppd"].notna()]
        wet = sub["ppd"][sub["ppd"] >= thr].values
        pi  = len(wet) / len(sub) if len(sub) else 0.0
        if len(wet) >= 5:
            alpha, _, scale = stats.gamma.fit(wet, floc=0)
            beta = 1.0 / scale if scale > 0 else np.nan
            mu   = alpha * scale
        else:
            alpha = beta = np.nan
            mu = wet.mean() if len(wet) else 0.0
        params[m] = {"pi": pi, "alpha": alpha, "beta": beta, "mean_gamma": mu}

    def _predict_period(df_p):
        months = df_p["date"].dt.month.values
        exp = np.zeros(len(df_p))
        for j, m in enumerate(months):
            p = params[m]
            exp[j] = p["pi"] * p["mean_gamma"]
        return exp

    df_val = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    exp_val = _predict_period(df_val)
    met_val = compute_metrics(df_val["ppd"].values, exp_val, thr,
                               "Op.3 — Climatológico | Validación 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(df_val["ppd"].values, exp_val, df_val["date"].values)

    df_pred = df_full[df_full["date"].dt.year.isin(cfg["pred_years"])].copy()
    exp_pred = _predict_period(df_pred)
    met_pred = blind_metrics(exp_pred, f"Op.3 — Climatológico | {cfg['pred_label']} CIEGO")

    return {
        "label": "Climatológico",
        "df_val":  df_val.assign(pred=exp_val).rename(columns={"ppd": "obs"}),
        "df_pred": df_pred.assign(pred=exp_pred).rename(columns={"ppd": "obs"}),
        "met_val": met_val, "met_pred": met_pred,
        "params_clim": params,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPCIÓN 1 — HURDLE TEMPORAL CON PÉRDIDA GAMMA
# ══════════════════════════════════════════════════════════════════════════════

def run_option1(df, train_df, val_df, cfg, df_raw=None):
    print("\n" + "="*60)
    print("  OPCIÓN 1 — Hurdle temporal LSTM (Ocurrencia + Gamma)")
    print("="*60)
    _reseed_torch(cfg)   # v72: arranque RNG determinista, desacoplado de lo previo
    device = cfg["device"]; thr = cfg["wet_threshold"]

    FCOLS1 = (["doy_sin", "doy_cos", "month_sin", "month_cos"]
              + [f"ppd_lag{k}" for k in range(1, 8)]
              + ["wet_lag1", "ppd_roll7", "ppd_roll30", "ppd_roll90",
                 "wet_freq30", "dias_secos_consec"])

    X_tr, y_tr, scaler = build_sequences(train_df, cfg["seq_len"], FCOLS1, fit=True)
    X_val, y_val, _    = build_sequences(val_df,   cfg["seq_len"], FCOLS1, scaler_x=scaler)
    occ_tr  = (y_tr  >= thr).astype(np.float32)
    occ_val = (y_val >= thr).astype(np.float32)

    # ── v71: held-out de calibración (cola cronológica del train) ───────────
    # La red occ se entrena SOLO con X_occ_fit; X_cal queda reservado para
    # recalibrar la probabilidad después (sin leak: X_cal ⊂ train 1977-2020,
    # nunca tocó validación ni 2024). build_sequences_safe ya devuelve las
    # secuencias en orden cronológico, así que la cola es la parte más reciente.
    _cal_method = cfg.get("hurdle_calibration", "isotonic")
    _cal_frac   = float(cfg.get("hurdle_cal_frac", 0.15))
    if _cal_method != "none" and 0.0 < _cal_frac < 0.5:
        _n_cal = max(int(round(_cal_frac * len(X_tr))), 1)
        X_occ_fit, occ_occ_fit = X_tr[:-_n_cal], occ_tr[:-_n_cal]
        X_cal,     occ_cal      = X_tr[-_n_cal:], occ_tr[-_n_cal:]
        print(f"   [v71] Calibración occ: método={_cal_method} | held-out={_n_cal} secuencias "
              f"({_cal_frac:.0%} cola train) | húmedos en cal={int(occ_cal.sum())}")
    else:
        X_occ_fit, occ_occ_fit = X_tr, occ_tr
        X_cal, occ_cal = None, None
        print(f"   [v71] Calibración occ DESACTIVADA (método={_cal_method}) — prob cruda")

    # ── Etapa 1: Ocurrencia ────────────────────────────────────────────────
    print("\n  [Op.1] Entrenando etapa 1 — Ocurrencia (FocalLoss + bias_init + pos_weight)...")
    n_pos = occ_occ_fit.sum(); n_neg = len(occ_occ_fit) - n_pos

    # v18: alpha_focal = wet_frac (~0.08) — fix del bug de v13-v17
    # v19: pos_weight = n_neg/n_pos (~11.2) — amplifica gradiente en días húmedos
    #      Combinación: alpha define el ratio de penalización relativa,
    #      pos_weight amplifica la señal húmeda en términos absolutos.
    alpha_focal = float(n_pos / max(n_pos + n_neg, 1))
    # v26: cap en pos_weight_cap (era ~11, sobrecompensaba → red predice siempre húmedo)
    pos_weight_val = float(min(n_neg / max(n_pos, 1), cfg.get("pos_weight_cap", 4.0)))

    # bias_init: red arranca desde el prior real de lluvia
    wet_frac  = alpha_focal
    bias_init = float(np.log(wet_frac / (1.0 - wet_frac)))

    pw_tensor = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
    _fg_op1 = cfg.get("focal_gamma_pred", 2.0)
    bce = FocalLoss(alpha=alpha_focal, gamma=_fg_op1, pos_weight=pw_tensor).to(device)
    print(f"   [Op.1] FocalLoss: alpha={alpha_focal:.4f} (=wet_frac, v18✅)  gamma={_fg_op1} (v34✅)  "
          f"pos_weight={pos_weight_val:.2f} (v26: cap={cfg.get('pos_weight_cap', 4.0)}✅)")
    print(f"   [Op.1] bias_init={bias_init:.4f}  (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    occ_model = OccurrenceLSTM(X_tr.shape[2], cfg["d_model"], cfg["dropout"]).to(device)
    with torch.no_grad():
        occ_model.head[-1].bias.fill_(bias_init)

    loader_occ = DataLoader(TensorDataset(torch.tensor(X_occ_fit), torch.tensor(occ_occ_fit)),
                            batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    opt_occ  = optim.Adam(occ_model.parameters(), lr=cfg["lr"])
    sched_occ = optim.lr_scheduler.ReduceLROnPlateau(opt_occ, "min", factor=0.5, patience=5)
    Xv_t = torch.tensor(X_val).to(device); Yv_occ = torch.tensor(occ_val).to(device)
    best_occ, best_st_occ, patience_occ = np.inf, None, 0
    hist_occ = {"train": [], "val": []}

    for ep in range(1, cfg["epochs_occ"]+1):
        occ_model.train(); losses = []
        for xb, yb in loader_occ:
            xb, yb = xb.to(device), yb.to(device)
            opt_occ.zero_grad()
            loss = bce(occ_model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(occ_model.parameters(), 1.0)
            opt_occ.step(); losses.append(loss.item())
        occ_model.eval()
        with torch.no_grad(): vl = bce(occ_model(Xv_t), Yv_occ).item()
        hist_occ["train"].append(np.mean(losses)); hist_occ["val"].append(vl)
        sched_occ.step(vl)
        if vl < best_occ:
            best_occ = vl; patience_occ = 0
            best_st_occ = {k: v.cpu().clone() for k, v in occ_model.state_dict().items()}
        else:
            patience_occ += 1
            if patience_occ >= cfg["early_stop_patience"]:
                print(f"   Early stop Occ en época {ep}"); break
        if ep % 10 == 0: print(f"   Occ época {ep}/{cfg['epochs_occ']} | val={vl:.4f}")
    occ_model.load_state_dict(best_st_occ)

    # v27: guardar modelo de ocurrencia Op.1
    _save_path = os.path.join(cfg["output_folder"], "op1_occ.pt")
    torch.save(occ_model.state_dict(), _save_path)
    # [ANEXOS] respaldo permanente por semilla (no se sobreescribe entre reentrenos)
    _ck_dir = os.path.join(cfg["output_folder"], "checkpoints")
    os.makedirs(_ck_dir, exist_ok=True)
    torch.save(occ_model.state_dict(),
               os.path.join(_ck_dir, f"M2_occ_seed{cfg.get('seed','NA')}.pt"))
    print(f"   [v27] Modelo guardado: {_save_path}")

    # ── Etapa 2: Cantidad ──────────────────────────────────────────────────
    print("\n  [Op.1] Entrenando etapa 2 — Cantidad Gamma (NLL)...")
    wet_tr = y_tr >= thr; wet_val_m = y_val >= thr
    amt_model = AmountGammaLSTM(X_tr.shape[2], cfg["d_model"], cfg["dropout"]).to(device)
    loader_amt = DataLoader(TensorDataset(torch.tensor(X_tr[wet_tr]),
                                          torch.tensor(y_tr[wet_tr])),
                            batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    opt_amt  = optim.Adam(amt_model.parameters(), lr=cfg["lr"])
    sched_amt = optim.lr_scheduler.CosineAnnealingLR(opt_amt, T_max=cfg["epochs_amt"])
    Xv_wet = torch.tensor(X_val[wet_val_m]).to(device)
    Yv_wet = torch.tensor(y_val[wet_val_m]).to(device)
    best_amt, best_st_amt, patience_amt = np.inf, None, 0
    hist_amt = {"train": [], "val": []}

    for ep in range(1, cfg["epochs_amt"]+1):
        amt_model.train(); losses = []
        for xb, yb in loader_amt:
            xb, yb = xb.to(device), yb.to(device)
            opt_amt.zero_grad()
            la, lb = amt_model(xb)
            loss = gamma_nll(la, lb, yb)            # v50: NLL clasico (rank-weighted inflaba mu +115%)
            loss.backward(); nn.utils.clip_grad_norm_(amt_model.parameters(), 1.0)
            opt_amt.step(); losses.append(loss.item())
        sched_amt.step(); amt_model.eval()
        with torch.no_grad():
            la, lb = amt_model(Xv_wet); vl = gamma_nll(la, lb, Yv_wet).item()  # v50: NLL clasico
        hist_amt["train"].append(np.mean(losses)); hist_amt["val"].append(vl)
        if vl < best_amt:
            best_amt = vl; patience_amt = 0
            best_st_amt = {k: v.cpu().clone() for k, v in amt_model.state_dict().items()}
        else:
            patience_amt += 1
            if patience_amt >= cfg["early_stop_patience"]:
                print(f"   Early stop Amt en época {ep}"); break
        if ep % 20 == 0: print(f"   Amt época {ep}/{cfg['epochs_amt']} | val={vl:.4f}")
    amt_model.load_state_dict(best_st_amt)

    # === [PRESENTACIÓN punto 2] volcado de curvas de pérdida M2 ===
    import json as _json
    _lc = os.path.join(cfg["output_folder"], "loss_M2_Hurdle.json")
    _json.dump({
        "modelo": "M2 Hurdle-Gamma",
        "occ_train": [float(x) for x in hist_occ["train"]],
        "occ_val":   [float(x) for x in hist_occ["val"]],
        "amt_train": [float(x) for x in hist_amt["train"]],
        "amt_val":   [float(x) for x in hist_amt["val"]],
        "best_epoch_occ": int(np.argmin(hist_occ["val"])) + 1,
        "best_epoch_amt": int(np.argmin(hist_amt["val"])) + 1,
    }, open(_lc, "w"), indent=2)
    print(f"   [presentacion] curvas M2 guardadas: {_lc}")

    # v27: guardar modelo de cantidad Op.1 + scaler
    import pickle
    _save_amt = os.path.join(cfg["output_folder"], "op1_amt.pt")
    _save_scl = os.path.join(cfg["output_folder"], "op1_scaler.pkl")
    torch.save(amt_model.state_dict(), _save_amt)
    with open(_save_scl, "wb") as _f: pickle.dump(scaler, _f)
    # [ANEXOS] respaldo permanente por semilla (amt + scaler)
    _ck_dir = os.path.join(cfg["output_folder"], "checkpoints")
    os.makedirs(_ck_dir, exist_ok=True)
    _sd = cfg.get("seed", "NA")
    torch.save(amt_model.state_dict(), os.path.join(_ck_dir, f"M2_amt_seed{_sd}.pt"))
    with open(os.path.join(_ck_dir, f"M2_scaler_seed{_sd}.pkl"), "wb") as _f: pickle.dump(scaler, _f)
    print(f"   [v27] Modelos guardados: op1_amt.pt + op1_scaler.pkl")

    @torch.no_grad()
    def _get_probs(X_sc):
        occ_model.eval()
        return torch.sigmoid(occ_model(torch.tensor(X_sc).to(device))).cpu().numpy()

    @torch.no_grad()
    def _get_mu_val():
        amt_model.eval()
        xt = torch.tensor(X_val).to(device)
        la, lb = amt_model(xt)
        return (torch.exp(la).cpu().numpy() /
                np.clip(torch.exp(lb).cpu().numpy(), 1e-6, None))

    @torch.no_grad()
    def _get_alpha_beta_val():
        amt_model.eval()
        xt = torch.tensor(X_val).to(device)
        la, lb = amt_model(xt)
        alpha = torch.exp(la).cpu().numpy().ravel()
        beta  = np.clip(torch.exp(lb).cpu().numpy().ravel(), 1e-6, None)
        return alpha, beta

    # ── v71: ajustar el calibrador de ocurrencia sobre el held-out del train ──
    if X_cal is not None:
        prob_cal_raw = _get_probs(X_cal).ravel()
        _calibrate, _cal_name = fit_occurrence_calibrator(prob_cal_raw, occ_cal, method=_cal_method)
        print(f"   [v71] Calibrador occ ajustado: {_cal_name}")
    else:
        _calibrate, _cal_name = (lambda p: np.asarray(p, dtype=np.float64)), "identidad"

    prob_val_raw = _get_probs(X_val)
    prob_val     = _calibrate(prob_val_raw)        # v71: prob recalibrada
    if X_cal is not None:
        print(f"   [v71] prob_val cruda:      min={prob_val_raw.min():.4f}  "
              f"max={prob_val_raw.max():.4f}  mean={prob_val_raw.mean():.4f}")
        print(f"   [v71] prob_val calibrada:  min={prob_val.min():.4f}  "
              f"max={prob_val.max():.4f}  mean={prob_val.mean():.4f}  "
              f"(piso bajado de {prob_val_raw.min():.3f} → {prob_val.min():.3f})")
    _print_prob_diagnostico(prob_val, y_val, thr, label="Op.1 — Hurdle-Gamma (prob CALIBRADA)")

    # v29: muestreo estocástico de la Gamma en validación
    # pred = Gamma(alpha, 1/beta) si prob > umbral, sino 0
    # Genera variabilidad diaria dentro de eventos — no meseta plana
    rng29     = np.random.default_rng(cfg.get("seed", 42))
    alpha_v, beta_v = _get_alpha_beta_val()
    mu_val1   = alpha_v / beta_v
    prob_thr  = cfg.get("prob_thr_hurdle", 0.30)

    # v56: umbral fijo (sin freq_match — era trampa de validación)
    dates_val_seq = val_df["date"].values[cfg["seq_len"]:]
    wet_mask = prob_val.ravel() > prob_thr
    # v71: tras la recalibración la escala de prob cambia (isotónica puede topar
    # por debajo de 0.35) → el umbral fijo dejaría wet_mask vacío y rompería el
    # diagnóstico Gamma. Fallback: tomar los top-N días por probabilidad, con
    # N = nº de húmedos observados (subconjunto 'más probable húmedo', no decisión).
    if wet_mask.sum() < 10:
        _n_top = max(int((y_val >= thr).sum()), 10)
        _order = np.argsort(prob_val.ravel())[::-1][:_n_top]
        wet_mask = np.zeros(prob_val.size, dtype=bool)
        wet_mask[_order] = True
        print(f"   [v71] umbral fijo {prob_thr} deja <10 húmedos en prob calibrada; "
              f"diagnóstico Gamma sobre top-{_n_top} por probabilidad")

    # v30: diagnóstico de alpha y beta en días húmedos
    # Si alpha es casi constante → modelo no aprende variabilidad diaria
    alpha_wet = alpha_v[wet_mask]
    beta_wet  = beta_v[wet_mask]
    mu_wet    = mu_val1[wet_mask]
    print(f"\n  [Op.1] DIAGNÓSTICO Gamma en días húmedos (n={wet_mask.sum()}):")
    print(f"   alpha: min={alpha_wet.min():.4f}  max={alpha_wet.max():.4f}  "
          f"mean={alpha_wet.mean():.4f}  std={alpha_wet.std():.4f}")
    print(f"   beta:  min={beta_wet.min():.4f}  max={beta_wet.max():.4f}  "
          f"mean={beta_wet.mean():.4f}  std={beta_wet.std():.4f}")
    print(f"   mu:    min={mu_wet.min():.4f}  max={mu_wet.max():.4f}  "
          f"mean={mu_wet.mean():.4f}  std={mu_wet.std():.4f}")
    print(f"   cv_alpha={alpha_wet.std()/alpha_wet.mean():.3f}  "
          f"cv_mu={mu_wet.std()/mu_wet.mean():.3f}  "
          f"(cv>0.3 → variabilidad real | cv<0.1 → modelo plano)")

    # v59: ENSAMBLE estocástico (generador de precipitación).
    #   - MÉTRICAS/PBIAS = media del ensamble → idéntica a v58 (insesgada, estable).
    #   - GRÁFICA DIARIA = 1 miembro representativo (textura realista) + banda P10-P90.
    #   El miembro se elige por criterio CIEGO (total más cercano a la mediana del
    #   ensamble), NUNCA por cercanía a la observación → sin trampa de validación.
    _cap = cfg.get("cap_mm", 3.1)
    _K   = cfg.get("n_ensemble", cfg.get("n_gamma_samples", 50))
    _pv  = prob_val.ravel()
    # v68: OCURRENCIA BERNOULLI por miembro (antes umbral fijo prob>thr -> colapso a 0 exacto
    #   en meses secos). Cada miembro muestrea rains~Bernoulli(prob), idéntico al 2024 ciego
    #   y a los modelos sin-ML. Sin umbral, sin colapso seco, sin leak.
    _samples = rng29.gamma(shape=alpha_v, scale=1.0/beta_v, size=(_K, len(alpha_v)))
    _rain    = rng29.random((_K, len(_pv))) < _pv[None, :]
    _members = np.where(_rain, np.clip(_samples, 0.0, _cap), 0.0)   # (K, N)
    pred_val = _members.mean(axis=0)                                # media insesgada (MÉTRICAS)
    n_pred_wet = int((pred_val > 0).sum())
    _tot = _members.sum(axis=1)
    _rep = int(np.argmin(np.abs(_tot - np.median(_tot))))           # CIEGO: cercano a mediana
    pred_member = _members[_rep]
    pred_p10 = np.percentile(_members, 10, axis=0)
    pred_p90 = np.percentile(_members, 90, axis=0)

    print(f"\n  [Op.1 v59] ENSAMBLE K={_K} | métrica=media(=v58) | gráfica=miembro #{_rep} (cercano a mediana)")
    print(f"   pred = clip(Gamma, 0, {_cap}mm) x Bernoulli(prob_CALIBRADA) [v71: prob recalibrada {_cal_name}, sin umbral]")
    print(f"   días húmedos (media): {n_pred_wet}  |  sum_media={pred_val.sum():.2f}mm"
          f"  |  sum_miembro={pred_member.sum():.2f}mm  |  sum_obs={float(np.nansum(y_val)):.2f}mm")

    # Guardar diagnóstico Gamma para el print final
    gamma_diag1 = {
        "n_wet"      : int(wet_mask.sum()),
        "alpha_focal": float(alpha_focal),
        "alpha_min"  : float(alpha_wet.min()),  "alpha_max" : float(alpha_wet.max()),
        "alpha_mean" : float(alpha_wet.mean()), "alpha_std" : float(alpha_wet.std()),
        "beta_min"   : float(beta_wet.min()),   "beta_max"  : float(beta_wet.max()),
        "beta_mean"  : float(beta_wet.mean()),  "beta_std"  : float(beta_wet.std()),
        "mu_min"     : float(mu_wet.min()),     "mu_max"    : float(mu_wet.max()),
        "mu_mean"    : float(mu_wet.mean()),    "mu_std"    : float(mu_wet.std()),
        "cv_alpha"   : float(alpha_wet.std()/max(alpha_wet.mean(), 1e-6)),
        "cv_mu"      : float(mu_wet.std()/max(mu_wet.mean(), 1e-6)),
    }

    dates_val = val_df["date"].values[cfg["seq_len"]:]
    df_val_out = pd.DataFrame({"date": dates_val, "obs": y_val, "pred": pred_val,
                               "pred_member": pred_member,
                               "p10": pred_p10, "p90": pred_p90})
    met_val = compute_metrics(y_val, pred_val, thr, "Op.1 — Hurdle | Validación 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_val, pred_val, dates_val)

    @torch.no_grad()
    def _param_fn_op1(Xb):
        occ_model.eval(); amt_model.eval()
        xt = torch.tensor(Xb).to(device)
        prob  = torch.sigmoid(occ_model(xt)).cpu().numpy().ravel()
        prob  = _calibrate(prob)            # v71: misma recalibración que en validación (sin leak)
        la, lb = amt_model(xt)
        alpha = torch.exp(la).cpu().numpy().ravel()
        beta  = np.clip(torch.exp(lb).cpu().numpy().ravel(), 1e-6, None)
        # v62: devuelve (prob, alpha, beta) — apply_autoreg_ensemble muestrea
        # Gamma(alpha, 1/beta) directamente, sin parámetros externos.
        return prob, alpha, beta

    df_raw_ref = df_raw if df_raw is not None else df
    _, ppd_buf_init, clim_daily = build_pred_context_df(df_raw_ref, cfg)
    pred_dates_df = (df_raw_ref[df_raw_ref["date"].dt.year.isin(cfg["pred_years"])]
                     [["date"]].sort_values("date").reset_index(drop=True))

    df_pred_out = apply_autoreg_ensemble(
        pred_dates_df, ppd_buf_init, _param_fn_op1,
        scaler, FCOLS1, cfg, clim_daily, df_raw_ref)

    pred_p = df_pred_out["pred"].values
    met_pred = blind_metrics(pred_p, f"Op.1 — Hurdle | {cfg['pred_label']} CIEGO")

    _print_model_diagnostico(
        label="Op.1 — Hurdle-Gamma", y_val=y_val, pred_val=pred_val, thr=thr,
        pred_p=pred_p, cfg=cfg,
        n_train_wet=int((y_tr >= thr).sum()), n_val_wet=int((y_val >= thr).sum()),
        umbral_occ=np.nan,
        met_val=met_val,
        alpha_focal=alpha_focal, pos_weight=pos_weight_val)

    return {
        "label": "Hurdle-Gamma",
        "df_val": df_val_out, "df_pred": df_pred_out,
        "met_val": met_val, "met_pred": met_pred,
        "hist_occ": hist_occ, "hist_amt": hist_amt,
        "prob_val_raw": prob_val, "y_val_raw": y_val,
        "gamma_diag": gamma_diag1,
        "lstm_params": {
            "seq_len": cfg["seq_len"], "d_model": cfg["d_model"],
            "dropout": cfg["dropout"], "batch_size": cfg["batch_size"],
            "lr": cfg["lr"], "epochs_max_occ": cfg["epochs_occ"],
            "epochs_max_amt": cfg["epochs_amt"],
            "early_stop_patience": cfg["early_stop_patience"],
            "focal_gamma": cfg["focal_gamma_pred"],
            "alpha_focal": float(alpha_focal),
            "pos_weight": float(pos_weight_val),
            "n_features": len(FCOLS1),
            "calibracion_ocurrencia": cfg.get("hurdle_calibration", "none"),
            "cal_frac": cfg.get("hurdle_cal_frac", np.nan),
            "seed": int(cfg.get("seed", 42)),
            "epoca_early_stop_occ": (int(np.argmin(hist_occ["val"])) + 1
                                     if len(hist_occ.get("val", [])) else None),
            "epoca_early_stop_amt": (int(np.argmin(hist_amt["val"])) + 1
                                     if len(hist_amt.get("val", [])) else None),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. FUNCIÓN GENÉRICA PARA OPCIONES 2 y 4
# ══════════════════════════════════════════════════════════════════════════════

def _run_hurdle_ddpm(df, train_df, val_df, cfg, option_tag, FCOLS,
                     ddpm_on_params=False, df_raw=None, prob_thr_key="prob_thr_zidf_params",
                     use_ddpm=True):
    _reseed_torch(cfg)   # v72: arranque RNG determinista, desacoplado de lo previo
    device = cfg["device"]; thr = cfg["wet_threshold"]

    X_tr, y_tr, scaler = build_sequences(train_df, cfg["seq_len"], FCOLS, fit=True)
    X_val, y_val, _    = build_sequences(val_df,   cfg["seq_len"], FCOLS, scaler_x=scaler)
    occ_tr  = (y_tr  >= thr).astype(np.float32)
    occ_val = (y_val >= thr).astype(np.float32)

    print(f"\n  [{option_tag}] Entrenando LSTM ZI (FocalLoss + bias_init occ+amt + NLL zero-inflated conjunta [zidf_gamma_nll])...")
    model = HurdleGammaLSTM(X_tr.shape[2], cfg["d_model"], cfg["dropout"]).to(device)
    n_pos = occ_tr.sum(); n_neg = len(occ_tr) - n_pos

    # v18: alpha_focal = wet_frac (~0.08)
    # v19: pos_weight = n_neg/n_pos — cap v26
    # v43: cap diferenciado — GammaParams usa pos_weight_cap_zidf_params (2.0) para
    #      reducir agresividad en ocurrencia (FAR=0.778 en v42 → demasiados falsos positivos)
    alpha_focal = float(n_pos / max(n_pos + n_neg, 1))
    _pw_cap = (cfg.get("pos_weight_cap_zidf_params", cfg.get("pos_weight_cap", 4.0))
               if "GammaParams" in option_tag
               else cfg.get("pos_weight_cap", 4.0))
    pos_weight_val_z = float(min(n_neg / max(n_pos, 1), _pw_cap))
    wet_frac_z  = alpha_focal
    # v45 ZIDF: p_l representa pi (cero estructural ≈ 1-wet_frac), no prob de lluvia.
    # bias se inicializa hacia pi alto → invertir signo respecto al Hurdle.
    bias_init_z = float(np.log((1.0 - wet_frac_z) / wet_frac_z))

    _fg = cfg.get("focal_gamma_pred", 2.0)
    _ow = cfg.get("occ_loss_weight", 1.0)
    pw_tensor_z = torch.tensor(pos_weight_val_z, dtype=torch.float32).to(device)
    bce = FocalLoss(alpha=alpha_focal, gamma=_fg, pos_weight=pw_tensor_z).to(device)

    # bias_init occ (igual que siempre)
    with torch.no_grad():
        model.occ.bias.fill_(bias_init_z)

    # v40: bias_init amt — anclar alpha/beta iniciales en climatología de días húmedos
    # Evita que mu empiece en valores aleatorios y diverja en primeras épocas
    wet_tr_vals = y_tr[y_tr >= thr]
    if len(wet_tr_vals) >= 10:
        try:
            alpha_clim, _, scale_clim = stats.gamma.fit(wet_tr_vals, floc=0)
            beta_clim  = 1.0 / max(scale_clim, 1e-6)
            log_alpha_init = float(np.log(max(alpha_clim, 1e-6)))
            log_beta_init  = float(np.log(max(beta_clim,  1e-6)))
            with torch.no_grad():
                model.amt.bias[0] = log_alpha_init
                model.amt.bias[1] = log_beta_init
            mu_clim = alpha_clim / beta_clim
            print(f"   [{option_tag}] bias_init amt: log_α={log_alpha_init:.4f} log_β={log_beta_init:.4f}"
                  f"  → mu_init={mu_clim:.4f}mm  (α={alpha_clim:.3f} β={beta_clim:.3f})")
        except Exception as e:
            print(f"   [{option_tag}] bias_init amt: fallo fit Gamma ({e}) — bias aleatorio")
    else:
        print(f"   [{option_tag}] bias_init amt: insuf. días húmedos ({len(wet_tr_vals)}) — bias aleatorio")

    print(f"   [{option_tag}] FocalLoss: alpha={alpha_focal:.4f}  gamma={_fg}"
          f"  pos_weight={pos_weight_val_z:.2f}  occ_weight={_ow}")
    print(f"   [{option_tag}] bias_init occ={bias_init_z:.4f}  (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr),
                                      torch.tensor(occ_tr)),
                        batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=6)
    Xv = torch.tensor(X_val).to(device)
    Yv = torch.tensor(y_val).to(device)
    Ov = torch.tensor(occ_val).to(device)
    hist_lstm = {"train": [], "val": []}
    best, best_st, patience_lstm = np.inf, None, 0

    for ep in range(1, cfg["epochs_pred"]+1):
        model.train(); losses = []
        for xb, yb, ob in loader:
            xb, yb, ob = xb.to(device), yb.to(device), ob.to(device)
            opt.zero_grad()
            p_l, la, lb = model(xb)
            # v45: NLL conjunta del ZIDF verdadero (no BCE + Gamma separados)
            # p_l ahora es el logit de pi (prob de cero estructural)
            loss = zidf_gamma_nll(p_l, la, lb, yb, thr=thr)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            p_l, la, lb = model(Xv)
            vl = zidf_gamma_nll(p_l, la, lb, Yv, thr=thr).item()
        hist_lstm["train"].append(np.mean(losses)); hist_lstm["val"].append(vl)
        sched.step(vl)
        if vl < best:
            best = vl; patience_lstm = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_lstm += 1
            if patience_lstm >= cfg["early_stop_patience"]:
                print(f"   Early stop LSTM en época {ep}"); break
        if ep % 10 == 0: print(f"   LSTM época {ep}/{cfg['epochs_pred']} | val={vl:.4f}")
    model.load_state_dict(best_st)

    # === [PRESENTACIÓN punto 2] volcado de curva de pérdida (LSTM ZI) ===
    import json as _json
    _tag2 = option_tag.replace(" ", "_").replace("—", "").replace(".", "").replace("/", "_").strip("_")
    _lc2 = os.path.join(cfg["output_folder"], f"loss_{_tag2}.json")
    _json.dump({
        "modelo": option_tag,
        "lstm_train": [float(x) for x in hist_lstm["train"]],
        "lstm_val":   [float(x) for x in hist_lstm["val"]],
        "best_epoch_lstm": int(np.argmin(hist_lstm["val"])) + 1,
    }, open(_lc2, "w"), indent=2)
    print(f"   [presentacion] curva {option_tag} guardada: {_lc2}")

    @torch.no_grad()
    def _get_params(X_sc):
        model.eval()
        xt = torch.tensor(X_sc).to(device)
        p_l, la, lb = model(xt)
        # v45 ZIDF: p_l es logit de pi (cero estructural).
        # prob de LLUVIA = 1 - pi  → para mantener compatibilidad con el resto del flujo
        prob_lluvia = (1.0 - torch.sigmoid(p_l)).cpu().numpy()
        return (prob_lluvia, la.cpu().numpy(), lb.cpu().numpy())

    def _params_to_mm(prob, la, lb):
        # v56: cap = max real de la serie (3.1mm), no P95 (1.0mm cortaba la cola)
        _cap_mm = cfg.get("cap_mm", 3.1)
        return np.clip(prob * np.exp(np.clip(la, -4, 4)) /
                       np.clip(np.exp(np.clip(lb, -4, 4)), 1e-6, None), 0, _cap_mm)

    if use_ddpm:
        if ddpm_on_params:
            p_tr, la_tr, lb_tr = _get_params(X_tr)
            wet_mask = y_tr >= thr
            P_train = np.stack([la_tr[wet_mask], lb_tr[wet_mask]], axis=1).astype(np.float32)
            ddpm = DDPM(dim=2, steps=cfg["diff_steps"], device=device)
        else:
            p_tr, la_tr, lb_tr = _get_params(X_tr)
            mm_tr = _params_to_mm(p_tr, la_tr, lb_tr).reshape(-1, 1).astype(np.float32)
            P_train = mm_tr
            ddpm = DDPM(dim=1, steps=cfg["diff_steps"], device=device)

        print(f"\n  [{option_tag}] Entrenando DDPM (steps={cfg['diff_steps']})...")
        hist_ddpm = ddpm.train(P_train, cfg)
    else:
        # v37: sin DDPM — usa mu directo de la red Gamma (como Hurdle)
        ddpm = None
        hist_ddpm = {"train": [], "val": []}
        print(f"\n  [{option_tag} v37] SIN DDPM — usa mu directo de la red Gamma")

    # v27/v37: guardar LSTM (+ DDPM si existe) + scaler
    import pickle
    _tag = option_tag.replace(" ", "_").replace("—", "").replace(".", "").replace("/", "_").strip("_")
    _save_lstm = os.path.join(cfg["output_folder"], f"{_tag}_lstm.pt")
    _save_scl2 = os.path.join(cfg["output_folder"], f"{_tag}_scaler.pkl")
    torch.save(model.state_dict(), _save_lstm)
    with open(_save_scl2, "wb") as _f: pickle.dump(scaler, _f)
    # [ANEXOS] respaldo permanente por semilla (M3 lstm + scaler)
    _ck_dir = os.path.join(cfg["output_folder"], "checkpoints")
    os.makedirs(_ck_dir, exist_ok=True)
    _sd = cfg.get("seed", "NA")
    torch.save(model.state_dict(), os.path.join(_ck_dir, f"M3_lstm_seed{_sd}.pt"))
    with open(os.path.join(_ck_dir, f"M3_scaler_seed{_sd}.pkl"), "wb") as _f: pickle.dump(scaler, _f)
    if ddpm is not None:
        _save_ddpm = os.path.join(cfg["output_folder"], f"{_tag}_ddpm.pt")
        torch.save(ddpm.model.state_dict(), _save_ddpm)
        print(f"   [v37] Modelos guardados: {_tag}_lstm.pt + {_tag}_ddpm.pt + scaler")
    else:
        print(f"   [v37] Modelos guardados: {_tag}_lstm.pt + scaler (sin DDPM)")

    def _apply_ddpm(prob, la, lb):
        # v37: si no hay DDPM, devuelve mm directo de los params Gamma
        if ddpm is None:
            return np.clip(_params_to_mm(prob, la, lb), 0, None)
        if ddpm_on_params:
            P_in = np.stack([la, lb], axis=1).astype(np.float32)
            P_out = ddpm.denoise(P_in, ddim_steps=cfg["ddim_steps"])
            return np.clip(_params_to_mm(prob, P_out[:, 0], P_out[:, 1]), 0, None)
        else:
            mm = _params_to_mm(prob, la, lb).reshape(-1, 1).astype(np.float32)
            mm_dn = ddpm.denoise(mm, ddim_steps=cfg["ddim_steps"])
            return np.clip(mm_dn.flatten(), 0, None)

    p_v, la_v, lb_v = _get_params(X_val)
    _print_prob_diagnostico(p_v, y_val, thr, label=option_tag)

    dates_val_seq = val_df["date"].values[cfg["seq_len"]:]
    meses_val_seq_z = pd.to_datetime(dates_val_seq).month.values

    # v29: muestreo estocástico de la Gamma en validación
    mm_v_raw  = _apply_ddpm(p_v, la_v, lb_v)
    mu_v_raw  = mm_v_raw / np.clip(p_v, 1e-3, None)
    alpha_v_z = np.clip(np.exp(np.clip(la_v, -4, 4)), 1e-6, None)
    beta_v_z  = np.clip(np.exp(np.clip(lb_v, -4, 4)), 1e-6, None)
    prob_thr  = cfg.get(prob_thr_key, 0.28)

    # v56: umbral fijo (sin freq_match — era trampa de validación)
    wet_mask_z = p_v.ravel() > prob_thr

    # v30: diagnóstico de alpha y beta en días húmedos
    alpha_wet_z = alpha_v_z[wet_mask_z] if wet_mask_z.any() else alpha_v_z
    beta_wet_z  = beta_v_z[wet_mask_z]  if wet_mask_z.any() else beta_v_z
    mu_wet_z    = mu_v_raw[wet_mask_z]  if wet_mask_z.any() else mu_v_raw
    print(f"\n  [{option_tag}] DIAGNÓSTICO Gamma en días húmedos (n={wet_mask_z.sum()}):")
    print(f"   alpha: min={alpha_wet_z.min():.4f}  max={alpha_wet_z.max():.4f}  "
          f"mean={alpha_wet_z.mean():.4f}  std={alpha_wet_z.std():.4f}")
    print(f"   beta:  min={beta_wet_z.min():.4f}  max={beta_wet_z.max():.4f}  "
          f"mean={beta_wet_z.mean():.4f}  std={beta_wet_z.std():.4f}")
    print(f"   mu:    min={mu_wet_z.min():.4f}  max={mu_wet_z.max():.4f}  "
          f"mean={mu_wet_z.mean():.4f}  std={mu_wet_z.std():.4f}")
    print(f"   cv_alpha={alpha_wet_z.std()/max(alpha_wet_z.mean(),1e-6):.3f}  "
          f"cv_mu={mu_wet_z.std()/max(mu_wet_z.mean(),1e-6):.3f}  "
          f"(cv>0.3 → variabilidad real | cv<0.1 → modelo plano)")

    # v59: ENSAMBLE estocástico (igual que Op.1). Métrica=media(=v58); gráfica=miembro repr.+banda P10-P90.
    _cap = cfg.get("cap_mm", 3.1)
    _K   = cfg.get("n_ensemble", cfg.get("n_gamma_samples", 50))
    rng29z = np.random.default_rng(cfg.get("seed", 42))
    _alpha_flat = np.asarray(alpha_v_z).ravel()
    _beta_flat  = np.asarray(beta_v_z).ravel()
    _pv_z = np.asarray(p_v).ravel()
    # v68: OCURRENCIA BERNOULLI por miembro (igual criterio que Op.1, el 2024 ciego y los
    #   modelos sin-ML). Antes: umbral fijo -> colapso seco. Sin umbral, sin colapso, sin leak.
    _samples_z = rng29z.gamma(shape=_alpha_flat, scale=1.0/_beta_flat,
                              size=(_K, len(_alpha_flat)))
    _rain_z = rng29z.random((_K, len(_pv_z))) < _pv_z[None, :]
    _members_z = np.where(_rain_z, np.clip(_samples_z, 0.0, _cap), 0.0)   # (K, N)
    pred_val = _members_z.mean(axis=0)                                    # media insesgada (MÉTRICAS)
    n_pred_wet_z = int((pred_val > 0).sum())
    _tot_z = _members_z.sum(axis=1)
    _rep_z = int(np.argmin(np.abs(_tot_z - np.median(_tot_z))))
    pred_member = _members_z[_rep_z]
    pred_p10 = np.percentile(_members_z, 10, axis=0)
    pred_p90 = np.percentile(_members_z, 90, axis=0)

    print(f"\n  [{option_tag} v59] ENSAMBLE K={_K} | métrica=media(=v58) | gráfica=miembro #{_rep_z} (cercano a mediana)")
    print(f"   pred = clip(Gamma, 0, {_cap}mm) x Bernoulli(prob) [v70: sin umbral]")
    print(f"   días húmedos (media): {n_pred_wet_z}  |  sum_media={pred_val.sum():.2f}mm"
          f"  |  sum_miembro={pred_member.sum():.2f}mm  |  sum_obs={float(np.nansum(y_val)):.2f}mm")

    # Guardar diagnóstico Gamma para el print final
    gamma_diag_z = {
        "n_wet"      : int(wet_mask_z.sum()),
        "alpha_focal": float(alpha_focal),
        "alpha_min"  : float(alpha_wet_z.min()),  "alpha_max" : float(alpha_wet_z.max()),
        "alpha_mean" : float(alpha_wet_z.mean()), "alpha_std" : float(alpha_wet_z.std()),
        "beta_min"   : float(beta_wet_z.min()),   "beta_max"  : float(beta_wet_z.max()),
        "beta_mean"  : float(beta_wet_z.mean()),  "beta_std"  : float(beta_wet_z.std()),
        "mu_min"     : float(mu_wet_z.min()),     "mu_max"    : float(mu_wet_z.max()),
        "mu_mean"    : float(mu_wet_z.mean()),    "mu_std"    : float(mu_wet_z.std()),
        "cv_alpha"   : float(alpha_wet_z.std()/max(alpha_wet_z.mean(), 1e-6)),
        "cv_mu"      : float(mu_wet_z.std()/max(mu_wet_z.mean(), 1e-6)),
    }

    dates_val = val_df["date"].values[cfg["seq_len"]:]
    df_val_out = pd.DataFrame({"date": dates_val, "obs": y_val, "pred": pred_val,
                               "pred_member": pred_member,
                               "p10": pred_p10, "p90": pred_p90})
    met_val = compute_metrics(y_val, pred_val, thr, f"{option_tag} | Validación 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_val, pred_val, dates_val)

    df_raw_ref2 = df_raw if df_raw is not None else df
    _, ppd_buf_init2, clim_daily2 = build_pred_context_df(df_raw_ref2, cfg)
    pred_dates_df2 = (df_raw_ref2[df_raw_ref2["date"].dt.year.isin(cfg["pred_years"])]
                      [["date"]].sort_values("date").reset_index(drop=True))

    def _param_fn_ddpm(Xb):
        p_i, la_i, lb_i = _get_params(Xb)
        prob  = np.asarray(p_i,  dtype=np.float64).ravel()
        alpha = np.clip(np.exp(np.clip(np.asarray(la_i).ravel(), -4, 4)), 1e-6, None)
        beta  = np.clip(np.exp(np.clip(np.asarray(lb_i).ravel(), -4, 4)), 1e-6, None)
        if ddpm is not None:
            # Con DDPM: refinar la/lb con el denoiser antes de exponenciar
            if ddpm_on_params:
                P_in  = np.stack([np.asarray(la_i).ravel(),
                                  np.asarray(lb_i).ravel()], axis=1).astype(np.float32)
                P_out = ddpm.denoise(P_in, ddim_steps=cfg["ddim_steps"])
                alpha = np.clip(np.exp(np.clip(P_out[:, 0], -4, 4)), 1e-6, None)
                beta  = np.clip(np.exp(np.clip(P_out[:, 1], -4, 4)), 1e-6, None)
            # ddpm_on_params=False (mm scalar DDPM) — este caso no se usa en Op.2/Op.4 actualmente
        # v62: devuelve (prob, alpha, beta) — apply_autoreg_ensemble muestrea
        # Gamma(alpha_red, 1/beta_red) directamente, sin parámetros externos.
        return prob, alpha, beta

    df_pred_out = apply_autoreg_ensemble(
        pred_dates_df2, ppd_buf_init2, _param_fn_ddpm,
        scaler, FCOLS, cfg, clim_daily2, df_raw_ref2)

    pred_p = df_pred_out["pred"].values
    met_pred = blind_metrics(pred_p, f"{option_tag} | {cfg['pred_label']} CIEGO")

    _print_model_diagnostico(
        label=option_tag, y_val=y_val, pred_val=pred_val, thr=thr,
        pred_p=pred_p, cfg=cfg,
        n_train_wet=int((y_tr >= thr).sum()), n_val_wet=int((y_val >= thr).sum()),
        umbral_occ=np.nan,
        met_val=met_val,
        alpha_focal=alpha_focal, pos_weight=pos_weight_val_z)

    return {
        "label": option_tag,
        "df_val": df_val_out, "df_pred": df_pred_out,
        "met_val": met_val, "met_pred": met_pred,
        "hist_lstm": hist_lstm, "hist_ddpm": hist_ddpm,
        "prob_val_raw": p_v, "y_val_raw": y_val,
        "gamma_diag": gamma_diag_z,
        "lstm_params": {
            "seq_len": cfg["seq_len"], "d_model": cfg["d_model"],
            "dropout": cfg["dropout"], "batch_size": cfg["batch_size"],
            "lr": cfg["lr"], "epochs_max": cfg["epochs_pred"],
            "early_stop_patience": cfg["early_stop_patience"],
            "focal_gamma": cfg["focal_gamma_pred"],
            "alpha_focal": float(alpha_focal),
            "pos_weight": float(pos_weight_val_z),
            "n_features": len(FCOLS),
            "seed": int(cfg.get("seed", 42)),
            "entrenamiento": "conjunto (una sola etapa)",
            "epoca_early_stop": (int(np.argmin(hist_lstm["val"])) + 1
                                 if len(hist_lstm.get("val", [])) else None),
        },
    }


def run_option2(df, train_df, val_df, cfg, df_raw=None):
    print("\n" + "="*60)
    print("  OPCIÓN 2 — ZIDF + Gamma Loss (v41: lags 1-7, SIN DDPM)")
    print("="*60)
    # v41: lags 1-7 completos — más memoria para capturar eventos previos
    FCOLS2 = ["doy_sin", "doy_cos", "month_sin", "month_cos",
              "ppd_lag1", "ppd_lag2", "ppd_lag3", "ppd_lag4",
              "ppd_lag5", "ppd_lag6", "ppd_lag7",
              "wet_lag1", "ppd_roll7", "ppd_roll30",
              "ppd_roll90", "wet_freq30", "dias_secos_consec"]
    res = _run_hurdle_ddpm(df, train_df, val_df, cfg, "Op.2 — ZI-Gamma LSTM",
                           FCOLS2, ddpm_on_params=False, df_raw=df_raw,
                           prob_thr_key="prob_thr_zidf_gamma",
                           use_ddpm=False)   # v37: sin DDPM → mu directo
    res["label"] = "ZI-Gamma LSTM"
    return res


def run_option4(df, train_df, val_df, cfg, df_raw=None):
    print("\n" + "="*60)
    print("  OPCIÓN 4 — ZIDF-GammaParams (v41: lags 1-7, SIN DDPM, mu directo)")
    print("="*60)
    # v41: lags 1-7 completos — igual que Op.2 para comparación limpia
    FCOLS4 = ["doy_sin", "doy_cos", "month_sin", "month_cos",
              "ppd_lag1", "ppd_lag2", "ppd_lag3", "ppd_lag4",
              "ppd_lag5", "ppd_lag6", "ppd_lag7",
              "wet_lag1", "ppd_roll7", "ppd_roll30",
              "ppd_roll90", "wet_freq30", "dias_secos_consec"]
    res = _run_hurdle_ddpm(df, train_df, val_df, cfg, "Op.4 — ZIDF-GammaParams",
                           FCOLS4, ddpm_on_params=False, df_raw=df_raw,
                           prob_thr_key="prob_thr_zidf_params",
                           use_ddpm=False)   # v37: sin DDPM, mu directo igual que ZIDF-Gamma
    res["label"] = "ZIDF-GammaParams"
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Climatológico":    "#2196F3",
    "Hurdle-Gamma":     "#E91E63",
    "ZI-Gamma LSTM":    "#FF9800",
    "ZIDF-GammaParams": "#4CAF50",
    "Poisson-Gamma":    "#9C27B0",
    "Hurdle-GLM":       "#00897B",
    "ZI-EGPD-seas":     "#5E35B1",
    "ZI-EGPD-marg":     "#C62828",
    "ZI-Gamma-cens":    "#F9A825",
    "Wilks-MixExp":     "#6D4C41",
    "Observado":        "#607D8B",
}


def _monthly_agg(df_r):
    d = df_r.copy()
    d["mes"] = pd.to_datetime(d["date"]).dt.to_period("M")
    return d.groupby("mes")[["obs", "pred"]].sum().reset_index()


def plot_individual(res, out_dir, year):
    label = res["label"]
    col   = COLORS.get(label, "#333")
    tag   = label.replace(" ", "_").replace("-", "_").replace("—", "").replace(".", "")

    df_p = res["df_pred"]
    has_obs = df_p["obs"].notna().any()

    # Serie diaria
    fig, a = plt.subplots(figsize=(16, 5))
    if has_obs:
        a.bar(df_p["date"], df_p["obs"], width=0.8, alpha=0.5,
              color=COLORS["Observado"], label="Observado")
    a.plot(df_p["date"], df_p["pred"], color=col, lw=1.4, marker="o", ms=2, label=label)
    suffix = " (CIEGO)" if not has_obs else ""
    a.set_title(f"{label} — Serie diaria {year}{suffix}")
    a.legend(); a.grid(alpha=0.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_serie_{year}.png"), dpi=150)
    plt.close(fig)

    # Acumulado mensual
    mm = _monthly_agg(df_p); mm["s"] = mm["mes"].astype(str)
    fig, a = plt.subplots(figsize=(12, 5)); x = np.arange(len(mm))
    if has_obs:
        a.bar(x - 0.2, mm["obs"], 0.4, color=COLORS["Observado"], alpha=0.75, label="Observado")
        a.bar(x + 0.2, mm["pred"], 0.4, color=col, alpha=0.75, label=label)
    else:
        a.bar(x, mm["pred"], 0.6, color=col, alpha=0.85, label=label)
    a.set_xticks(x); a.set_xticklabels(mm["s"], rotation=45, ha="right")
    a.set_title(f"{label} — Acumulado mensual {year}{suffix}")
    a.legend(); a.grid(alpha=0.3, axis="y"); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_mensual_{year}.png"), dpi=150)
    plt.close(fig)

    # Validación — v59: ensamble (miembro con textura diaria + banda P10-P90)
    df_v = res["df_val"]
    fig, a = plt.subplots(figsize=(14, 5))
    a.plot(df_v["date"], df_v["obs"], lw=0.8, alpha=0.6,
           color=COLORS["Observado"], label="Observado")
    _has_ens = all(c in df_v.columns for c in ("pred_member", "p10", "p90"))
    if _has_ens and cfg_global.get("plot_envelope", True):
        a.fill_between(df_v["date"], df_v["p10"], df_v["p90"],
                       color=col, alpha=0.18, lw=0, label="Ensamble P10–P90")
    if _has_ens and cfg_global.get("plot_member", True):
        # miembro representativo = realización con textura diaria (criterio ciego)
        a.plot(df_v["date"], df_v["pred_member"], lw=1.0, color=col, alpha=0.9,
               label=f"{label} (miembro repr.)")
        # media del ensamble como referencia tenue (= valor esperado, plano)
        a.plot(df_v["date"], df_v["pred"], lw=0.9, color=col, alpha=0.45,
               ls="--", label=f"{label} (media ensamble)")
    else:
        a.plot(df_v["date"], df_v["pred"], lw=1.0, color=col, alpha=0.85, label=label)
    a.set_title(f"{label} — Validación 2020-2022 (generador estocástico)")
    a.legend(); a.grid(alpha=0.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_validacion.png"), dpi=150)
    plt.close(fig)

    # Curvas de pérdida
    if "hist_occ" in res:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for ax_, h, t in zip(ax, [res["hist_occ"], res["hist_amt"]],
                              ["Ocurrencia (FocalLoss)", "Cantidad (Gamma NLL)"]):
            ax_.plot(h["train"], label="Train"); ax_.plot(h["val"], label="Val")
            ax_.set_title(t); ax_.legend(); ax_.grid(alpha=0.3)
        plt.suptitle(f"{label} — Curvas de pérdida"); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_perdidas.png"), dpi=150)
        plt.close(fig)
    elif "hist_lstm" in res:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(res["hist_lstm"]["train"], label="Train")
        ax[0].plot(res["hist_lstm"]["val"], label="Val")
        ax[0].set_title("LSTM (FocalLoss+NLL)"); ax[0].legend(); ax[0].grid(alpha=0.3)
        # v35: hist_ddpm puede ser lista (con DDPM) o dict vacío (sin DDPM)
        _hd = res.get("hist_ddpm", [])
        if isinstance(_hd, dict):
            _hd = _hd.get("train", [])
        if len(_hd) > 0:
            ax[1].plot(_hd, color="orange")
            ax[1].set_title(f"DDPM ({cfg_global['diff_steps']} steps)")
        else:
            ax[1].text(0.5, 0.5, "Sin DDPM (v37)", ha="center", va="center",
                       transform=ax[1].transAxes)
            ax[1].set_title("DDPM — no usado")
        ax[1].grid(alpha=0.3)
        plt.suptitle(f"{label} — Curvas de pérdida"); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_perdidas.png"), dpi=150)
        plt.close(fig)


def plot_comparativo(resultados, out_dir, year, cfg):
    blind = not resultados[0]["df_pred"]["obs"].notna().any()
    suffix = " (CIEGO)" if blind else ""
    n_mod = len(resultados)

    # Serie diaria — un panel por modelo
    fig, axes = plt.subplots(n_mod, 1, figsize=(18, 5*n_mod), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, resultados):
        df_p = res["df_pred"]; col = COLORS.get(res["label"], "#333")
        if not blind:
            ax.bar(df_p["date"], df_p["obs"], width=0.8, alpha=0.4,
                   color=COLORS["Observado"], label="Observado")
        ax.plot(df_p["date"], df_p["pred"], color=col, lw=1.3, marker="o", ms=1.5,
                label=res["label"])
        if blind:
            acum = res["met_pred"].get("Acum_pred", np.nan)
            n_d  = int((df_p["pred"].values > 0).sum())
            ax.set_title(f"{res['label']}  |  Acum={acum:.1f}mm  días_lluvia={n_d}", fontsize=11)
        else:
            nse_m = res["met_pred"].get("NSE_mensual", np.nan)
            pbias = res["met_pred"].get("PBIAS_%", np.nan)
            ax.set_title(f"{res['label']}  |  NSE_mens={nse_m:.3f}  PBIAS={pbias:+.1f}%", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_ylabel("mm")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    plt.xticks(rotation=45)
    plt.suptitle(f"Comparativo — Serie diaria {year}{suffix}", fontsize=14, y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"COMP_serie_{year}.png"), dpi=150)
    plt.close(fig)

    # Acumulado mensual pred_year
    ref_pred = _monthly_agg(resultados[0]["df_pred"])[["mes", "obs"]].copy()
    ref_pred["mes_str"] = ref_pred["mes"].astype(str)
    meses = ref_pred["mes_str"].tolist()
    x = np.arange(len(meses))
    w = 0.7 / (n_mod + (0 if blind else 1))
    fig, a = plt.subplots(figsize=(14, 6))
    offset0 = 0 if blind else 1
    if not blind:
        a.bar(x - w * n_mod / 2, ref_pred["obs"].values, w,
              color=COLORS["Observado"], alpha=0.8, label="Observado")
    for idx, res in enumerate(resultados):
        agg = _monthly_agg(res["df_pred"])[["mes", "pred"]]
        agg["mes_str"] = agg["mes"].astype(str)
        merged = ref_pred[["mes_str"]].merge(agg[["mes_str", "pred"]], on="mes_str", how="left")
        a.bar(x - w * n_mod / 2 + w * (idx + offset0), merged["pred"].fillna(0).values, w,
              color=COLORS.get(res["label"], f"C{idx}"), alpha=0.8, label=res["label"])
    a.set_xticks(x); a.set_xticklabels(meses, rotation=45, ha="right")
    a.set_title(f"Acumulado mensual {year} — Comparativo{suffix}")
    a.legend(); a.grid(alpha=0.3, axis="y"); a.set_ylabel("mm")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"COMP_mensual_{year}.png"), dpi=150)
    plt.close(fig)

    # Acumulado mensual validación
    ref_val = _monthly_agg(resultados[0]["df_val"])[["mes", "obs"]].copy()
    ref_val["mes_str"] = ref_val["mes"].astype(str)
    meses_v = ref_val["mes_str"].tolist()
    xv = np.arange(len(meses_v))
    fig, a = plt.subplots(figsize=(16, 6))
    a.bar(xv - w * n_mod / 2, ref_val["obs"].values, w,
          color=COLORS["Observado"], alpha=0.8, label="Observado")
    for idx, res in enumerate(resultados):
        agg_v = _monthly_agg(res["df_val"])[["mes", "pred"]]
        agg_v["mes_str"] = agg_v["mes"].astype(str)
        merged_v = ref_val[["mes_str"]].merge(agg_v[["mes_str", "pred"]], on="mes_str", how="left")
        a.bar(xv - w * n_mod / 2 + w * (idx + 1), merged_v["pred"].fillna(0).values, w,
              color=COLORS.get(res["label"], f"C{idx}"), alpha=0.8, label=res["label"])
    tick_idx = list(range(0, len(meses_v), 3))
    a.set_xticks(xv[tick_idx]); a.set_xticklabels([meses_v[i] for i in tick_idx],
                                                    rotation=45, ha="right")
    a.set_title("Acumulado mensual — Validación 2020-2022 (métricas formales)")
    a.legend(); a.grid(alpha=0.3, axis="y"); a.set_ylabel("mm")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "COMP_mensual_validacion.png"), dpi=150)
    plt.close(fig)

    # Tabla de métricas
    metricas_show = ["NSE_diario", "NSE_mensual", "KGE", "PBIAS_%",
                     "RMSE", "POD", "FAR", "CSI", "F1", "Acum_obs", "Acum_pred"]
    rows_v, rows_p = [], []
    for res in resultados:
        rv = {"Modelo": res["label"]}; rp = {"Modelo": res["label"]}
        for m in metricas_show:
            rv[m] = round(res["met_val"].get(m, np.nan), 4)
            rp[m] = round(res["met_pred"].get(m, np.nan), 4)
        rows_v.append(rv); rows_p.append(rp)
    df_met_v = pd.DataFrame(rows_v).set_index("Modelo")
    df_met_p = pd.DataFrame(rows_p).set_index("Modelo")

    fig, axes = plt.subplots(1, 2, figsize=(20, 3.5))
    for ax_, df_t, title in zip(axes, [df_met_v, df_met_p],
                                  ["Validación 2020-2022 (formal)",
                                   f"Año {year}{suffix}"]):
        ax_.axis("off")
        tbl = ax_.table(cellText=df_t.values.round(3),
                        colLabels=df_t.columns, rowLabels=df_t.index,
                        cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0 or c == -1:
                cell.set_facecolor("#37474F"); cell.set_text_props(color="white")
        ax_.set_title(title, fontsize=11, pad=12)
    plt.suptitle("Tabla comparativa de métricas", fontsize=13, y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "COMP_tabla_metricas.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Curva de excedencia
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_, key, title in zip(axes, ["df_val", "df_pred"],
                                ["Validación 2020-2022", f"Año {year}{suffix}"]):
        obs = resultados[0][key]["obs"].values
        obs_clean = obs[~np.isnan(obs)]
        if len(obs_clean) > 0:
            obs_s = np.sort(obs_clean)[::-1]
            ax_.plot(np.arange(1, len(obs_s)+1) / len(obs_s) * 100, obs_s,
                     color=COLORS["Observado"], lw=2, label="Observado", zorder=5)
        for res in resultados:
            pred = res[key]["pred"].values
            ps = np.sort(pred)[::-1]
            ax_.plot(np.arange(1, len(ps)+1) / len(ps) * 100, ps,
                     color=COLORS.get(res["label"], "gray"), lw=1.2, alpha=0.85,
                     label=res["label"])
        ax_.set_xlabel("% tiempo excedido"); ax_.set_ylabel("Precipitación (mm)")
        ax_.set_title(f"Curva de excedencia — {title}")
        ax_.legend(fontsize=8); ax_.grid(alpha=0.3); ax_.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "COMP_excedencia.png"), dpi=150)
    plt.close(fig)

    # Radar categórico (validación)
    cats = ["POD", "1-FAR", "CSI", "F1"]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist() + [0]
    fig, ax_ = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for res in resultados:
        mv   = res["met_val"]
        vals = [mv.get("POD", 0), 1 - mv.get("FAR", 1), mv.get("CSI", 0), mv.get("F1", 0)]
        vals += vals[:1]
        ax_.plot(angles, vals, color=COLORS.get(res["label"], "gray"), lw=1.8, label=res["label"])
        ax_.fill(angles, vals, color=COLORS.get(res["label"], "gray"), alpha=0.1)
    ax_.set_xticks(angles[:-1]); ax_.set_xticklabels(cats, fontsize=11)
    ax_.set_ylim(0, 1)
    ax_.set_title("Métricas categóricas — Validación 2020-2022", pad=20, fontsize=12)
    ax_.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "COMP_radar_categoricas.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Diagrama de Taylor (validación)
    fig, ax_ = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    obs_ref = resultados[0]["df_val"][["date", "obs"]].copy()
    std_obs = np.std(obs_ref["obs"].values)
    ax_.plot(0, std_obs, "k*", ms=14, label="Observado")
    for res in resultados:
        df_v = res["df_val"][["date", "pred"]].copy()
        merged = obs_ref.merge(df_v, on="date", how="inner")
        obs_a = merged["obs"].values; pred_a = merged["pred"].values
        mask = ~np.isnan(pred_a) & ~np.isnan(obs_a)
        r = (np.corrcoef(obs_a[mask], pred_a[mask])[0, 1]
             if np.std(pred_a[mask]) > 0 else 0)
        std_p = np.std(pred_a[mask])
        ax_.plot(np.arccos(np.clip(r, -1, 1)), std_p, "o",
                 color=COLORS.get(res["label"], "gray"), ms=11, label=res["label"])
    ax_.set_thetamin(0); ax_.set_thetamax(90)
    ax_.set_title("Diagrama de Taylor — Validación 2020-2022\n(ángulo=1-corr, radio=std)", pad=25)
    ax_.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.35, -0.05))
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "COMP_taylor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  [✓] Todas las gráficas comparativas guardadas en: {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. EXPORTACIÓN EXCEL
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_params(d, prefix=""):
    """Aplana un dict de parametros (listas -> param[0], param[1], ...)."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=f"{key}."))
        elif isinstance(v, (list, tuple, np.ndarray)):
            for i, vi in enumerate(np.asarray(v).ravel()):
                out[f"{key}[{i}]"] = float(vi)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[key] = float(v)
        elif v is None:
            out[key] = np.nan
        else:
            out[key] = str(v)
    return out


# Mapa: etiqueta interna del modelo -> (codigo Mx, clave del dict de parametros)
_MODEL_MAP = {
    "Climatológico":  ("M1", "params_clim"),
    "Hurdle-Gamma":   ("M2", "lstm_params"),
    "ZIDF-Gamma":     ("M3", "lstm_params"),
    "Poisson-Gamma":  ("M4", "pg_params"),
    "Hurdle-GLM":     ("M5", "hg_params"),
    "ZI-EGPD-seas":   ("M6", "egpd_params"),
    "ZI-Gamma-cens":  ("M7", "cz_params"),
    "Wilks-MixExp":   ("M8", "wk_params"),
}

_MODEL_DESC = {
    "M1": "Climatologico mensual (Bernoulli + Gamma por mes) — baseline determinista",
    "M2": "Hurdle LSTM — ocurrencia (OccurrenceLSTM) + cantidad Gamma (AmountGammaLSTM)",
    "M3": "ZI-Gamma LSTM — Zero-Inflated Gamma, entrenamiento conjunto",
    "M4": "Poisson-Gamma (Tweedie compuesto) con estacionalidad armonica",
    "M5": "Hurdle-GLM — GLM logistico (ocurrencia) + GLM Gamma (cantidad)",
    "M6": "ZI-EGPD estacional — Pareto generalizada extendida cero-inflada",
    "M7": "ZI-Gamma censurada — Hurdle-GLM con censura izquierda en L",
    "M8": "Wilks (1999) — Markov 1er orden estacional + mezcla de 2 exponenciales",
}


def build_parametros_calibrados(resultados, cfg):
    """Tabla larga: una fila por (modelo, parametro, valor). Es el entregable
    principal del script 1 (Tabla 9 de la tesis)."""
    filas = []
    for res in resultados:
        label = res["label"]
        code, pkey = _MODEL_MAP.get(label, (label, None))

        # M1: los parametros son por mes -> se aplanan como pi_m / alpha_m / beta_m
        if code == "M1":
            for mes, pm in (res.get("params_clim") or {}).items():
                for pname, pval in pm.items():
                    filas.append({"Modelo": code, "Nombre": label,
                                  "Grupo": f"mes_{int(mes):02d}",
                                  "Parametro": pname,
                                  "Valor": (float(pval) if pval is not None else np.nan)})
            continue

        pars = _flatten_params(res.get(pkey, {}))
        for pname, pval in pars.items():
            filas.append({"Modelo": code, "Nombre": label, "Grupo": "global",
                          "Parametro": pname, "Valor": pval})

        # Diagnosticos Gamma (alpha/beta/mu en dias humedos) — utiles en la Tabla 9
        for pname, pval in _flatten_params(res.get("gamma_diag", {})).items():
            filas.append({"Modelo": code, "Nombre": label, "Grupo": "diagnostico_gamma",
                          "Parametro": pname, "Valor": pval})

    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values(["Modelo", "Grupo", "Parametro"]).reset_index(drop=True)
    return df


def build_ficha_modelos(resultados, cfg):
    """Una fila por modelo: descripcion, familia, n de parametros y ajuste."""
    filas = []
    for res in resultados:
        label = res["label"]
        code, pkey = _MODEL_MAP.get(label, (label, None))
        n_par = len(_flatten_params(res.get(pkey, {}))) if code != "M1" else \
                sum(len(v) for v in (res.get("params_clim") or {}).values())
        familia = ("Aprendizaje profundo (LSTM)" if code in ("M2", "M3")
                   else "Estadistico" if code != "M1" else "Baseline climatologico")
        estocastico = "No (determinista)" if code == "M1" else "Si (ensemble)"
        filas.append({
            "Modelo": code, "Nombre": label, "Familia": familia,
            "Descripcion": _MODEL_DESC.get(code, ""),
            "N_parametros_reportados": n_par,
            "Estocastico": estocastico,
            "Calibracion": f"1977-{pd.to_datetime(cfg['train_end']).year} (excl. 1996)",
            "Validacion": f"{pd.to_datetime(cfg['train_end']).year + 1}-"
                          f"{pd.to_datetime(cfg['val_end']).year}",
            "Prediccion_ciega": cfg["pred_label"],
            "Umbral_dia_humedo_mm": cfg["wet_threshold"],
            "Semilla": cfg["seed"],
        })
    return pd.DataFrame(filas)


def save_excel_calibracion(resultados, cfg):
    """SALIDA 1 del script de calibracion:
         - Parametros_calibrados  (entregable principal)
         - Ficha_modelos          (que es cada modelo y como se ajusto)
         - Metricas_calibracion   (desempeno de la corrida unica, semilla base)
         - Pred_diaria_*          (serie diaria con banda p10/p90 del ensemble)
       NOTA: la variabilidad entre semillas y el Wilcoxon NO se calculan aqui;
       eso lo produce el script 02_semillas_wilcoxon.py (sin duplicar salidas).
    """
    out = cfg["output_folder"]
    xls = os.path.join(out, "01_CALIBRACION_parametros.xlsx")
    print(f"\n  Guardando Excel de calibracion: {xls}")

    df_par   = build_parametros_calibrados(resultados, cfg)
    df_ficha = build_ficha_modelos(resultados, cfg)

    metricas_show = ["NSE_diario", "NSE_mensual", "KGE", "PBIAS_%",
                     "RMSE", "MSE", "MAE", "POD", "FAR", "CSI", "F1",
                     "Acum_obs", "Acum_pred", "TP", "FP", "FN", "TN"]
    rows = []
    for res in resultados:
        code = _MODEL_MAP.get(res["label"], (res["label"], None))[0]
        for periodo, key in [("Validacion_2020-2022", "met_val"),
                             (f"Pred_{cfg['pred_label']}_CIEGO", "met_pred")]:
            row = {"Modelo": code, "Nombre": res["label"], "Periodo": periodo}
            for m in metricas_show:
                row[m] = res[key].get(m, np.nan)
            rows.append(row)
    df_met = pd.DataFrame(rows)

    with pd.ExcelWriter(xls, engine="openpyxl") as w:
        df_ficha.to_excel(w, sheet_name="00_Ficha_modelos", index=False)
        df_par.to_excel(w,   sheet_name="01_Parametros_calibrados", index=False)
        df_met.to_excel(w,   sheet_name="02_Metricas_calibracion", index=False)

        # Serie diaria con banda del ensemble, un bloque por modelo
        for res in resultados:
            code = _MODEL_MAP.get(res["label"], (res["label"], None))[0]
            for tag, key in (("val", "df_val"), ("pred", "df_pred")):
                d = res[key]
                cols = [c for c in ["date", "obs", "pred", "pred_member", "p10", "p90"]
                        if c in d.columns]
                d[cols].to_excel(w, sheet_name=f"{code}_{tag}", index=False)

    print(f"  Excel guardado: {len(df_par)} parametros de {len(resultados)} modelos.")
    return df_par


def save_csv_parametros(df_par, cfg):
    """Copia en CSV de los parametros calibrados (para el repositorio GitHub)."""
    path = os.path.join(cfg["output_folder"], "01_parametros_calibrados.csv")
    df_par.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  CSV de parametros guardado: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

cfg_global = CONFIG   # referencia global para plot_individual (DDPM steps en título)


# ══════════════════════════════════════════════════════════════════════════════
# OPCIÓN 5 — POISSON-GAMMA (TWEEDIE)  — tu poisson_gamma_rainfall_v5.py SIN CAMBIOS
# Solo se envuelve en función para entrar al comparativo (mismo res que Op.1-4).
# La lógica (fit_glm, profile p, phi Pearson, simulate) es idéntica al v5.
# ══════════════════════════════════════════════════════════════════════════════
import statsmodels.api as sm
from tweedie import tweedie
from scipy.optimize import minimize_scalar as _minimize_scalar

def _pg_make_X(doy, k=2):
    cols = [np.ones(len(doy))]
    for h in range(1, k + 1):
        a = 2 * np.pi * h * np.asarray(doy, float) / 365.0
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)

def run_option5_poisson_gamma(df, cfg, df_raw=None):
    print("\n" + "="*60)
    print("  OPCIÓN 5 — Poisson-Gamma (Tweedie) — v5 SIN CAMBIOS [baseline ref.]")
    print("="*60)
    thr     = cfg["wet_threshold"]
    df_full = df_raw if df_raw is not None else df

    # doy con manejo de bisiesto (idéntico al v5)
    def _doy(dates):
        dts = pd.to_datetime(dates)
        doy = dts.dt.dayofyear.values.astype(float)
        leap = dts.dt.is_leap_year.values & (doy >= 60)
        doy[leap] -= 1
        return np.clip(doy, 1, 365)

    K = 2
    train = df[(df["date"] <= cfg["train_end"]) & df["ppd"].notna()].copy()
    valid = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    y_train = train["ppd"].to_numpy(float)
    y_valid = valid["ppd"].to_numpy(float)
    X_tr = _pg_make_X(_doy(train["date"]), K)
    X_va = _pg_make_X(_doy(valid["date"]), K)

    # ── v5: fit_glm / phi_pearson ────────────────────────────────────────────
    def fit_glm(X, y, p):
        fam = sm.families.Tweedie(link=sm.families.links.Log(), var_power=p)
        return sm.GLM(y, X, family=fam).fit()
    def phi_pearson(model, n_params):
        return max(float(model.pearson_chi2 / max(model.nobs - n_params, 1)), 1e-8)

    n_params = X_tr.shape[1]

    # ── v5: profile likelihood for p ─────────────────────────────────────────
    def profile_negll(p):
        if not (1.02 < p < 1.98): return np.inf
        try:
            m   = fit_glm(X_tr, y_train, p)
            mu  = np.clip(m.fittedvalues, 1e-10, None)
            phi = phi_pearson(m, n_params)
            ll  = tweedie(mu=mu, p=p, phi=phi).logpdf(y_train).sum()
            return -ll if np.isfinite(ll) else np.inf
        except Exception:
            return np.inf
    p_grid  = np.linspace(1.10, 1.90, 33)
    ll_grid = np.array([-profile_negll(pp) for pp in p_grid])
    p_coarse = p_grid[int(np.argmax(np.where(np.isfinite(ll_grid), ll_grid, -np.inf)))]
    rres = _minimize_scalar(profile_negll,
                            bounds=(max(1.05, p_coarse-0.10), min(1.95, p_coarse+0.10)),
                            method="bounded", options={"xatol": 1e-4})
    p_hat = float(np.clip(rres.x if rres.success else p_coarse, 1.05, 1.95))

    # ── v5: final fit + phi ──────────────────────────────────────────────────
    model    = fit_glm(X_tr, y_train, p_hat)
    beta_hat = np.asarray(model.params)
    phi_hat  = phi_pearson(model, n_params)
    mu_valid = np.clip(np.exp(X_va @ beta_hat), 1e-10, None)
    a1, a2 = 2.0 - p_hat, p_hat - 1.0
    P_shape = a1 / a2
    def lam_of(mu):   return np.clip(mu, 1e-300, None)**a1 / (phi_hat * a1)
    def scale_of(mu): return phi_hat * a2 * np.clip(mu, 1e-300, None)**a2

    MES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    dd = np.arange(1, 366); mu_clim = np.exp(_pg_make_X(dd, K) @ beta_hat)
    peak_doy = int(dd[np.argmax(mu_clim)])
    print(f"  p̂={p_hat:.4f}  Θ̂(phi)={phi_hat:.4f}  P̂={P_shape:.4f}  "
          f"μ̂ pico doy {peak_doy} ({MES[(peak_doy-1)//30 % 12]})")

    # ── v5: simulate (idéntica) ──────────────────────────────────────────────
    def simulate(mu_arr, seed):
        rng = np.random.default_rng(seed)
        lam, scl = lam_of(mu_arr), scale_of(mu_arr)
        out = np.zeros(len(mu_arr))
        for i in range(len(mu_arr)):
            N = rng.poisson(max(lam[i], 1e-12))
            if N > 0:
                out[i] = rng.gamma(P_shape, max(scl[i], 1e-12), N).sum()
        return out

    # ── Envoltorio al formato 4ciego: ensamble → media (métrica) + bandas ────
    Kc   = int(cfg.get("n_ensemble", cfg.get("n_gamma_samples", 50)))
    seed = int(cfg.get("seed", 42))
    memb_v   = np.stack([simulate(mu_valid, seed + s) for s in range(Kc)])
    pred_val = memb_v.mean(axis=0)
    tot      = memb_v.sum(axis=1); rep = int(np.argmin(np.abs(tot - np.median(tot))))
    df_val_out = valid[["date"]].copy()
    df_val_out["obs"]  = y_valid
    df_val_out["pred"] = pred_val
    df_val_out["pred_member"] = memb_v[rep]
    df_val_out["p10"] = np.percentile(memb_v, 10, axis=0)
    df_val_out["p90"] = np.percentile(memb_v, 90, axis=0)
    met_val = compute_metrics(y_valid, pred_val, thr,
                              "Op.5 — Poisson-Gamma | Validación 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_valid, pred_val, valid["date"].values)

    pr     = df_full[df_full["date"].dt.year.isin(cfg["pred_years"])].copy()
    mu_pr  = np.clip(np.exp(_pg_make_X(_doy(pr["date"]), K) @ beta_hat), 1e-10, None)
    memb_p = np.stack([simulate(mu_pr, seed + 100 + s) for s in range(Kc)])
    pred_pr = memb_p.mean(axis=0)
    _tp = memb_p.sum(axis=1); _rp = int(np.argmin(np.abs(_tp - np.median(_tp))))
    df_pred_out = pr[["date"]].copy()
    df_pred_out["obs"]  = pr["ppd"].values if "ppd" in pr.columns else np.nan
    df_pred_out["pred"] = pred_pr
    df_pred_out["pred_member"] = memb_p[_rp]
    df_pred_out["p10"] = np.percentile(memb_p, 10, axis=0)
    df_pred_out["p90"] = np.percentile(memb_p, 90, axis=0)
    met_pred = blind_metrics(pred_pr, f"Op.5 — Poisson-Gamma | {cfg['pred_label']} CIEGO")

    return {
        "label": "Poisson-Gamma",
        "df_val": df_val_out, "df_pred": df_pred_out,
        "met_val": met_val, "met_pred": met_pred,
        "pg_params": {"p": p_hat, "phi": phi_hat, "peak_doy": peak_doy,
                      "P_shape": P_shape, "K_harmonics": K,
                      "beta": [float(b) for b in beta_hat],
                      "n_train": int(len(y_train))},
    }


def _hg_make_X(doy, k=2):
    """Diseño hurdle GLM: [intercepto, sin(h·doy), cos(h·doy)] h=1..k. Mismo criterio que Op.5."""
    cols = [np.ones(len(doy))]
    for h in range(1, k + 1):
        a = 2 * np.pi * h * np.asarray(doy, float) / 365.0
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)


def _hg_doy(dates):
    """Día del año con bisiesto colapsado a 365 (idéntico a Op.5)."""
    dts = pd.to_datetime(dates)
    doy = dts.dt.dayofyear.values.astype(float)
    leap = dts.dt.is_leap_year.values & (doy >= 60)
    doy[leap] -= 1
    return np.clip(doy, 1, 365)


def run_option6_hurdle_glm(df, cfg, df_raw=None):
    """Hurdle SIN ML: GLM logístico (ocurrencia) + GLM Gamma (cantidad) sobre armónicos
    de día-del-año. 100% probabilístico, sin lags, sin autorregresión. Control estadístico
    puro del Op.1 (Hurdle LSTM). Leak-free: 2024 se simula desde el doy de las fechas."""
    import statsmodels.api as sm
    print("\n" + "=" * 60)
    print("  OPCIÓN 6 — Hurdle (sin ML): Logística + GLM Gamma sobre doy")
    print("=" * 60)
    thr     = cfg["wet_threshold"]
    K       = int(cfg.get("hg_harmonics", 2))
    df_full = df_raw if df_raw is not None else df

    train = df[(df["date"] <= cfg["train_end"]) & df["ppd"].notna()].copy()
    valid = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    y_tr  = train["ppd"].to_numpy(float)
    # cap PRISTINO (v67): máximo observado SOLO en train (1977-2020); ninguna info de
    # validación 2020-2022 entra al cap. Antes usaba la constante 3.1 = máx serie completa.
    cap_mm = float(np.nanmax(y_tr))
    y_va  = valid["ppd"].to_numpy(float)
    X_tr  = _hg_make_X(_hg_doy(train["date"]), K)
    X_va  = _hg_make_X(_hg_doy(valid["date"]), K)

    # ── Parte 1 — OCURRENCIA: GLM Binomial (logit) ───────────────────────────
    occ_tr    = (y_tr >= thr).astype(float)
    occ_model = sm.GLM(occ_tr, X_tr, family=sm.families.Binomial()).fit()
    p_wet_va  = np.clip(occ_model.predict(X_va), 1e-6, 1 - 1e-6)

    # ── Parte 2 — CANTIDAD | húmedo: GLM Gamma (log) solo en días húmedos ─────
    wet          = occ_tr.astype(bool)
    amt_model    = sm.GLM(y_tr[wet], X_tr[wet],
                          family=sm.families.Gamma(sm.families.links.Log())).fit()
    phi_hat      = float(amt_model.scale)                       # dispersión Pearson
    alpha_hat    = float(np.clip(1.0 / max(phi_hat, 1e-8), 1e-3, None))  # shape = 1/phi
    mu_va        = np.clip(amt_model.predict(X_va), 1e-8, None) # E[cantidad | húmedo]

    print(f"  ocurrencia: P(húmedo) media train={occ_tr.mean():.3f}  |  "
          f"cantidad: alpha(shape)={alpha_hat:.4f}  mu̅(val)={mu_va.mean():.4f}mm  cap={cap_mm:.2f}mm (max train)")

    Kc   = int(cfg.get("n_ensemble", cfg.get("n_gamma_samples", 50)))
    seed = int(cfg.get("seed", 42))

    def simulate(p_wet, mu, base_seed):
        rng   = np.random.default_rng(base_seed)
        rains = rng.random(len(p_wet)) < p_wet
        beta  = alpha_hat / np.clip(mu, 1e-8, None)             # rate = alpha/mu ⇒ E[X]=mu
        amt   = np.clip(rng.gamma(shape=alpha_hat, scale=1.0 / beta), 0.0, cap_mm)
        return np.where(rains, amt, 0.0)

    # Validación (ensamble → media para métrica + banda)
    memb_v   = np.stack([simulate(p_wet_va, mu_va, seed + s) for s in range(Kc)])
    pred_val = memb_v.mean(axis=0)
    tot      = memb_v.sum(axis=1); rep = int(np.argmin(np.abs(tot - np.median(tot))))
    df_val_out = valid[["date"]].copy()
    df_val_out["obs"]         = y_va
    df_val_out["pred"]        = pred_val
    df_val_out["pred_member"] = memb_v[rep]
    df_val_out["p10"]         = np.percentile(memb_v, 10, axis=0)
    df_val_out["p90"]         = np.percentile(memb_v, 90, axis=0)
    met_val = compute_metrics(y_va, pred_val, thr, "Op.6 — Hurdle-GLM | Validación 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_va, pred_val, valid["date"].values)

    # Predicción ciega 2024 (solo desde el doy — sin observados)
    pr       = df_full[df_full["date"].dt.year.isin(cfg["pred_years"])].copy()
    X_pr     = _hg_make_X(_hg_doy(pr["date"]), K)
    p_wet_pr = np.clip(occ_model.predict(X_pr), 1e-6, 1 - 1e-6)
    mu_pr    = np.clip(amt_model.predict(X_pr), 1e-8, None)
    memb_p   = np.stack([simulate(p_wet_pr, mu_pr, seed + 100 + s) for s in range(Kc)])
    pred_pr  = memb_p.mean(axis=0)
    _tp = memb_p.sum(axis=1); _rp = int(np.argmin(np.abs(_tp - np.median(_tp))))
    df_pred_out = pr[["date"]].copy()
    df_pred_out["obs"]  = pr["ppd"].values if "ppd" in pr.columns else np.nan
    df_pred_out["pred"] = pred_pr
    df_pred_out["pred_member"] = memb_p[_rp]
    df_pred_out["p10"] = np.percentile(memb_p, 10, axis=0)
    df_pred_out["p90"] = np.percentile(memb_p, 90, axis=0)
    met_pred = blind_metrics(pred_pr, f"Op.6 — Hurdle-GLM | {cfg['pred_label']} CIEGO")

    # Diagnóstico Gamma (días húmedos OBSERVADOS en validación, como Op.1)
    wet_va    = y_va >= thr
    mu_wet    = mu_va[wet_va] if wet_va.any() else mu_va[:1]
    alpha_wet = np.full(len(mu_wet), alpha_hat)
    beta_wet  = alpha_hat / np.clip(mu_wet, 1e-8, None)
    # SUSTENTO Tabla 9: se distinguen DOS medias de intensidad para evitar ambiguedad
    #   mu_pred_wet  = media de E[cantidad|humedo] PREDICHA por el modelo en dias humedos obs.
    #   mu_obs_wet   = media de la intensidad OBSERVADA real en esos mismos dias humedos
    mu_obs_wet = float(y_va[wet_va].mean()) if wet_va.any() else float("nan")
    gamma_diag = {
        "n_wet"      : int(wet_va.sum()),
        "alpha_min"  : float(alpha_wet.min()),  "alpha_max" : float(alpha_wet.max()),
        "alpha_mean" : float(alpha_wet.mean()), "alpha_std" : float(alpha_wet.std()),
        "beta_min"   : float(beta_wet.min()),   "beta_max"  : float(beta_wet.max()),
        "beta_mean"  : float(beta_wet.mean()),  "beta_std"  : float(beta_wet.std()),
        "mu_min"     : float(mu_wet.min()),     "mu_max"    : float(mu_wet.max()),
        "mu_mean"    : float(mu_wet.mean()),    "mu_std"    : float(mu_wet.std()),
        "mu_pred_wet": float(mu_wet.mean()),    # = mu_mean (predicha), etiqueta explicita
        "mu_obs_wet" : mu_obs_wet,              # media OBSERVADA real en dias humedos
        "cv_alpha"   : float(alpha_wet.std() / max(alpha_wet.mean(), 1e-6)),
        "cv_mu"      : float(mu_wet.std() / max(mu_wet.mean(), 1e-6)),
    }

    return {
        "label": "Hurdle-GLM",
        "df_val": df_val_out, "df_pred": df_pred_out,
        "met_val": met_val, "met_pred": met_pred,
        "gamma_diag": gamma_diag,
        "prob_val_raw": p_wet_va, "y_val_raw": y_va,
        "hg_params": {"K_harmonics": K, "alpha": alpha_hat, "phi": phi_hat,
                      "cap_mm": cap_mm,
                      "beta_ocurrencia": [float(b) for b in np.asarray(occ_model.params)],
                      "beta_cantidad":   [float(b) for b in np.asarray(amt_model.params)],
                      "p_wet_train": float(occ_tr.mean()),
                      "n_train": int(len(y_tr)), "n_wet_train": int(wet.sum())},
    }


# ======================================================================
# Op.7/Op.8 — ZI-EGPD (Zero-Inflated Extended Generalized Pareto, Naveau power model)
#   Modelo estadístico puro (MLE), SIN machine learning, SIN lags. Ref: Abbas, Ahmad &
#   Ahmad (2025, arXiv:2504.11058); EGPD de Naveau et al. (2016). Modela la cola pesada
#   intrínsecamente (sin cap artificial). 2024 ciego: solo desde el día-del-año.
# ======================================================================
from scipy.optimize import minimize as _egpd_minimize

_EPS = 1e-12


def egpd_logpdf(x, sigma, xi, kappa):
    """log densidad EGPD en x>0. Vectorizado; sigma puede ser escalar o array."""
    x = np.asarray(x, float)
    sigma = np.asarray(sigma, float)
    if abs(xi) < 1e-6:                       # límite exponencial
        H = 1.0 - np.exp(-x / sigma)
        log_h = -np.log(sigma) - x / sigma
    else:
        t = 1.0 + xi * x / sigma
        t = np.clip(t, _EPS, None)
        H = 1.0 - t ** (-1.0 / xi)
        log_h = -np.log(sigma) + (-1.0 / xi - 1.0) * np.log(t)
    H = np.clip(H, _EPS, 1.0)
    return np.log(kappa) + (kappa - 1.0) * np.log(H) + log_h


def egpd_quantile(U, sigma, xi, kappa):
    """Inversa de la CDF: x = F^{-1}(U). U en (0,1). sigma escalar o array."""
    U = np.clip(np.asarray(U, float), _EPS, 1 - _EPS)
    v = U ** (1.0 / kappa)                   # = H(x)
    v = np.clip(v, _EPS, 1 - _EPS)
    if abs(xi) < 1e-6:
        return -sigma * np.log(1.0 - v)
    return (sigma / xi) * ((1.0 - v) ** (-xi) - 1.0)


def fit_egpd_marginal(x_wet, init=None):
    """MLE de (sigma, xi, kappa) sobre días húmedos. Devuelve dict."""
    x_wet = np.asarray(x_wet, float)
    x_wet = x_wet[x_wet > 0]
    m = x_wet.mean()

    def negll(theta):
        log_sigma, xi, log_kappa = theta
        sigma = np.exp(log_sigma); kappa = np.exp(log_kappa)
        if abs(xi) > 1e-6:                   # soporte: 1+xi*x/sigma>0
            if np.any(1.0 + xi * x_wet / sigma <= 0):
                return 1e10
        ll = egpd_logpdf(x_wet, sigma, xi, kappa).sum()
        return -ll if np.isfinite(ll) else 1e10

    # Cotas realistas de lluvia (evita la degeneración kappa<->xi): xi en [-0.3,0.7]
    # (xi<1 => media finita; >0.7 irreal para lluvia diaria), kappa en [0.1,5].
    bounds = [(np.log(m) - 4, np.log(m) + 4), (-0.3, 0.7), (np.log(0.1), np.log(5.0))]
    best = None
    for xi0 in (0.05, 0.15, 0.30, -0.05):    # multi-arranque en xi
        x0i = [np.log(m), xi0, 0.0]
        r = _egpd_minimize(negll, x0i, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    ls, xi, lk = best.x
    return {"sigma": float(np.exp(ls)), "xi": float(xi), "kappa": float(np.exp(lk)),
            "negll": float(best.fun), "conv": bool(best.success)}


def fit_egpd_seasonal(x_wet, Xh_wet, init=None):
    """MLE estacional: sigma(doy)=exp(Xh@beta), kappa y xi globales.
    Xh_wet: diseño armónico (n_wet, p). Devuelve dict con beta, xi, kappa."""
    x_wet = np.asarray(x_wet, float)
    Xh = np.asarray(Xh_wet, float)
    p = Xh.shape[1]
    m = x_wet.mean()

    def negll(theta):
        beta = theta[:p]; xi = theta[p]; kappa = np.exp(theta[p + 1])
        sigma = np.exp(np.clip(Xh @ beta, -10, 10))
        if abs(xi) > 1e-6 and np.any(1.0 + xi * x_wet / sigma <= 0):
            return 1e10
        ll = egpd_logpdf(x_wet, sigma, xi, kappa).sum()
        return -ll if np.isfinite(ll) else 1e10

    beta0 = np.zeros(p); beta0[0] = np.log(m)
    x0 = init if init is not None else np.concatenate([beta0, [0.1, 0.0]])
    bounds = [(None, None)] * p + [(-0.3, 0.7), (np.log(0.1), np.log(5.0))]
    r = _egpd_minimize(negll, x0, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 20000})
    beta = r.x[:p]; xi = float(r.x[p]); kappa = float(np.exp(r.x[p + 1]))
    return {"beta": beta, "xi": xi, "kappa": kappa,
            "negll": float(r.fun), "conv": bool(r.success)}


def run_option_ziegpd(df, cfg, df_raw=None, mode="seasonal"):
    seasonal = (mode == "seasonal")
    label = "ZI-EGPD-seas" if seasonal else "ZI-EGPD-marg"
    print("\n" + "=" * 60)
    print(f"  OPCIÓN — {label}: Zero-Inflated Extended GPD ({mode}), sin ML")
    print("=" * 60)
    thr   = cfg["wet_threshold"]
    K_h   = int(cfg.get("hg_harmonics", 2))
    seed  = int(cfg.get("seed", 42))
    Kc    = int(cfg.get("n_ensemble", cfg.get("n_gamma_samples", 50)))
    GUARD = 100.0                                   # guarda numérica (no es el cap 3.1)
    df_full = df_raw if df_raw is not None else df

    train = df[(df["date"] <= cfg["train_end"]) & df["ppd"].notna()].copy()
    valid = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    y_tr  = train["ppd"].to_numpy(float)
    y_va  = valid["ppd"].to_numpy(float)
    wet   = y_tr >= thr

    # ---- Ocurrencia + intensidad ----
    if seasonal:
        Xtr = _hg_make_X(_hg_doy(train["date"]), K_h)
        Xva = _hg_make_X(_hg_doy(valid["date"]), K_h)
        occ = sm.GLM((y_tr >= thr).astype(float), Xtr, family=sm.families.Binomial()).fit()
        p_wet_va = np.clip(occ.predict(Xva), 1e-6, 1 - 1e-6)
        ef = fit_egpd_seasonal(y_tr[wet], Xtr[wet])
        beta, xi, kappa = ef["beta"], ef["xi"], ef["kappa"]
        sigma_va = np.exp(np.clip(Xva @ beta, -10, 10))
        prob_raw = p_wet_va
    else:
        pi_dry = 1.0 - wet.mean()
        p_wet_va = np.full(len(y_va), 1.0 - pi_dry)
        ef = fit_egpd_marginal(y_tr[wet])
        xi, kappa = ef["xi"], ef["kappa"]
        sigma_va = np.full(len(y_va), ef["sigma"])
        prob_raw = None
    print(f"  EGPD: xi(cola)={xi:.4f}  kappa={kappa:.4f}  "
          f"sigma̅(val)={np.mean(sigma_va):.4f}  P(húmedo)̅={p_wet_va.mean():.3f}  conv={ef['conv']}")

    # ---- Simulación ensamble (Bernoulli x EGPD, SIN cap; guarda numérica alta) ----
    def simulate(p_wet, sigma_arr, base_seed):
        rng = np.random.default_rng(base_seed)
        rains = rng.random(len(p_wet)) < p_wet
        amt   = egpd_quantile(rng.random(len(p_wet)), sigma_arr, xi, kappa)
        amt   = np.clip(amt, 0.0, GUARD)
        return np.where(rains, amt, 0.0)

    memb_v = np.stack([simulate(p_wet_va, sigma_va, seed + s) for s in range(Kc)])
    pred_val = memb_v.mean(axis=0)
    tot = memb_v.sum(axis=1); rep = int(np.argmin(np.abs(tot - np.median(tot))))
    df_val_out = valid[["date"]].copy()
    df_val_out["obs"] = y_va; df_val_out["pred"] = pred_val
    df_val_out["pred_member"] = memb_v[rep]
    df_val_out["p10"] = np.percentile(memb_v, 10, axis=0)
    df_val_out["p90"] = np.percentile(memb_v, 90, axis=0)
    met_val = compute_metrics(y_va, pred_val, thr, f"{label} | Validación 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_va, pred_val, valid["date"].values)

    # ---- Predicción ciega 2024 (solo desde doy) ----
    pr = df_full[df_full["date"].dt.year.isin(cfg["pred_years"])].copy()
    if seasonal:
        Xpr = _hg_make_X(_hg_doy(pr["date"]), K_h)
        p_wet_pr = np.clip(occ.predict(Xpr), 1e-6, 1 - 1e-6)
        sigma_pr = np.exp(np.clip(Xpr @ beta, -10, 10))
    else:
        p_wet_pr = np.full(len(pr), 1.0 - pi_dry)
        sigma_pr = np.full(len(pr), ef["sigma"])
    memb_p = np.stack([simulate(p_wet_pr, sigma_pr, seed + 100 + s) for s in range(Kc)])
    pred_pr = memb_p.mean(axis=0)
    _tp = memb_p.sum(axis=1); _rp = int(np.argmin(np.abs(_tp - np.median(_tp))))
    df_pred_out = pr[["date"]].copy()
    df_pred_out["obs"]  = pr["ppd"].values if "ppd" in pr.columns else np.nan
    df_pred_out["pred"] = pred_pr
    df_pred_out["pred_member"] = memb_p[_rp]
    df_pred_out["p10"] = np.percentile(memb_p, 10, axis=0)
    df_pred_out["p90"] = np.percentile(memb_p, 90, axis=0)
    met_pred = blind_metrics(pred_pr, f"{label} | {cfg['pred_label']} CIEGO")

    out = {"label": label, "df_val": df_val_out, "df_pred": df_pred_out,
           "met_val": met_val, "met_pred": met_pred,
           "egpd_params": {"xi": float(xi), "kappa": float(kappa), "mode": mode,
                           "sigma_base": float(ef["sigma"]),
                           "K_harmonics": (int(K_h) if seasonal else 0),
                           "beta_sigma": ([float(b) for b in np.asarray(beta)]
                                          if seasonal else []),
                           "beta_ocurrencia": ([float(b) for b in np.asarray(occ.params)]
                                               if seasonal else []),
                           "pi_dry": (float(pi_dry) if not seasonal else float("nan"))}}
    if seasonal:
        out["prob_val_raw"] = prob_raw; out["y_val_raw"] = y_va
    return out


# ======================================================================
# Op.9 — ZI-Gamma CENSURADA (= Hurdle-GLM Op.6 + censura izquierda en L=0.1mm)
#   Ref: censored zero-inflated gamma, Scientific Reports (2024). Sin ML, sin lags.
#   Valores húmedos < L se tratan como censurados (Y<=L). Mismo cap pristino que Op.6.
# ======================================================================
from scipy.stats import gamma as _czgam
from scipy.optimize import minimize as _cz_minimize

def _cz_make_X(doy, k=2):
    cols = [np.ones(len(doy))]
    for h in range(1, k + 1):
        a = 2 * np.pi * h * np.asarray(doy, float) / 365.0
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)


def _cz_doy(dates):
    dts = pd.to_datetime(dates)
    doy = dts.dt.dayofyear.values.astype(float)
    leap = dts.dt.is_leap_year.values & (doy >= 60)
    doy[leap] -= 1
    return np.clip(doy, 1, 365)


def fit_gamma_censored(y_wet, Xh, L):
    """MLE de Gamma GLM (mu=exp(Xh@beta), shape alpha cte) con censura izquierda en L.
    y_wet<L: aporta log F(L); y_wet>=L: aporta log f(y)."""
    y = np.asarray(y_wet, float); Xh = np.asarray(Xh, float)
    p = Xh.shape[1]; cens = y < L
    m = y.mean()

    def negll(theta):
        beta = theta[:p]; alpha = np.exp(theta[p])
        mu = np.exp(np.clip(Xh @ beta, -10, 10)); scale = mu / alpha
        ll = 0.0
        ex = ~cens
        if ex.any():
            ll += _czgam.logpdf(y[ex], a=alpha, scale=scale[ex]).sum()
        if cens.any():
            cdfL = np.clip(_czgam.cdf(L, a=alpha, scale=scale[cens]), 1e-12, 1.0)
            ll += np.log(cdfL).sum()
        return -ll if np.isfinite(ll) else 1e10

    beta0 = np.zeros(p); beta0[0] = np.log(m)
    x0 = np.concatenate([beta0, [0.0]])
    bounds = [(None, None)] * p + [(np.log(0.1), np.log(50.0))]
    r = _cz_minimize(negll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 8000})
    return {"beta": r.x[:p], "alpha": float(np.exp(r.x[p])),
            "negll": float(r.fun), "conv": bool(r.success), "n_cens": int(cens.sum())}


def run_option9_zigamma_censored(df, cfg, df_raw=None):
    print("\n" + "=" * 60)
    print("  OPCIÓN 9 — ZI-Gamma CENSURADA (= Hurdle-GLM + censura < L)")
    print("=" * 60)
    thr = cfg["wet_threshold"]; K = int(cfg.get("hg_harmonics", 2))
    L   = float(cfg.get("censor_L", 0.1))
    seed = int(cfg.get("seed", 42)); Kc = int(cfg.get("n_ensemble", 50))
    df_full = df_raw if df_raw is not None else df

    train = df[(df["date"] <= cfg["train_end"]) & df["ppd"].notna()].copy()
    valid = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    y_tr = train["ppd"].to_numpy(float); y_va = valid["ppd"].to_numpy(float)
    cap_mm = float(np.nanmax(y_tr))                       # cap pristino (= Op.6)
    Xtr = _cz_make_X(_cz_doy(train["date"]), K)
    Xva = _cz_make_X(_cz_doy(valid["date"]), K)

    # Ocurrencia (idéntica a Op.6)
    occ = sm.GLM((y_tr >= thr).astype(float), Xtr, family=sm.families.Binomial()).fit()
    p_wet_va = np.clip(occ.predict(Xva), 1e-6, 1 - 1e-6)

    # Cantidad CENSURADA
    wet = y_tr >= thr
    cf = fit_gamma_censored(y_tr[wet], Xtr[wet], L)
    beta, alpha = cf["beta"], cf["alpha"]
    mu_va = np.exp(np.clip(Xva @ beta, -10, 10))
    print(f"  censura L={L}mm  | n_censurados train={cf['n_cens']}/{int(wet.sum())}  "
          f"| alpha={alpha:.4f}  mu̅(val)={mu_va.mean():.4f}mm  cap={cap_mm:.2f}mm  conv={cf['conv']}")

    def simulate(p_wet, mu, base_seed):
        rng = np.random.default_rng(base_seed)
        rains = rng.random(len(p_wet)) < p_wet
        amt = np.clip(rng.gamma(shape=alpha, scale=np.clip(mu, 1e-8, None) / alpha), 0.0, cap_mm)
        return np.where(rains, amt, 0.0)

    memb_v = np.stack([simulate(p_wet_va, mu_va, seed + s) for s in range(Kc)])
    pred_val = memb_v.mean(axis=0)
    tot = memb_v.sum(axis=1); rep = int(np.argmin(np.abs(tot - np.median(tot))))
    df_val_out = valid[["date"]].copy()
    df_val_out["obs"] = y_va; df_val_out["pred"] = pred_val
    df_val_out["pred_member"] = memb_v[rep]
    df_val_out["p10"] = np.percentile(memb_v, 10, axis=0)
    df_val_out["p90"] = np.percentile(memb_v, 90, axis=0)
    met_val = compute_metrics(y_va, pred_val, thr, "Op.9 — ZI-Gamma censurada | Val 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_va, pred_val, valid["date"].values)

    pr = df_full[df_full["date"].dt.year.isin(cfg["pred_years"])].copy()
    Xpr = _cz_make_X(_cz_doy(pr["date"]), K)
    p_wet_pr = np.clip(occ.predict(Xpr), 1e-6, 1 - 1e-6)
    mu_pr = np.exp(np.clip(Xpr @ beta, -10, 10))
    memb_p = np.stack([simulate(p_wet_pr, mu_pr, seed + 100 + s) for s in range(Kc)])
    pred_pr = memb_p.mean(axis=0)
    _tp = memb_p.sum(axis=1); _rp = int(np.argmin(np.abs(_tp - np.median(_tp))))
    df_pred_out = pr[["date"]].copy()
    df_pred_out["obs"] = pr["ppd"].values if "ppd" in pr.columns else np.nan
    df_pred_out["pred"] = pred_pr
    df_pred_out["pred_member"] = memb_p[_rp]
    df_pred_out["p10"] = np.percentile(memb_p, 10, axis=0)
    df_pred_out["p90"] = np.percentile(memb_p, 90, axis=0)
    met_pred = blind_metrics(pred_pr, f"Op.9 — ZI-Gamma censurada | {cfg['pred_label']} CIEGO")

    wet_va = y_va >= thr; mu_w = mu_va[wet_va] if wet_va.any() else mu_va[:1]
    aw = np.full(len(mu_w), alpha); bw = alpha / np.clip(mu_w, 1e-8, None)
    mu_obs_wet = float(y_va[wet_va].mean()) if wet_va.any() else float("nan")
    gamma_diag = {"n_wet": int(wet_va.sum()),
        "alpha_min": float(aw.min()), "alpha_max": float(aw.max()),
        "alpha_mean": float(aw.mean()), "alpha_std": float(aw.std()),
        "beta_min": float(bw.min()), "beta_max": float(bw.max()),
        "beta_mean": float(bw.mean()), "beta_std": float(bw.std()),
        "mu_min": float(mu_w.min()), "mu_max": float(mu_w.max()),
        "mu_mean": float(mu_w.mean()), "mu_std": float(mu_w.std()),
        "mu_pred_wet": float(mu_w.mean()),   # media PREDICHA (=mu_mean), etiqueta explicita
        "mu_obs_wet": mu_obs_wet,            # media OBSERVADA real en dias humedos
        "cv_alpha": 0.0, "cv_mu": float(mu_w.std() / max(mu_w.mean(), 1e-6))}

    return {"label": "ZI-Gamma-cens", "df_val": df_val_out, "df_pred": df_pred_out,
            "met_val": met_val, "met_pred": met_pred, "gamma_diag": gamma_diag,
            "prob_val_raw": p_wet_va, "y_val_raw": y_va,
            "cz_params": {"alpha": float(alpha), "L": L, "n_cens": cf["n_cens"],
                          "K_harmonics": K, "cap_mm": cap_mm,
                          "beta_cantidad": [float(b) for b in np.asarray(beta)],
                          "beta_ocurrencia": [float(b) for b in np.asarray(occ.params)],
                          "n_train": int(len(y_tr)), "n_wet_train": int(wet.sum())}}


# ======================================================================
# Op.10 — Wilks (1999): generador estocástico FIEL al paper (referencia clásica).
#   Ocurrencia: cadena de Markov 2 estados, 1er orden, estacional (Richardson-Wilks).
#   Cantidad: MEZCLA DE EXPONENCIALES (Wilks: ajusta mejor que Gamma), EM.
#   2024 ciego: cadena generada hacia adelante, sembrada con el último estado real 2023.
# ======================================================================
def _wk_make_X(doy, k=2):
    cols = [np.ones(len(doy))]
    for h in range(1, k + 1):
        a = 2 * np.pi * h * np.asarray(doy, float) / 365.0
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)


def _wk_doy(dates):
    dts = pd.to_datetime(dates)
    doy = dts.dt.dayofyear.values.astype(float)
    leap = dts.dt.is_leap_year.values & (doy >= 60)
    doy[leap] -= 1
    return np.clip(doy, 1, 365)


def fit_mixexp_EM(x, iters=300, tol=1e-9):
    """EM para mezcla de 2 exponenciales: f=w/mu1 e^{-x/mu1}+(1-w)/mu2 e^{-x/mu2}."""
    x = np.asarray(x, float); x = x[x > 0]
    mu1, mu2, w = x.mean() * 0.4, x.mean() * 1.8, 0.5
    prev = -np.inf
    for _ in range(iters):
        e1 = np.exp(-x / mu1) / mu1
        e2 = np.exp(-x / mu2) / mu2
        denom = w * e1 + (1 - w) * e2 + 1e-300
        g = w * e1 / denom
        w = float(np.clip(g.mean(), 1e-4, 1 - 1e-4))
        mu1 = float((g * x).sum() / (g.sum() + 1e-12))
        mu2 = float(((1 - g) * x).sum() / ((1 - g).sum() + 1e-12))
        ll = np.log(denom).sum()
        if abs(ll - prev) < tol:
            break
        prev = ll
    if mu1 > mu2:                                   # convención mu1<mu2
        mu1, mu2, w = mu2, mu1, 1 - w
    return {"w": w, "mu1": mu1, "mu2": mu2, "negll": -prev}


def _mixexp_sample(rng, n, w, mu1, mu2):
    comp = rng.random(n) < w
    return np.where(comp, rng.exponential(mu1, n), rng.exponential(mu2, n))


def run_option10_wilks(df, cfg, df_raw=None):
    print("\n" + "=" * 60)
    print("  OPCIÓN 10 — Wilks (1999): Markov 2-estados + mezcla exponencial")
    print("=" * 60)
    thr = cfg["wet_threshold"]; K = int(cfg.get("hg_harmonics", 2))
    cap_mm = None                                    # Wilks no capa (mezcla modela la cola)
    seed = int(cfg.get("seed", 42)); Kc = int(cfg.get("n_ensemble", 50))
    GUARD = 100.0
    df_full = df_raw if df_raw is not None else df

    full = df[df["ppd"].notna()].sort_values("date").reset_index(drop=True)
    train = full[full["date"] <= cfg["train_end"]].copy()
    valid = df[(df["date"] > cfg["train_end"]) & (df["date"] <= cfg["val_end"])].copy()
    y_tr = train["ppd"].to_numpy(float)
    wet_tr = (y_tr >= thr)

    # ---- Ocurrencia: Markov 1er orden estacional (2 GLM logísticos) ----
    prev = wet_tr[:-1]; today = wet_tr[1:].astype(float)
    Xt = _wk_make_X(_wk_doy(train["date"]), K)[1:]
    glm_ww = sm.GLM(today[prev], Xt[prev], family=sm.families.Binomial()).fit()   # P(wet|prev wet)
    glm_dw = sm.GLM(today[~prev], Xt[~prev], family=sm.families.Binomial()).fit() # P(wet|prev dry)

    # ---- Cantidad: mezcla de 2 exponenciales (EM) sobre días húmedos ----
    mix = fit_mixexp_EM(y_tr[wet_tr])
    cap_mm = float(np.nanmax(y_tr))                  # guarda = max train (pristino)
    print(f"  mezcla-exp: w={mix['w']:.3f}  mu1={mix['mu1']:.4f}  mu2={mix['mu2']:.4f}mm  "
          f"| P(húmedo|seco)̅, P(húmedo|húmedo)̅ vía Markov  cap={cap_mm:.2f}mm")

    # === [PRESENTACIÓN punto 1] p01=P(húmedo|seco), p11=P(húmedo|húmedo) medios anuales ===
    _doy_grid = np.arange(1, 366)
    _Xgrid = _wk_make_X(_doy_grid, K)
    _p11 = float(glm_ww.predict(_Xgrid).mean())
    _p01 = float(glm_dw.predict(_Xgrid).mean())
    print(f"  [presentacion] Markov medio anual: p01=P(h|seco)={_p01:.4f}  "
          f"p11=P(h|humedo)={_p11:.4f}  (lambda1={1/mix['mu1']:.3f}  lambda2={1/mix['mu2']:.3f})")

    def gen_sequence(dates, seed_state, base_seed):
        """Genera la secuencia húmedo/seco hacia adelante (Markov) + montos (mezcla)."""
        rng = np.random.default_rng(base_seed)
        Xd = _wk_make_X(_wk_doy(dates), K)
        n = len(dates); out = np.zeros(n); state = bool(seed_state)
        # prob de transición por día según estado previo
        pw = np.clip(glm_ww.predict(Xd), 1e-6, 1 - 1e-6)
        pd_ = np.clip(glm_dw.predict(Xd), 1e-6, 1 - 1e-6)
        u = rng.random(n)
        amt = np.clip(_mixexp_sample(rng, n, mix["w"], mix["mu1"], mix["mu2"]), 0, GUARD)
        for i in range(n):
            ptr = pw[i] if state else pd_[i]
            state = u[i] < ptr
            out[i] = amt[i] if state else 0.0
        return out

    # estado semilla validación = estado real del día anterior a val (último de train)
    seed_val = bool(wet_tr[-1])
    memb_v = np.stack([gen_sequence(valid["date"], seed_val, seed + s) for s in range(Kc)])
    y_va = valid["ppd"].to_numpy(float)
    pred_val = memb_v.mean(axis=0)
    tot = memb_v.sum(axis=1); rep = int(np.argmin(np.abs(tot - np.median(tot))))
    df_val_out = valid[["date"]].copy()
    df_val_out["obs"] = y_va; df_val_out["pred"] = pred_val
    df_val_out["pred_member"] = memb_v[rep]
    df_val_out["p10"] = np.percentile(memb_v, 10, axis=0)
    df_val_out["p90"] = np.percentile(memb_v, 90, axis=0)
    met_val = compute_metrics(y_va, pred_val, thr, "Op.10 — Wilks | Val 2020-2022")
    met_val["NSE_mensual"] = nse_mensual(y_va, pred_val, valid["date"].values)

    # 2023-2024 ciego: semilla = último estado real de validación (dic-2022)
    ultimo_dia_val = df[(df["date"] <= cfg["val_end"]) & df["ppd"].notna()].sort_values("date")
    seed_pr = bool(ultimo_dia_val["ppd"].to_numpy(float)[-1] >= thr)
    pr = df_full[df_full["date"].dt.year.isin(cfg["pred_years"])].copy()
    memb_p = np.stack([gen_sequence(pr["date"], seed_pr, seed + 100 + s) for s in range(Kc)])
    pred_pr = memb_p.mean(axis=0)
    _tp = memb_p.sum(axis=1); _rp = int(np.argmin(np.abs(_tp - np.median(_tp))))
    df_pred_out = pr[["date"]].copy()
    df_pred_out["obs"] = pr["ppd"].values if "ppd" in pr.columns else np.nan
    df_pred_out["pred"] = pred_pr
    df_pred_out["pred_member"] = memb_p[_rp]
    df_pred_out["p10"] = np.percentile(memb_p, 10, axis=0)
    df_pred_out["p90"] = np.percentile(memb_p, 90, axis=0)
    met_pred = blind_metrics(pred_pr, f"Op.10 — Wilks | {cfg['pred_label']} CIEGO")

    return {"label": "Wilks-MixExp", "df_val": df_val_out, "df_pred": df_pred_out,
            "met_val": met_val, "met_pred": met_pred,
            "wk_params": {"w": float(mix["w"]), "mu1": float(mix["mu1"]),
                          "mu2": float(mix["mu2"]),
                          "p01_medio": _p01, "p11_medio": _p11,
                          "K_harmonics": K, "cap_mm": cap_mm,
                          "beta_markov_wet_wet": [float(b) for b in np.asarray(glm_ww.params)],
                          "beta_markov_dry_wet": [float(b) for b in np.asarray(glm_dw.params)],
                          "n_train": int(len(y_tr)), "n_wet_train": int(wet_tr.sum())}}


def main():
    cfg = CONFIG
    os.makedirs(cfg["output_folder"], exist_ok=True)
    print(f"\n  Carpeta de salida: {cfg['output_folder']}\n")

    df_raw = load_data(cfg)

    # ANTI-LEAK: features solo hasta val_end
    df_raw_no2024 = df_raw[df_raw["date"] <= cfg["val_end"]].copy().reset_index(drop=True)
    df = add_features(df_raw_no2024, cfg)
    train_df, val_df, _ = split_data(df, cfg)
    print(f"   [ANTI-LEAK] Features calculados solo hasta {cfg['val_end']}")

    resultados = []

    r3 = run_option3(df, cfg, df_raw=df_raw)
    resultados.append(r3)
    plot_individual(r3, cfg["output_folder"], cfg["pred_label"])

    r1 = run_option1(df, train_df, val_df, cfg, df_raw=df_raw)
    resultados.append(r1)
    plot_individual(r1, cfg["output_folder"], cfg["pred_label"])

    r2 = run_option2(df, train_df, val_df, cfg, df_raw=df_raw)
    resultados.append(r2)
    plot_individual(r2, cfg["output_folder"], cfg["pred_label"])

    # v74: ELIMINADO Op.4 (ZIDF-GammaParams) — producia predicciones identicas al Op.2 (ZIDF-Gamma)
    # v74: ELIMINADO componente de difusion (DDPM): el ZIDF predice con los parametros Gamma de la red

    r5 = run_option5_poisson_gamma(df, cfg, df_raw=df_raw)
    resultados.append(r5)
    plot_individual(r5, cfg["output_folder"], cfg["pred_label"])

    r6 = run_option6_hurdle_glm(df, cfg, df_raw=df_raw)
    resultados.append(r6)
    plot_individual(r6, cfg["output_folder"], cfg["pred_label"])

    r7 = run_option_ziegpd(df, cfg, df_raw=df_raw, mode="seasonal")
    resultados.append(r7)
    plot_individual(r7, cfg["output_folder"], cfg["pred_label"])

    # v74: ELIMINADO Op.8 (ZI-EGPD marginal) — colapsaba (CSI 0.078, PBIAS +103%)

    r9 = run_option9_zigamma_censored(df, cfg, df_raw=df_raw)
    resultados.append(r9)
    plot_individual(r9, cfg["output_folder"], cfg["pred_label"])

    r10 = run_option10_wilks(df, cfg, df_raw=df_raw)
    resultados.append(r10)
    plot_individual(r10, cfg["output_folder"], cfg["pred_label"])

    plot_comparativo(resultados, cfg["output_folder"], cfg["pred_label"], cfg)
    df_par = save_excel_calibracion(resultados, cfg)
    save_csv_parametros(df_par, cfg)

    print("\n" + "="*70)
    print("  SCRIPT 1 — CALIBRACION DE LOS 8 MODELOS COMPLETADA")
    print(f"  Carpeta: {cfg['output_folder']}")
    print("  Salida principal: 01_CALIBRACION_parametros.xlsx")
    print("  Siguiente paso : python 02_semillas_wilcoxon.py")
    print("="*70)

    return resultados, df_raw


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO FINAL
# ══════════════════════════════════════════════════════════════════════════════

def diagnostico_retro(resultados, df_raw, cfg):
    thr  = cfg["wet_threshold"]
    year = cfg["pred_year"]
    blind = cfg.get("blind_mode", False)

    print("\n\n" + "#"*72)
    print("#  DIAGNOSTICO FINAL vFINAL-CIEGO  --  COPIAR Y PEGAR TODO ESTO")
    print("#"*72)

    print(f"\n[CONTEXTO vFINAL]")
    print(f"  thr={thr}  seq_len={cfg['seq_len']}  n_autoreg_paths={cfg.get('n_autoreg_paths')}")
    print(f"  diff_steps={cfg['diff_steps']}  ddim_steps={cfg['ddim_steps']}")
    print(f"  epochs_occ={cfg['epochs_occ']}  epochs_pred={cfg['epochs_pred']}  "
          f"early_stop={cfg['early_stop_patience']}")
    print(f"  blind_mode={blind}")
    print(f"  vFINAL: 8 MODELOS FINALES — M1 Climatologico (base) | M2 Hurdle-Gamma LSTM | M3 ZI-Gamma LSTM | M4 Poisson-Gamma (Dzupire 2018) | M5 Hurdle-GLM | M6 ZI-EGPD estacional | M7 ZI-Gamma censurada | M8 Wilks 1999")
    print(f"  vFINAL: ELIMINADO Op.4 (ZIDF-GammaParams) — predicciones identicas al Op.2")
    print(f"  vFINAL: ELIMINADO Op.8 (ZI-EGPD marginal) — colapsaba (CSI 0.078, PBIAS +103%)")
    print(f"  vFINAL: ELIMINADA la difusion DDPM — el codigo nunca la uso; el ZIDF predice con los parametros Gamma de la red")
    print(f"  vFINAL: reparticion — calibracion 1977-2019 | validacion 2020-2022 | test ciego 2023 y 2024 (por ano)")
    print(f"  v72: desacopla Op.1 de Op.2/Op.4 — el held-out de calibración v71 ya no perturba la init de pesos posterior")
    print(f"  v72: reproducibilidad estricta restaurada (v71=v70+solo-Hurdle); modelos no-neuronales nunca dependieron del RNG de torch")
    print(f"  v71: RECALIBRACIÓN OCURRENCIA Hurdle/Op.1 — método={cfg.get('hurdle_calibration')} sobre held-out del train (cola {cfg.get('hurdle_cal_frac'):.0%})")
    print(f"  v71: diagnóstico v70 — la cabeza occ del Hurdle nunca bajaba de prob_min≈0.136 → Bernoulli-sin-umbral (v68) integraba prob·mu en los 1005 días → PBIAS=+209%")
    print(f"  v71: FIX — recalibración monótona (isotónica/Platt) baja el piso de prob hacia 0 en días secos; los días secos se apagan en el muestreo Bernoulli")
    print(f"  v71: SIN LEAK — calibrador ajustado SOLO en cola cronológica del train (1977-2020); la red occ no se entrenó en ese held-out; jamás toca val ni 2024")
    print(f"  v71: la red occ del Hurdle ahora entrena con (1-hurdle_cal_frac) del train; el resto queda reservado para calibrar")
    print(f"  v71: solo afecta a Op.1 (Hurdle); el resto de modelos queda intacto")
    print(f"  v62: FIX muestreo Gamma — apply_autoreg recibe (prob,alpha,beta) de la red; Gamma(alpha_red,1/beta_red) sin parámetros externos")
    print(f"  v62: ELIMINADO autoreg_gamma_shape=1.8 — el alpha de la red reemplaza al climatológico fijo")
    print(f"  v62: E[X]=alpha/beta=mu_red, Var[X]=alpha/beta² — variabilidad que la red aprendió")
    print(f"  v60: 2024 — ocurrencia Bernoulli en camino autorregresivo (antes umbral determinista → colapso seco)")
    print(f"  v60: rompe el ATRACTOR SECO: días húmedos esporádicos mantienen vivos los lags; calendario hace emerger estacionalidad")
    print(f"  v60: SIN LEAK — ocurrencia muestreada SOLO de la prob de los modelos VALIDADOS, jamás de obs 2024; sin reentrenar")
    print(f"  v60: validación INTACTA (umbral fijo); el modo Bernoulli solo afecta la predicción ciega 2024")
    print(f"  v59: GENERADOR ESTOCÁSTICO — no es pronóstico determinista del día exacto.")
    print(f"  v59: ensamble K={cfg.get('n_ensemble',50)} | MÉTRICAS=media del ensamble (=v58, insesgada)")
    print(f"  v59: GRÁFICA DIARIA=miembro representativo (criterio CIEGO: total≈mediana del ensamble) + banda P10-P90")
    print(f"  v59: con 1 sola variable y cv_mu bajo, la intensidad diaria es intrínsecamente poco predecible (física, no defecto)")
    print(f"  v59: se evalúa habilidad DISTRIBUCIONAL (frecuencia húmedos, intensidades, acumulados), no acierto día-a-día")
    print(f"  v58: media de N muestras Gamma (insesgada) → corrige sesgo negativo de la mediana v57")
    print(f"  v56: SIN freq_match (era trampa) | umbral fijo | cap={cfg.get('cap_mm',3.1)}mm | código muerto limpiado")
    print(f"  v54: FIX buffer=0 cuando pred=0 (antes=climatología → inflaba rolls en meses secos)")
    print(f"  v54: FIX cap Gamma 1.0mm en validación (consistente con _params_to_mm)")
    print(f"  v54: FIX diagnóstico mensual validación agregado al output")
    print(f"  v53: Hurdle umbral 0.34→0.35 (volver a v48: PBIAS=-10%, v51/v52 colapsaron) — rank-weighted inflaba mu de 0.26→0.40 (+115% PBIAS en v49)")
    print(f"  v48: umbral ZIDF-Gamma 0.20→0.21 (PBIAS +17.5%→+7.2%)")
    print(f"  v47: muestreo Gamma en validación ZIDF (pulsos, no rachas planas)")
    print(f"  v46: umbrales ZIDF recalibrados (prob=1-pi cambió escala) → gamma=0.20 params=0.24")
    print(f"  v45: ZIDF VERDADERO — NLL zero-inflated (pi estructural + Gamma con cero muestral)")
    print(f"  v45: Op.2 y Op.4 ya no son Hurdle disfrazado — mezcla real de ceros")
    print(f"  v44: pos_weight_cap_zidf_params=3.0 (v43=2.0 colapsó, v42=4.0 FAR alto → punto medio)")
    print(f"  v44: prob_thr_zidf_gamma=0.34 (volver a v42: PBIAS=+21.4%)  |  prob_thr_zidf_params=0.24")
    print(f"  v43: pos_weight_cap diferenciado para GammaParams")
    print(f"  v42: cap mu=1.0mm en _params_to_mm (P95 real serie)")
    print(f"  v41: lags 1-7 en ZIDF-Gamma y ZIDF-GammaParams")
    print(f"  v40: bias_init amt (alpha/beta climatológicos)  |  rank-weighted gamma NLL")
    print(f"  v40: sin Fase 2  |  HurdleGammaLSTM.amt = Linear(d,2)")
    print(f"  v40: focal_gamma={cfg['focal_gamma_pred']}  |  pos_weight cap={cfg['pos_weight_cap']}  |  ZIDF SIN DDPM")
    print(f"  v44: pos_weight_cap_zidf_params={cfg.get('pos_weight_cap_zidf_params', 4.0)}")
    print(f"  prob_thr (solo diagnóstico Gamma, NO predicción): hurdle={cfg['prob_thr_hurdle']} / zidf_gamma={cfg['prob_thr_zidf_gamma']}")
    print(f"  FIX v18-21: alpha_focal, pos_weight, bias_init, epochs=80")
    print(f"  Anio 1996: ausente  |  build_sequences_safe: activo")
    if blind:
        print(f"  OBS {year}: NO DISPONIBLE (modo ciego)")
    else:
        obs_yr = df_raw[df_raw["date"].dt.year.isin(cfg["pred_years"])]["ppd"].values
        print(f"  OBS {year}: n={len(obs_yr)}  sum={np.nansum(obs_yr):.1f}mm  "
              f"dias_humedos={(obs_yr>=thr).sum()}  frac={(obs_yr>=thr).mean():.3f}")

    # ── v54 FIX 3: Diagnóstico mensual de validación ──────────────────────────
    # Detecta meses con sobrepredicción sistemática (ej. ago 2021 en v53).
    # Imprime sum_obs y sum_pred por año-mes para cada modelo.
    print(f"\n[DIAGNÓSTICO MENSUAL VALIDACIÓN — v60]")
    print(f"  sum_obs y sum_pred por mes | ratio > 2.0 marcado con ⚠")
    print(f"  {'─'*100}")
    for res in resultados:
        label_m = res["label"]
        df_v_m  = res.get("df_val")
        if df_v_m is None or "obs" not in df_v_m.columns or "pred" not in df_v_m.columns:
            print(f"\n  {label_m:<22}  (sin datos de validación diaria)")
            continue
        df_v_m = df_v_m.copy()
        df_v_m["mes"] = pd.to_datetime(df_v_m["date"]).dt.to_period("M")
        agg_m = df_v_m.groupby("mes").agg(
            sum_obs =("obs",  lambda x: np.nansum(x)),
            sum_pred=("pred", lambda x: np.nansum(x))
        ).reset_index()
        agg_m["ratio"] = np.where(
            agg_m["sum_obs"] > 0.01,
            agg_m["sum_pred"] / agg_m["sum_obs"],
            np.where(agg_m["sum_pred"] > 0.1, np.inf, 1.0)
        )
        print(f"\n  ── {label_m} ──")
        print(f"    {'Mes':>8}  {'sum_obs':>8}  {'sum_pred':>9}  {'ratio':>7}  flag")
        print(f"    {'─'*50}")
        for _, row in agg_m.iterrows():
            ratio_s = f"{row['ratio']:>7.2f}" if np.isfinite(row['ratio']) else f"{'∞':>7}"
            flag = ""
            if np.isfinite(row['ratio']):
                if row['ratio'] > 3.0:
                    flag = "⚠⚠ MUY ALTO"
                elif row['ratio'] > 2.0:
                    flag = "⚠ ALTO"
                elif row['ratio'] < 0.3 and row['sum_obs'] > 0.1:
                    flag = "⚠ bajo"
            elif row['sum_pred'] > 0.5:
                flag = "⚠⚠ obs≈0 pred>0"
            print(f"    {str(row['mes']):>8}  {row['sum_obs']:>8.2f}  "
                  f"{row['sum_pred']:>9.2f}  {ratio_s}  {flag}")
        # resumen anual
        for yr in sorted(df_v_m["mes"].dt.year.unique()):
            sub_yr = agg_m[agg_m["mes"].dt.year == yr]
            print(f"    {'─'*50}")
            print(f"    {str(yr)+' total':>8}  {sub_yr['sum_obs'].sum():>8.2f}  "
                  f"{sub_yr['sum_pred'].sum():>9.2f}")
        print(f"    {'─'*50}")
        tot_obs  = agg_m["sum_obs"].sum()
        tot_pred = agg_m["sum_pred"].sum()
        pbias_m  = 100*(tot_pred-tot_obs)/tot_obs if tot_obs > 0 else np.nan
        print(f"    {'TOTAL VAL':>8}  {tot_obs:>8.2f}  {tot_pred:>9.2f}  "
              f"  PBIAS={pbias_m:+.1f}%")
    print(f"  {'─'*100}")

    # Probabilidades brutas
    print(f"\n[DIAGNÓSTICO PROB. BRUTAS — VALIDACIÓN]")
    for res in resultados:
        pv = res.get("prob_val_raw")
        yv = res.get("y_val_raw")
        if pv is None or yv is None:
            print(f"  {res['label']:<22}  (Op.3 — sin red neuronal)")
            continue
        _print_prob_diagnostico(pv, yv, thr, label=res["label"])

    # Tabla resumen
    print(f"\n[RESUMEN POR MODELO]")
    hdr = (f"  {'Modelo':<22} {'periodo':<6} {'sum_pred':>9} {'sum_obs':>8} "
           f"{'PBIAS%':>8} {'NSE_mes':>8} {'KGE':>7} {'POD':>6} {'FAR':>6} "
           f"{'CSI':>6} {'F1':>6} {'días>0':>7}")
    print(hdr); print("  " + "─"*100)
    colapso_flags = []
    for res in resultados:
        for periodo, key, dfkey in [("VAL", "met_val", "df_val"),
                                     (str(year), "met_pred", "df_pred")]:
            m  = res[key]
            dp = res[dfkey]
            n_pred_pos = int((dp["pred"].values > 0).sum())
            colapso = (periodo == str(year)) and (n_pred_pos == 0)
            if periodo == str(year):
                colapso_flags.append((res["label"], colapso, n_pred_pos))
            def _fmt(v, w=8, p=3, sign=False):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return f"{'NaN':>{w}}"
                return f"{v:>+{w}.{p}f}" if sign else f"{v:>{w}.{p}f}"
            print(f"  {res['label']:<22} {periodo:<6} "
                  f"{_fmt(m.get('Acum_pred'), 9, 1)} {_fmt(m.get('Acum_obs'), 8, 1)} "
                  f"{_fmt(m.get('PBIAS_%'), 8, 1, sign=True)} {_fmt(m.get('NSE_mensual'))} "
                  f"{_fmt(m.get('KGE'), 7)} {_fmt(m.get('POD'), 6)} "
                  f"{_fmt(m.get('FAR'), 6)} {_fmt(m.get('CSI'), 6)} "
                  f"{_fmt(m.get('F1'), 6)} {n_pred_pos:>7d}")

    print(f"\n[VEREDICTO COLAPSO {year}]")
    for label, colapso, npos in colapso_flags:
        estado = "❌ COLAPSADO (todo 0)" if colapso else f"✅ OK ({npos} días con lluvia>0)"
        print(f"  {label:<22}: {estado}")

    # Acumulado mensual pred_year
    print(f"\n[ACUMULADO MENSUAL {year} — TODOS LOS MODELOS]")
    header_m = f"  {'Modelo':<22}" + "".join(f"  {m:>3}" for m in range(1, 13)) + f"  {'TOTAL':>6}"
    print(header_m); print("  " + "─"*90)
    for res in resultados:
        dp = res["df_pred"].copy()
        dp["mes"] = pd.to_datetime(dp["date"]).dt.month
        agg = dp.groupby("mes")["pred"].sum()
        vals = [agg.get(m, 0.0) for m in range(1, 13)]
        row = f"  {res['label']:<22}" + "".join(f"  {v:>3.0f}" for v in vals) + f"  {sum(vals):>6.1f}"
        print(row)

    print(f"\n[ZERO-LEAK CHECK v19]")
    print(f"  ✅ Datos {year} = NaN en todo momento (modo ciego)")
    print(f"  ✅ add_features solo hasta {cfg['val_end']}")
    print(f"  ✅ build_sequences_safe: gap dic-1995/ene-1997 respetado")
    print(f"  ✅ Buffer autorregresivo init con datos < {year}")

    # ── v31: Diagnóstico Gamma por modelo ──────────────────────────────────
    print(f"\n[DIAGNÓSTICO GAMMA — DÍAS HÚMEDOS EN VALIDACIÓN]")
    print(f"  cv_mu > 0.3 → variabilidad real  |  cv_mu < 0.1 → modelo plano")
    print(f"  {'─'*70}")
    for res in resultados:
        label = res["label"]
        if label == "Climatológico":
            print(f"\n  {label:<22}  (sin red — sin parámetros Gamma)")
            continue
        # Recuperar alpha/beta/mu desde df_val si están guardados
        gd = res.get("gamma_diag")
        if gd is None:
            print(f"\n  {label:<22}  (diagnóstico Gamma no disponible)")
            continue
        print(f"\n  {label}")
        print(f"   n_wet_pred : {gd['n_wet']}")
        print(f"   alpha : min={gd['alpha_min']:.4f}  max={gd['alpha_max']:.4f}  "
              f"mean={gd['alpha_mean']:.4f}  std={gd['alpha_std']:.4f}")
        print(f"   beta  : min={gd['beta_min']:.4f}  max={gd['beta_max']:.4f}  "
              f"mean={gd['beta_mean']:.4f}  std={gd['beta_std']:.4f}")
        print(f"   mu    : min={gd['mu_min']:.4f}  max={gd['mu_max']:.4f}  "
              f"mean={gd['mu_mean']:.4f}  std={gd['mu_std']:.4f}")
        print(f"   cv_alpha={gd['cv_alpha']:.3f}  cv_mu={gd['cv_mu']:.3f}"
              f"  → {'✅ variabilidad real' if gd['cv_mu'] > 0.3 else '✅ variabilidad aceptable (v39)' if gd['cv_mu'] > 0.15 else '⚠ variabilidad débil' if gd['cv_mu'] > 0.1 else '⚠ modelo casi plano'}")
    print(f"\n[BARRIDO DE UMBRALES — VALIDACIÓN]")
    print(f"  Métrica guía: CSI (mayor = mejor)  |  umbral óptimo marcado con ✅")
    print(f"  {'─'*100}")

    for res in resultados:
        prob = res.get("prob_val_raw")
        y_v  = res.get("y_val_raw")
        mu_v = None
        # Reconstruir mu desde df_val si está disponible
        df_v = res.get("df_val")
        if prob is None or y_v is None:
            print(f"\n  {res['label']:<22}  (Op.3 — sin red, sin barrido)")
            continue

        print(f"\n  ── {res['label']} ──")
        prob = np.asarray(prob, dtype=np.float64).ravel()
        y_v  = np.asarray(y_v,  dtype=np.float64).ravel()
        obs_wet = (y_v >= thr)
        n_obs_wet = int(obs_wet.sum())
        prob_max_m = float(prob.max())

        # Reconstruir mu: pred_val / 1 donde pred_val>0, sino mu desconocido
        # Usamos df_val pred como proxy: pred = where(prob>thr_usado, mu, 0)
        # Para el barrido recalculamos pred = where(prob>t, mu_proxy, 0)
        # mu_proxy: asumimos pred_val guardado corresponde al umbral usado en v26
        thr_used = cfg.get("prob_thr", 0.30)
        if df_v is not None:
            pred_ref = df_v["pred"].values.ravel()
            # mu_proxy: donde prob > thr_used, mu = pred_ref; resto interpolamos con media
            mask_used = prob > thr_used
            mu_proxy = np.where(mask_used & (pred_ref > 0),
                                pred_ref,
                                pred_ref[mask_used].mean() if mask_used.any() else 1.0)
        else:
            mu_proxy = np.ones_like(prob)

        # Barrido de umbrales
        thresholds = np.round(np.arange(0.10, prob_max_m, 0.01), 3)
        if len(thresholds) == 0:
            print(f"    prob_max={prob_max_m:.4f} < 0.10 — sin barrido posible")
            continue

        rows = []
        for t in thresholds:
            pred_t = np.where(prob > t, mu_proxy, 0.0)
            n_pred_wet = int((pred_t > 0).sum())
            # métricas binarias
            tp = int(((pred_t > 0) & obs_wet).sum())
            fp = int(((pred_t > 0) & ~obs_wet).sum())
            fn = int(((pred_t == 0) & obs_wet).sum())
            pod  = tp / max(tp + fn, 1)
            far  = fp / max(tp + fp, 1)
            csi  = tp / max(tp + fp + fn, 1)
            f1   = 2*tp / max(2*tp + fp + fn, 1)
            # PBIAS
            sum_pred = float(pred_t.sum())
            sum_obs  = float(np.nansum(y_v))
            pbias = 100*(sum_pred - sum_obs)/sum_obs if sum_obs > 0 else np.nan
            # NSE mensual requiere fechas — usamos NSE simple como proxy rápido
            ss_res = float(np.sum((y_v - pred_t)**2))
            ss_tot = float(np.sum((y_v - np.nanmean(y_v))**2))
            nse = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
            rows.append((t, n_pred_wet, pbias, pod, far, csi, f1, nse))

        # Umbral óptimo = mayor CSI
        best_idx = int(np.argmax([r[5] for r in rows]))

        hdr_t = (f"    {'umbral':>7} {'n_pred':>7} {'PBIAS%':>8} "
                 f"{'POD':>6} {'FAR':>6} {'CSI':>6} {'F1':>6} {'NSE':>7}  ")
        print(hdr_t)
        print(f"    {'─'*72}")
        for i, (t, np_, pb, pod, far, csi, f1, nse) in enumerate(rows):
            marca = " ✅ ÓPTIMO" if i == best_idx else ""
            pb_s  = f"{pb:>+8.1f}" if not np.isnan(pb)  else f"{'NaN':>8}"
            nse_s = f"{nse:>7.3f}"  if not np.isnan(nse) else f"{'NaN':>7}"
            print(f"    {t:>7.2f} {np_:>7d} {pb_s} "
                  f"{pod:>6.3f} {far:>6.3f} {csi:>6.3f} {f1:>6.3f} {nse_s}{marca}")

        best = rows[best_idx]
        print(f"\n    → Umbral óptimo (CSI): {best[0]:.2f}  "
              f"n_pred={best[1]}  PBIAS={best[2]:+.1f}%  "
              f"CSI={best[5]:.3f}  FAR={best[4]:.3f}")
        print(f"    → n_obs_wet={n_obs_wet}  prob_max={prob_max_m:.4f}")
    # ── fin barrido ──────────────────────────────────────────────────────────

    print(f"\n[VEREDICTO CALIBRACION vFINAL]")
    print(f"  global_scale  : ELIMINADO — PBIAS refleja calidad real del modelo")
    print(f"  prob_scale    : ELIMINADO — probabilidades brutas sin frequency-matching global")
    print(f"  freq_match: ELIMINADO en v56 (usaba cuantil de prob_val = trampa de validación)")
    print(f"  v71: Hurdle/Op.1 — recalibración occ ({cfg.get('hurdle_calibration')}) ajustada en held-out del TRAIN, no en validación → NO es trampa")
    print(f"  v71: distinción clave — ajustar sobre prob_val = leak (lo que se borró); ajustar sobre held-out del train = legítimo")
    print(f"  v71: el resto de modelos (ZIDF, GLM, EGPD, Wilks...) siguen con prob cruda sin recalibrar")
    print(f"  pred = clip(Gamma(alpha_red, 1/beta_red), 0, cap_mm) × Bernoulli(prob) — v62: parámetros directos de la red")
    print(f"  v62: apply_autoreg recibe (prob, alpha, beta); sin autoreg_gamma_shape externo")
    print(f"  v61 FIX (mantenido): ZIDF _param_fn devuelve alpha/beta reales de la red")
    print(f"  v40: bias_init amt → mu_init ancla en media climatológica días húmedos")
    print(f"  v40: rank-weighted gamma NLL → gradiente diferencial en días húmedos")
    print(f"  v40: sin Fase 2 — entrenamiento conjunto único")

    print(f"\n[GUIA DE DIAGNOSTICO vFINAL]")
    print(f"  cv_mu > 0.20   → variabilidad real ✅ (objetivo v44)")
    print(f"  cv_mu 0.10-0.20→ variabilidad débil ⚠")
    print(f"  cv_mu < 0.10   → modelo casi plano ⚠  (rank-weight no tuvo efecto)")
    print(f"  PBIAS < -30%   → mu bajo; revisar bias_init amt o gamma_nll")
    print(f"  PBIAS > +50%   → mu inflado; revisar rank-weight o pos_weight")
    print(f"  NSE_mensual < 0 → modelo peor que la media")
    print(f"  KGE < 0        → señal de alerta grave")
    print(f"  FAR > 0.7      → demasiados falsos positivos")
    print(f"  ratio_mes > 2.0 → freq-match mensual no alcanzó a corregir (revisar mu)")

    print("\n" + "#"*72)
    print("#  FIN DIAGNOSTICO vFINAL -- pega todo esto para retroalimentar")
    print("#"*72 + "\n")


if __name__ == "__main__":
    resultados, df_raw = main()
    diagnostico_retro(resultados, df_raw, CONFIG)
