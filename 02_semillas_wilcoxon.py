# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT 2 de 2 — SEMILLAS DE TODOS LOS MODELOS + CONTRASTE DE WILCOXON
================================================================================
Reemplaza y unifica los antiguos:
    vfinal_multirun.py  +  vfinal_multirun_estadisticos.py  +  wilcoxon_pareado.py

QUE HACE
    Ejecuta los siete modelos estocasticos (M2-M8) con las MISMAS 20 semillas
    (42-61) y el baseline determinista M1 una sola vez, y a partir de ahi calcula
    la variabilidad entre corridas y el contraste estadistico pareado.

    Diferencia entre modelos (importante para la tesis):
      · M2 y M3 (LSTM): cada semilla es un REENTRENAMIENTO completo -> varian los
        pesos de la red y, por tanto, los parametros.
      · M4-M8 (estadisticos): el ajuste es determinista (MLE / GLM / EM); la
        semilla solo re-muestrea el ensemble de K miembros. Sus parametros deben
        salir IDENTICOS en las 20 corridas y el script lo verifica y lo reporta.

QUE ENTREGA  ->  02_SEMILLAS_WILCOXON.xlsx
    00_DICCIONARIO           Que significa cada hoja y cada metrica.
    01_Estadisticos_por_seed CRUDO: una fila por (modelo, semilla, periodo) con
                             las 7 metricas. Es la base de todo lo demas y la
                             fuente del Anexo de metricas por corrida.
    02_Resumen_media_std     Agregado media +/- desviacion estandar (20 semillas).
    03_Resumen_mediana_iqr   Agregado mediana [IQR] (20 semillas).
    04_Wilcoxon_pareado      Las 21 comparaciones entre M2-M8, pareadas por
                             semilla, para 7 metricas x 3 periodos = 441 filas.
    05_Wilcoxon_vs_M1        Test de 1 muestra de cada modelo contra el valor
                             determinista de M1 (M1 no tiene semillas).
    06_Estabilidad_params    Verificacion de que M4-M8 dan parametros identicos
                             entre semillas, y rango de parametros de M2/M3.
    07_EarlyStop_LSTM        Epoca de early-stopping por semilla (M2 y M3), base
                             de la figura de curvas de perdida.

    Y en la carpeta  semillas/  (un archivo por modelo y semilla):
        {Mx}_seed{NN}_val.csv          date, obs, pred, pred_member, p10, p90
        {Mx}_seed{NN}_pred.csv         idem para el test ciego 2023-2024
        {Mx}_seed{NN}_loss.json        curvas de perdida (solo M2 y M3)

    -> Los p10/p90 diarios del ensemble para CADA semilla salen en esos CSV.

LO QUE **NO** HACE (para no duplicar salidas)
    No vuelve a exportar los parametros calibrados en detalle: eso ya lo entrega
    el SCRIPT 1 (01_CALIBRACION_parametros.xlsx). Aqui solo se reporta su
    ESTABILIDAD entre semillas, que es informacion distinta.

USO
    python 02_semillas_wilcoxon.py
    python 02_semillas_wilcoxon.py --seeds 20 --out D:\\0final\\semillas
    (debe estar en la misma carpeta que 01_calibracion.py)

REQUIERE
    serie4f.xlsx  -> observaciones reales 2023-2024, usadas SOLO para evaluar el
                     test ciego a posteriori. Nunca entran al entrenamiento ni a
                     la calibracion.
================================================================================
"""

import os
import sys
import copy
import json
import argparse
import importlib.util
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DEL SCRIPT 1 COMO LIBRERIA
# ══════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.join(_HERE, "01_calibracion.py")
if not os.path.exists(_PIPELINE):
    sys.exit(f"[ERROR] No se encontro {_PIPELINE}. "
             "El script 2 debe estar en la misma carpeta que 01_calibracion.py")
_SPEC = importlib.util.spec_from_file_location("calibracion", _PIPELINE)
M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M)   # el guard __main__ de 01 evita que corra su main()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════════
N_RUNS       = 20
SEED_BASE    = 42                                   # semillas 42 .. 61
SERIE4F_PATH = r"D:\0final\serie4f.xlsx"            # obs real 2023-2024 (solo evaluar)
OUT_DIR      = M.CONFIG["output_folder"]

METRICS  = ["NSE_mensual", "KGE", "PBIAS_%", "POD", "FAR", "CSI", "F1"]
PERIODOS = ["Validacion 2020-2022", "Test ciego 2023", "Test ciego 2024"]
ALPHA    = 0.05

# Direccion de cada metrica: True = mayor es mejor, False = menor es mejor,
# None = se compara el valor absoluto (mas cerca de 0 es mejor).
MAYOR_MEJOR = {"NSE_mensual": True, "KGE": True, "POD": True,
               "CSI": True, "F1": True, "FAR": False, "PBIAS_%": None}

# Los siete modelos estocasticos, en orden. M1 va aparte (determinista).
MODELOS = {
    "M2": ("Hurdle-Gamma",  "lstm",  M.run_option1,                 {}),
    "M3": ("ZIDF-Gamma",    "lstm",  M.run_option2,                 {}),
    "M4": ("Poisson-Gamma", "stat",  M.run_option5_poisson_gamma,   {}),
    "M5": ("Hurdle-GLM",    "stat",  M.run_option6_hurdle_glm,      {}),
    "M6": ("ZI-EGPD-seas",  "stat",  M.run_option_ziegpd,           {"mode": "seasonal"}),
    "M7": ("ZI-Gamma-cens", "stat",  M.run_option9_zigamma_censored, {}),
    "M8": ("Wilks-MixExp",  "stat",  M.run_option10_wilks,          {}),
}
LSTM_MODELS = [k for k, v in MODELOS.items() if v[1] == "lstm"]
STAT_MODELS = [k for k, v in MODELOS.items() if v[1] == "stat"]

# Clave del dict de parametros de cada modelo dentro del objeto que devuelve
PARAM_KEY = {"M2": "lstm_params", "M3": "lstm_params", "M4": "pg_params",
             "M5": "hg_params",   "M6": "egpd_params", "M7": "cz_params",
             "M8": "wk_params"}


# ══════════════════════════════════════════════════════════════════════════════
# TEST EXTERNO — serie4f SOLO para comparar (nunca entra a train/calibracion)
# ══════════════════════════════════════════════════════════════════════════════
def load_external_obs(path):
    if not os.path.exists(path):
        print(f"  [AVISO] No se encontro {path}. El test ciego 2023-2024 quedara vacio.")
        return None
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "ppd"]].rename(columns={"ppd": "obs_real"})


def external_test_metrics(df_pred, obs_ext, thr):
    """Metricas del test ciego contra las observaciones reales, por ano."""
    out = {2023: {}, 2024: {}}
    if obs_ext is None:
        return out
    d = df_pred[["date", "pred"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.merge(obs_ext, on="date", how="left").dropna(subset=["obs_real", "pred"])
    for yr in (2023, 2024):
        dy = d[d["date"].dt.year == yr]
        if len(dy) == 0:
            continue
        m = M.compute_metrics(dy["obs_real"].values, dy["pred"].values, thr)
        m["NSE_mensual"] = M.nse_mensual(dy["obs_real"].values,
                                         dy["pred"].values, dy["date"].values)
        out[yr] = m
    return out


def period_metrics(res, obs_ext, thr):
    ext = external_test_metrics(res["df_pred"], obs_ext, thr)
    return {"Validacion 2020-2022": res["met_val"],
            "Test ciego 2023": ext.get(2023, {}),
            "Test ciego 2024": ext.get(2024, {})}


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCION MULTI-SEMILLA
# ══════════════════════════════════════════════════════════════════════════════
def setup_datos():
    cfg = M.CONFIG
    df_raw = M.load_data(cfg)
    df_raw_hasta_val = df_raw[df_raw["date"] <= cfg["val_end"]].copy().reset_index(drop=True)
    df = M.add_features(df_raw_hasta_val, cfg)
    train_df, val_df, _ = M.split_data(df, cfg)
    return df, train_df, val_df, df_raw


def _flatten(d, prefix=""):
    """Aplana un dict de parametros a {nombre: valor} para poder compararlo."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
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


def _early_stop_epoch(hist):
    try:
        return int(np.argmin(hist["val"])) + 1
    except Exception:
        return None


def guardar_csv_semilla(code, seed, res, out_dir):
    """Serie diaria por semilla, con la banda p10/p90 del ensemble."""
    os.makedirs(out_dir, exist_ok=True)
    for tag, key in (("val", "df_val"), ("pred", "df_pred")):
        d = res.get(key)
        if d is None:
            continue
        cols = [c for c in ["date", "obs", "pred", "pred_member", "p10", "p90"]
                if c in d.columns]
        d[cols].to_csv(os.path.join(out_dir, f"{code}_seed{seed}_{tag}.csv"),
                       index=False)
    # Curvas de perdida (solo LSTM)
    loss = {}
    if "hist_occ" in res:
        loss = {"occ_train": res["hist_occ"]["train"], "occ_val": res["hist_occ"]["val"],
                "amt_train": res["hist_amt"]["train"], "amt_val": res["hist_amt"]["val"],
                "epoca_early_stop_occ": _early_stop_epoch(res["hist_occ"]),
                "epoca_early_stop_amt": _early_stop_epoch(res["hist_amt"])}
    elif "hist_lstm" in res:
        loss = {"lstm_train": res["hist_lstm"]["train"], "lstm_val": res["hist_lstm"]["val"],
                "epoca_early_stop": _early_stop_epoch(res["hist_lstm"])}
    if loss:
        with open(os.path.join(out_dir, f"{code}_seed{seed}_loss.json"), "w") as f:
            json.dump({"modelo": code, "seed": seed, **_jsonable(loss)}, f, indent=2)


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def correr_semillas(n_runs, out_dir):
    """Corre M1 una vez y M2-M8 con n_runs semillas. Devuelve los crudos."""
    thr     = M.CONFIG["wet_threshold"]
    obs_ext = load_external_obs(SERIE4F_PATH)
    df, train_df, val_df, df_raw = setup_datos()
    csv_dir = os.path.join(out_dir, "semillas")
    os.makedirs(csv_dir, exist_ok=True)

    # ── M1: determinista, una sola corrida ────────────────────────────────────
    print("\n" + "#" * 70)
    print("  M1 Climatologico (determinista, sin semillas)")
    print("#" * 70)
    r_m1 = M.run_option3(df, M.CONFIG, df_raw=df_raw)
    m1_per = period_metrics(r_m1, obs_ext, thr)
    guardar_csv_semilla("M1", SEED_BASE, r_m1, csv_dir)

    # ── M2-M8: n_runs semillas ────────────────────────────────────────────────
    filas_metricas = []
    filas_params   = []
    filas_epocas   = []

    for i in range(n_runs):
        seed = SEED_BASE + i
        cfg = copy.deepcopy(M.CONFIG)
        cfg["seed"] = seed
        print("\n" + "#" * 70)
        print(f"  SEMILLA {seed}   ({i + 1}/{n_runs})")
        print("#" * 70)

        for code, (nombre, tipo, fn, kw) in MODELOS.items():
            res = fn(df, train_df, val_df, cfg, df_raw=df_raw, **kw) \
                if tipo == "lstm" else fn(df, cfg, df_raw=df_raw, **kw)

            per = period_metrics(res, obs_ext, thr)
            for periodo in PERIODOS:
                fila = {"Modelo": code, "Nombre": nombre, "seed": seed,
                        "periodo": periodo}
                for met in METRICS:
                    fila[met] = per.get(periodo, {}).get(met, np.nan)
                filas_metricas.append(fila)

            pars = _flatten(res.get(PARAM_KEY[code], {}))
            for pname, pval in pars.items():
                filas_params.append({"Modelo": code, "seed": seed,
                                     "Parametro": pname, "Valor": pval})

            if tipo == "lstm":
                if "hist_occ" in res:
                    filas_epocas.append({
                        "Modelo": code, "seed": seed,
                        "early_stop_ocurrencia": _early_stop_epoch(res["hist_occ"]),
                        "early_stop_cantidad":   _early_stop_epoch(res["hist_amt"]),
                        "early_stop_conjunto":   None})
                else:
                    filas_epocas.append({
                        "Modelo": code, "seed": seed,
                        "early_stop_ocurrencia": None,
                        "early_stop_cantidad":   None,
                        "early_stop_conjunto":   _early_stop_epoch(res["hist_lstm"])})

            guardar_csv_semilla(code, seed, res, csv_dir)

    df_met    = pd.DataFrame(filas_metricas)
    df_params = pd.DataFrame(filas_params)
    df_epocas = pd.DataFrame(filas_epocas)
    return df_met, df_params, df_epocas, m1_per, csv_dir


# ══════════════════════════════════════════════════════════════════════════════
# AGREGACION
# ══════════════════════════════════════════════════════════════════════════════
def resumen_media_std(df_met):
    filas = []
    for (code, nombre, periodo), g in df_met.groupby(["Modelo", "Nombre", "periodo"],
                                                     sort=False):
        fila = {"Modelo": code, "Nombre": nombre, "periodo": periodo,
                "n_semillas": int(g["seed"].nunique())}
        for met in METRICS:
            v = g[met].dropna().values
            fila[f"{met}_media"] = float(np.mean(v)) if len(v) else np.nan
            fila[f"{met}_std"]   = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(["periodo", "Modelo"]).reset_index(drop=True)


def resumen_mediana_iqr(df_met):
    filas = []
    for (code, nombre, periodo), g in df_met.groupby(["Modelo", "Nombre", "periodo"],
                                                     sort=False):
        fila = {"Modelo": code, "Nombre": nombre, "periodo": periodo,
                "n_semillas": int(g["seed"].nunique())}
        for met in METRICS:
            v = g[met].dropna().values
            if len(v):
                q1, q3 = np.percentile(v, [25, 75])
                fila[f"{met}_mediana"] = float(np.median(v))
                fila[f"{met}_Q1"] = float(q1)
                fila[f"{met}_Q3"] = float(q3)
                fila[f"{met}_IQR"] = float(q3 - q1)
            else:
                fila[f"{met}_mediana"] = fila[f"{met}_Q1"] = np.nan
                fila[f"{met}_Q3"] = fila[f"{met}_IQR"] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(["periodo", "Modelo"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# WILCOXON PAREADO
# ══════════════════════════════════════════════════════════════════════════════
def _par_alineado(df_met, mA, mB, periodo, met):
    """Vectores a, b alineados por semilla (solo semillas presentes en ambos)."""
    sel = df_met[df_met["periodo"] == periodo]
    a = sel[sel["Modelo"] == mA][["seed", met]].set_index("seed")[met]
    b = sel[sel["Modelo"] == mB][["seed", met]].set_index("seed")[met]
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().sort_index()
    return j["a"].values, j["b"].values


def _mejor(mA, mB, med_a, med_b, met):
    if np.isnan(med_a) or np.isnan(med_b):
        return ""
    direc = MAYOR_MEJOR[met]
    if direc is None:                       # PBIAS: menor |valor| es mejor
        va, vb = abs(med_a), abs(med_b)
        return "empate" if va == vb else (mA if va < vb else mB)
    if med_a == med_b:
        return "empate"
    if direc:
        return mA if med_a > med_b else mB
    return mA if med_a < med_b else mB


def wilcoxon_pareado(df_met):
    """21 comparaciones entre M2-M8, pareadas por semilla."""
    if not _HAVE_SCIPY:
        print("  [AVISO] scipy no disponible: se omite el Wilcoxon.")
        return pd.DataFrame()
    modelos = sorted(df_met["Modelo"].unique())
    filas = []
    for periodo in PERIODOS:
        for mA, mB in combinations(modelos, 2):
            for met in METRICS:
                a, b = _par_alineado(df_met, mA, mB, periodo, met)
                n = len(a)
                if n == 0:
                    continue
                dif = a - b
                n_ceros = int(np.sum(dif == 0))
                dnz = dif[dif != 0]
                # empates: |diferencias| repetidas, o diferencias nulas
                tiene_empates = bool(len(np.unique(np.abs(dnz))) < len(dnz)) or (n_ceros > 0)
                try:
                    if np.all(dif == 0):
                        p, w = 1.0, 0.0
                    else:
                        r = wilcoxon(a, b)
                        p, w = float(r.pvalue), float(r.statistic)
                except Exception:
                    p, w = np.nan, np.nan
                med_a, med_b = float(np.median(a)), float(np.median(b))
                filas.append({
                    "periodo": periodo,
                    "comparacion": f"{mA} vs {mB}",
                    "modelo_A": mA, "modelo_B": mB,
                    "metrica": met, "n_pares": n,
                    "mediana_A": round(med_a, 4), "mediana_B": round(med_b, 4),
                    "W_stat": w, "valor_p": p,
                    "significativo_0.05": ("si" if (not np.isnan(p) and p < ALPHA) else "no"),
                    "mejor_modelo": _mejor(mA, mB, med_a, med_b, met),
                    "n_ceros": n_ceros, "tiene_empates": tiene_empates,
                })
    return pd.DataFrame(filas)


def wilcoxon_vs_m1(df_met, m1_per):
    """Test de 1 muestra de cada modelo contra el valor determinista de M1."""
    if not _HAVE_SCIPY:
        return pd.DataFrame()
    filas = []
    for periodo in PERIODOS:
        base_all = m1_per.get(periodo, {})
        for code in sorted(df_met["Modelo"].unique()):
            for met in METRICS:
                vals = df_met[(df_met["Modelo"] == code) &
                              (df_met["periodo"] == periodo)][met].dropna().values
                base = base_all.get(met, np.nan)
                if len(vals) == 0 or (isinstance(base, float) and np.isnan(base)):
                    continue
                dif = vals - base
                try:
                    p = 1.0 if np.all(dif == 0) else float(wilcoxon(dif).pvalue)
                except Exception:
                    p = np.nan
                filas.append({
                    "periodo": periodo, "comparacion": f"{code} vs M1",
                    "modelo": code, "metrica": met, "n": int(len(vals)),
                    "mediana_modelo": round(float(np.median(vals)), 4),
                    "valor_M1": round(float(base), 4), "valor_p": p,
                    "significativo_0.05": ("si" if (not np.isnan(p) and p < ALPHA) else "no"),
                })
    return pd.DataFrame(filas)


# ══════════════════════════════════════════════════════════════════════════════
# ESTABILIDAD DE PARAMETROS
# ══════════════════════════════════════════════════════════════════════════════
def estabilidad_parametros(df_params):
    """Para cada (modelo, parametro): rango entre semillas y si es identico.

    En M4-M8 'identico_entre_semillas' DEBE ser True (el ajuste es determinista);
    en M2-M3 sera False, y el rango cuantifica la variabilidad del reentrenamiento.
    """
    if df_params.empty:
        return pd.DataFrame()
    filas = []
    for (code, pname), g in df_params.groupby(["Modelo", "Parametro"], sort=False):
        v = pd.to_numeric(g["Valor"], errors="coerce").dropna().values
        if len(v) == 0:
            continue
        rng = float(np.max(v) - np.min(v))
        filas.append({
            "Modelo": code, "Parametro": pname,
            "n_semillas": int(g["seed"].nunique()),
            "media": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)), "max": float(np.max(v)), "rango": rng,
            "identico_entre_semillas": bool(rng < 1e-9),
            "tipo": ("determinista (esperado identico)" if code in STAT_MODELS
                     else "reentrenado (se espera variacion)"),
        })
    return pd.DataFrame(filas).sort_values(["Modelo", "Parametro"]).reset_index(drop=True)


def diccionario():
    filas = [
        ("01_Estadisticos_por_seed", "CRUDO. Una fila por modelo-semilla-periodo con las 7 metricas. Fuente de todo lo demas."),
        ("02_Resumen_media_std", "Media +/- desviacion estandar de cada metrica sobre las 20 semillas."),
        ("03_Resumen_mediana_iqr", "Mediana y rango intercuartilico (Q1, Q3, IQR) sobre las 20 semillas."),
        ("04_Wilcoxon_pareado", "Wilcoxon de rangos con signo, pareado por semilla, entre todos los pares M2-M8."),
        ("05_Wilcoxon_vs_M1", "Test de 1 muestra de cada modelo contra el valor determinista del baseline M1."),
        ("06_Estabilidad_params", "Rango de cada parametro entre semillas. Verifica que M4-M8 sean deterministas."),
        ("07_EarlyStop_LSTM", "Epoca de early-stopping por semilla en M2 y M3."),
        ("CSV por semilla", "Carpeta 'semillas/': serie diaria con obs, pred y banda p10/p90 del ensemble."),
        ("NSE_mensual", "Nash-Sutcliffe a escala mensual. Mayor es mejor."),
        ("KGE", "Kling-Gupta Efficiency, escala diaria. Mayor es mejor."),
        ("PBIAS_%", "Sesgo porcentual. Positivo = sobreestimacion. Mejor cuanto mas cerca de 0."),
        ("POD", "Probability of Detection (aciertos sobre eventos observados). Mayor es mejor."),
        ("FAR", "False Alarm Ratio. MENOR es mejor."),
        ("CSI", "Critical Success Index. Mayor es mejor."),
        ("F1", "Media armonica de precision y POD. Mayor es mejor."),
        ("Semillas", f"{SEED_BASE} a {SEED_BASE + N_RUNS - 1} ({N_RUNS} corridas por modelo)."),
        ("M1", "Determinista: no tiene semillas, se corre una sola vez."),
        ("M2, M3", "LSTM: cada semilla es un reentrenamiento completo."),
        ("M4-M8", "Estadisticos: el ajuste es determinista; la semilla solo re-muestrea el ensemble."),
        ("Empates en POD", "POD toma pocos valores discretos: genera empates y scipy pasa a la aproximacion "
                           "normal. Las filas afectadas van marcadas con tiene_empates=True. No cambia la significancia."),
        ("Nivel de significancia", f"alpha = {ALPHA} (valor p de dos colas)."),
        ("scipy", (__import__("scipy").__version__ if _HAVE_SCIPY else "no disponible")),
    ]
    return pd.DataFrame(filas, columns=["Concepto", "Detalle"])


# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO
# ══════════════════════════════════════════════════════════════════════════════
def guardar_excel(out_dir, df_met, media_std, mediana_iqr, wilco, wilco_m1,
                  estab, df_epocas):
    path = os.path.join(out_dir, "02_SEMILLAS_WILCOXON.xlsx")
    print(f"\n  Guardando Excel: {path}")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        diccionario().to_excel(xw,  sheet_name="00_DICCIONARIO", index=False)
        df_met.to_excel(xw,         sheet_name="01_Estadisticos_por_seed", index=False)
        media_std.to_excel(xw,      sheet_name="02_Resumen_media_std", index=False)
        mediana_iqr.to_excel(xw,    sheet_name="03_Resumen_mediana_iqr", index=False)
        if not wilco.empty:
            wilco.to_excel(xw,      sheet_name="04_Wilcoxon_pareado", index=False)
        if not wilco_m1.empty:
            wilco_m1.to_excel(xw,   sheet_name="05_Wilcoxon_vs_M1", index=False)
        if not estab.empty:
            estab.to_excel(xw,      sheet_name="06_Estabilidad_params", index=False)
        if not df_epocas.empty:
            df_epocas.to_excel(xw,  sheet_name="07_EarlyStop_LSTM", index=False)
    # Copia CSV del crudo, util para el repositorio
    df_met.to_csv(os.path.join(out_dir, "02_estadisticos_por_seed.csv"),
                  index=False, encoding="utf-8-sig")
    print("  Excel y CSV guardados.")


def _resumen_consola(df_met, wilco, estab):
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    print(f"  Filas crudas (modelo x semilla x periodo): {len(df_met)}")
    if not wilco.empty:
        sig = int((wilco["significativo_0.05"] == "si").sum())
        print(f"  Comparaciones Wilcoxon: {len(wilco)}  |  significativas: {sig}")
    if not estab.empty:
        mal = estab[(estab["Modelo"].isin(STAT_MODELS)) &
                    (~estab["identico_entre_semillas"])]
        if len(mal) == 0:
            print("  Estabilidad M4-M8: OK, todos los parametros identicos entre semillas.")
        else:
            print(f"  [ATENCION] {len(mal)} parametros de M4-M8 NO son identicos entre "
                  "semillas. Revisar la hoja 06_Estabilidad_params:")
            print(mal[["Modelo", "Parametro", "rango"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global SERIE4F_PATH

    ap = argparse.ArgumentParser(
        description="Corre los 8 modelos con multiples semillas y calcula el Wilcoxon pareado.")
    ap.add_argument("--seeds", type=int, default=N_RUNS,
                    help=f"Numero de semillas (por defecto {N_RUNS}, desde {SEED_BASE}).")
    ap.add_argument("--out", default=OUT_DIR, help="Carpeta de salida.")
    ap.add_argument("--serie4f", default=SERIE4F_PATH,
                    help="Ruta a serie4f.xlsx (observado real 2023-2024).")
    args = ap.parse_args()

    SERIE4F_PATH = args.serie4f
    os.makedirs(args.out, exist_ok=True)

    print("=" * 70)
    print("  SCRIPT 2 — SEMILLAS DE TODOS LOS MODELOS + WILCOXON PAREADO")
    print(f"  Semillas : {SEED_BASE} a {SEED_BASE + args.seeds - 1} ({args.seeds} corridas)")
    print(f"  Modelos  : M1 (1 corrida) + {', '.join(MODELOS.keys())} ({args.seeds} c/u)")
    print(f"  Salida   : {args.out}")
    print("=" * 70)

    df_met, df_params, df_epocas, m1_per, csv_dir = correr_semillas(args.seeds, args.out)

    print("\n  Agregando estadisticos por semilla...")
    media_std   = resumen_media_std(df_met)
    mediana_iqr = resumen_mediana_iqr(df_met)

    print("  Calculando Wilcoxon pareado (M2-M8)...")
    wilco = wilcoxon_pareado(df_met)

    print("  Calculando contraste contra el baseline M1...")
    wilco_m1 = wilcoxon_vs_m1(df_met, m1_per)

    print("  Verificando estabilidad de parametros entre semillas...")
    estab = estabilidad_parametros(df_params)

    guardar_excel(args.out, df_met, media_std, mediana_iqr, wilco, wilco_m1,
                  estab, df_epocas)
    _resumen_consola(df_met, wilco, estab)

    print(f"\n  CSV diarios por semilla (con p10/p90): {csv_dir}")
    print("  Listo.")


if __name__ == "__main__":
    main()
