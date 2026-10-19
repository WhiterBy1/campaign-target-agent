#!/usr/bin/env python3
"""Pruebas de la Fase 3: cada medida de desempeno (M1-M6) contra los escenarios de
umbrales_fase3.json, con 5 repeticiones por escenario (el ambiente es estocastico).

Uso:
    python pruebas_fase3.py                    # corre todo (~10 min)
    python pruebas_fase3.py --rapido           # 2 repeticiones, solo para depurar
    python pruebas_fase3.py --escenarios N1 F2 # subconjunto

Salidas en resultados_fase3/ (CSV/JSON) y logs de cada corrida en logs_fase3/.
"""
import argparse
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from campaign_target_agent import (CampaignTargetAgent, SimuladorBanco, cargar_datos,
                                   dividir, modelos_base, preparar)

RAIZ = Path(__file__).parent
CICLO_DANADO = 2  # el dano se inyecta en el segundo lote
SECUENCIA_OK = ["percepcion", "decision", "accion", "critica", "aprendizaje"]
SECUENCIA_RECHAZO = ["percepcion", "rechazo"]

_ctx = {}


def _contexto():
    """Carga datos una vez por proceso (cada worker tiene el suyo)."""
    if not _ctx:
        X, y = preparar(cargar_datos())
        X_train, X_test, y_train, y_test = dividir(X, y, 42)
        _ctx.update(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    return _ctx


DANOS = {
    "sin_columna": lambda b: b.drop(columns=["balance"]),
    "con_nan": lambda b: b.assign(balance=np.nan),
    "vacio": lambda b: b.iloc[0:0],
    "balance_texto": lambda b: b.assign(balance="sin dato"),
}


def revisar_log(ruta):
    """M5: devuelve (ciclos_completos, ciclos_procesados) segun la secuencia esperada."""
    por_ciclo = {}
    prev_ts, ts_ok = None, True
    for linea in open(ruta, encoding="utf-8"):
        r = json.loads(linea)
        ts = datetime.fromisoformat(r["timestamp"])
        if prev_ts and ts < prev_ts:
            ts_ok = False
        prev_ts = ts
        por_ciclo.setdefault(r["ciclo"], []).append(r["evento"])
    buenos = sum(1 for ev in por_ciclo.values() if ev in (SECUENCIA_OK, SECUENCIA_RECHAZO))
    return (buenos if ts_ok else 0), len(por_ciclo)


def correr(args):
    """Una corrida: un escenario con una semilla. Devuelve las metricas de esa corrida."""
    esc, seed = args
    c = _contexto()
    X_test, y_test = c["X_test"], c["y_test"]
    ruta_log = RAIZ / "logs_fase3" / f"{esc['id']}_s{seed}.jsonl"
    sim = SimuladorBanco(X_test, y_test, esc["tandas"], esc["fallo"], seed)
    agente = CampaignTargetAgent(modelos_base()["boosting"], c["X_train"], c["y_train"],
                                 esc["frac"], ruta_log)
    dano = DANOS.get(esc.get("dano"))
    base_n = base_pos = resp_n = resp_pos = 0
    aucs, caps = [], []
    correctos, errores = 0, []
    for i in range(esc["tandas"]):
        ciclo = i + 1
        lote = sim.llegan_clientes(i)
        inyectado = dano is not None and ciclo == CICLO_DANADO
        if inyectado:
            lote = dano(lote)
        try:
            r = agente.ejecutar_ciclo(lote, ciclo, sim)
        except Exception as ex:  # una excepcion no manejada es una falla de robustez
            errores.append(f"{type(ex).__name__}: {ex}")
            continue
        if (r["estado"] == "rechazado") == inyectado:
            correctos += 1
        if r["estado"] != "ok":
            continue
        yl = y_test.loc[r["lote"].index]
        base_n += len(yl); base_pos += int(yl.sum())
        resp_n += len(r["respuestas"]); resp_pos += int(r["respuestas"].sum())
        caps.append(y_test.loc[r["lista"].index].sum() / max(int(yl.sum()), 1))
        if yl.nunique() == 2:
            aucs.append(roc_auc_score(yl, r["probs"].loc[yl.index]))
    completos, procesados = revisar_log(ruta_log)
    base = base_pos / base_n if base_n else np.nan
    conv = resp_pos / resp_n if resp_n else np.nan
    return {"id": esc["id"], "semilla": seed,
            "M1": conv / base - 1 if resp_n else np.nan,
            "M2": float(np.mean(aucs)) if aucs else np.nan,
            "M3": float(np.mean(caps)) if caps else np.nan,
            "M5": completos / procesados if procesados else np.nan,
            "M6": correctos / esc["tandas"],
            "ciclos_ok": len(caps), "alcanzados": resp_n,
            "errores": " | ".join(errores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="2 repeticiones (solo depuracion)")
    ap.add_argument("--escenarios", nargs="*", help="ids a correr (por defecto, todos)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    ruta_umbrales = RAIZ / "umbrales_fase3.json"
    cfg = json.loads(ruta_umbrales.read_text(encoding="utf-8"))
    sha = hashlib.sha256(ruta_umbrales.read_bytes()).hexdigest()
    seeds = cfg["semillas"][:2] if a.rapido else cfg["semillas"]
    escenarios = [e for e in cfg["escenarios"] if not a.escenarios or e["id"] in a.escenarios]
    out = RAIZ / "resultados_fase3"
    out.mkdir(exist_ok=True)
    (RAIZ / "logs_fase3").mkdir(exist_ok=True)

    trabajos = [(e, s) for e in escenarios for s in seeds]
    print(f"{len(escenarios)} escenarios x {len(seeds)} repeticiones = {len(trabajos)} corridas")
    inicio = datetime.now()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        filas = list(ex.map(correr, trabajos))
    runs = pd.DataFrame(filas)
    runs.to_csv(out / "corridas.csv", index=False)

    # ---- agregacion por escenario ----------------------------------------
    medidas = cfg["medidas"]
    meta = {e["id"]: e for e in escenarios}
    por_esc = []
    for eid, g in runs.groupby("id", sort=False):
        e = meta[eid]
        op = e["frac"] == 0.20 and e["tipo"] != "falla"      # punto de operacion del diseno
        fila = {"id": eid, "tipo": e["tipo"], "descripcion": e["descripcion"], "repeticiones": len(g),
                "M1_media": g.M1.mean(), "M1_sd": g.M1.std(ddof=1),
                "M2_media": g.M2.mean(), "M2_sd": g.M2.std(ddof=1),
                "M3_media": g.M3.mean(), "M3_sd": g.M3.std(ddof=1),
                "M5_media": g.M5.mean(), "M6_media": g.M6.mean(),
                "ciclos_ok_media": g.ciclos_ok.mean(), "errores": "; ".join(sorted(set(x for x in g.errores if x)))}
        fila["M1_cumple"] = bool(fila["M1_media"] >= medidas["M1"]["umbral"]) if op else None
        fila["M2_cumple"] = bool(fila["M2_media"] >= medidas["M2"]["umbral"]) if e["tipo"] != "falla" else None
        fila["M3_cumple"] = bool(fila["M3_media"] >= medidas["M3"]["umbral"]) if op else None
        fila["M4_cumple"] = bool(fila["M1_sd"] <= medidas["M4"]["umbral"]) if op else None
        fila["M5_cumple"] = bool(fila["M5_media"] >= medidas["M5"]["umbral"])
        fila["M6_cumple"] = bool(fila["M6_media"] >= medidas["M6"]["umbral"])
        fila["prueba_nula_cumple"] = bool(abs(fila["M1_media"]) <= 0.05) if eid == "L2" else None
        por_esc.append(fila)
    por_esc = pd.DataFrame(por_esc)
    por_esc.to_csv(out / "resultados_por_escenario.csv", index=False)

    # ---- tabla obligatoria: una fila por medida ----------------------------
    tabla = []
    for m, d in medidas.items():
        col_cumple = por_esc[f"{m}_cumple"].dropna()
        if col_cumple.empty:
            continue
        ids = por_esc.loc[col_cumple.index, "id"].tolist()
        sub = runs[runs.id.isin(ids)]
        if m == "M4":
            valor = por_esc.loc[col_cumple.index, "M1_sd"]
            res = f"sd media {valor.mean():.3f} (max {valor.max():.3f})"
        else:
            v = sub[m]
            res = f"media {v.mean():.3f} (sd {v.std(ddof=1):.3f}; min por escenario {por_esc.loc[col_cumple.index, m + '_media'].min():.3f})"
        tabla.append({"medida": f"{m} {d['nombre']}", "formula": d["formula"],
                      "umbral": f"{d['regla']} {d['umbral']}",
                      "escenarios": ",".join(ids), "casos_ejecutados": len(sub),
                      "resultado": res,
                      "escenarios_que_cumplen": f"{int(col_cumple.sum())}/{len(col_cumple)}",
                      "cumple": bool(col_cumple.all()),
                      "evidencia": "resultados_fase3/corridas.csv, resultados_por_escenario.csv, logs_fase3/<id>_s<semilla>.jsonl"})
    pd.DataFrame(tabla).to_csv(out / "tabla_resultados.csv", index=False)

    resumen = {"ejecutado_en": inicio.isoformat(timespec="seconds"),
               "duracion_s": round((datetime.now() - inicio).total_seconds()),
               "umbrales_sha256": sha, "umbrales_fijados_en": cfg["fijado_en"],
               "python": sys.version.split()[0], "sklearn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__, "so": platform.platform(),
               "corridas": len(runs), "logs": len(list((RAIZ / "logs_fase3").glob("*.jsonl")))}
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print("\n== Por escenario ==")
    print(por_esc.drop(columns=["descripcion"]).round(3).to_string(index=False))
    print("\n== Tabla de resultados ==")
    print(pd.DataFrame(tabla)[["medida", "umbral", "casos_ejecutados", "resultado",
                               "escenarios_que_cumplen", "cumple"]].to_string(index=False))
    print("\nResumen:", json.dumps(resumen, ensure_ascii=False))


if __name__ == "__main__":
    main()
