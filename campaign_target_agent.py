#!/usr/bin/env python3
"""CampaignTarget-Agent: agente de aprendizaje que prioriza clientes para marketing directo.

Ciclo del agente (una tanda de clientes = un ciclo):
    percibir -> decidir -> actuar -> criticar -> aprender

Correspondencia con el agente de aprendizaje de Russell y Norvig (AIMA):
    elemento de desempeno  -> percibir() + decidir() + actuar()
    critico                -> criticar()
    elemento de aprendizaje-> aprender()
    generador de problemas -> NO implementado (ver README, limitaciones)

Datos: UCI Bank Marketing (bank-full.csv). Sin LLM: todo es scikit-learn.

Uso:
    python campaign_target_agent.py                      # usa data/bank-full.csv o lo descarga de UCI
    python campaign_target_agent.py --data ruta/bank-full.csv
"""
import argparse
import io
import json
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RUTA_DATOS_DEFECTO = Path(__file__).parent / "data" / "bank-full.csv"


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def descargar_datos(destino: Path) -> Path:
    """Baja bank-full.csv de UCI (zip dentro de zip)."""
    print(f"Descargando datos de {UCI_URL} ...")
    externo = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(UCI_URL).read()))
    interno = zipfile.ZipFile(io.BytesIO(externo.read("bank.zip")))
    destino.mkdir(parents=True, exist_ok=True)
    interno.extract("bank-full.csv", destino)
    return destino / "bank-full.csv"


def cargar_datos(ruta=None) -> pd.DataFrame:
    """Lee bank-full.csv desde `ruta`, desde data/ o lo descarga de UCI."""
    ruta = Path(ruta) if ruta else RUTA_DATOS_DEFECTO
    if not ruta.exists():
        ruta = descargar_datos(RUTA_DATOS_DEFECTO.parent)
    return pd.read_csv(ruta, sep=";")


def preparar(df: pd.DataFrame):
    """Limpieza: sin 'duration' (fuga), y a 0/1, bandera pdays, one-hot (42 columnas)."""
    d = df.drop(columns=["duration"]).copy()  # duration se conoce solo tras la llamada
    d["y"] = (d["y"] == "yes").astype(int)
    d["nunca_contactado"] = (d["pdays"] == -1).astype(int)
    d.loc[d["pdays"] == -1, "pdays"] = 0
    X = pd.get_dummies(d.drop(columns="y"), drop_first=True)
    return X, d["y"]


def dividir(X, y, seed=42):
    """Particion 80/20 estratificada: la misma de la Fase 1 y de las pruebas."""
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def modelos_base():
    return {
        "logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    }


# --------------------------------------------------------------------------
# Ambiente simulado
# --------------------------------------------------------------------------
class SimuladorBanco:
    """Manda clientes por tandas en orden aleatorio y revela la respuesta solo
    DESPUES de que el agente actua. Estocastico: una fraccion de contactos falla
    (cliente no alcanzado). Parcialmente observable: el agente nunca ve y antes de actuar."""

    def __init__(self, X, y, n_tandas=5, tasa_fallo=0.10, seed=42):
        if not 0.0 <= tasa_fallo <= 1.0:
            raise ValueError("tasa_fallo debe estar entre 0 y 1")
        self.rng = np.random.default_rng(seed)
        self.X, self.y = X, y
        self.tandas = np.array_split(self.rng.permutation(X.index), n_tandas)
        self.tasa_fallo = tasa_fallo

    def llegan_clientes(self, i):
        return self.X.loc[self.tandas[i]]

    def contactar(self, ids):
        ids = list(ids)
        ok = self.rng.random(len(ids)) >= self.tasa_fallo if self.tasa_fallo < 1.0 \
            else np.zeros(len(ids), dtype=bool)
        alcanzados = [c for c, o in zip(ids, ok) if o]
        return self.y.loc[alcanzados]


# --------------------------------------------------------------------------
# Agente
# --------------------------------------------------------------------------
class LoteInvalido(ValueError):
    """El lote percibido no cumple el esquema esperado; el ciclo se rechaza."""


class CampaignTargetAgent:
    """Agente de aprendizaje: percibir -> decidir -> actuar -> criticar -> aprender."""

    MAX_FALTANTES = 0.20  # un lote con mas de 20 % de datos faltantes en una columna se rechaza

    def __init__(self, modelo, X_train, y_train, frac=0.20,
                 log_path="log_ejecucion.jsonl", salida_dir=None):
        if not 0.0 < frac <= 1.0:
            raise ValueError("frac debe estar en (0, 1]")
        self.modelo = clone(modelo).fit(X_train, y_train)
        self.X, self.y = X_train.copy(), y_train.copy()
        self.columnas = list(X_train.columns)
        self.frac = frac
        self.log_path = Path(log_path)
        self.salida_dir = Path(salida_dir) if salida_dir else None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")  # log limpio en cada ejecucion
        if self.salida_dir:
            self.salida_dir.mkdir(parents=True, exist_ok=True)

    # ---- registro -----------------------------------------------------
    def _log(self, evento, ciclo, **datos):
        reg = {"timestamp": datetime.now().isoformat(timespec="milliseconds"),
               "ciclo": ciclo, "evento": evento, **datos}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    # ---- PERCEPCION (sensor) -------------------------------------------
    def percibir(self, clientes, ciclo):
        """SENSOR: recibe el lote de clientes candidatos y valida su esquema.
        Lanza LoteInvalido (y registra 'rechazo') si el lote no es utilizable."""
        n = int(len(clientes)) if hasattr(clientes, "__len__") else 0
        self._log("percepcion", ciclo, n_clientes=n)
        motivo = self._validar(clientes)
        if motivo:
            self._log("rechazo", ciclo, motivo=motivo)
            raise LoteInvalido(motivo)
        return clientes[self.columnas]

    def _validar(self, clientes):
        if not isinstance(clientes, pd.DataFrame) or len(clientes) == 0:
            return "lote_vacio"
        faltan = [c for c in self.columnas if c not in clientes.columns]
        if faltan:
            return f"columnas_faltantes:{','.join(faltan[:5])}"
        lote = clientes[self.columnas]
        no_num = [c for c in self.columnas if not pd.api.types.is_numeric_dtype(lote[c])]
        if no_num:
            return f"tipo_invalido:{','.join(no_num[:5])}"
        frac_nan = lote.isna().mean()
        if (frac_nan > self.MAX_FALTANTES).any():
            return f"datos_faltantes:{frac_nan.idxmax()}={frac_nan.max():.2f}"
        return None

    # ---- DECISION ------------------------------------------------------
    def decidir(self, clientes, ciclo):
        """DECISION: probabilidad de compra, ranking y corte al top `frac`."""
        r = clientes.copy()
        r["prob"] = self.modelo.predict_proba(clientes)[:, 1]
        r = r.sort_values("prob", ascending=False)
        self.ultimas_probs = r["prob"].copy()  # puntaje de todo el lote (para evaluar el ranking)
        lista = r.head(max(1, int(len(r) * self.frac)))
        self._log("decision", ciclo, n_elegidos=int(len(lista)),
                  prob_min=round(float(lista["prob"].min()), 4),
                  prob_media=round(float(lista["prob"].mean()), 4))
        return lista

    # ---- ACCION (actuador) ---------------------------------------------
    def actuar(self, lista, ciclo, ambiente):
        """ACTUADOR: exporta la lista priorizada (CSV) y contacta a los elegidos
        a traves del ambiente (aqui, el simulador)."""
        if self.salida_dir:
            lista[["prob"]].to_csv(self.salida_dir / f"lista_ciclo_{ciclo}.csv")
        respuestas = ambiente.contactar(lista.index)
        self._log("accion", ciclo, n_enviados=int(len(lista)), n_alcanzados=int(len(respuestas)))
        return respuestas

    # ---- CRITICO -------------------------------------------------------
    def criticar(self, lista, respuestas, ciclo):
        """CRITICO: compara lo predicho con la respuesta real de los alcanzados."""
        if len(respuestas) == 0:
            self._log("critica", ciclo, n_alcanzados=0, brier=None, sesgo_prob=None)
            return {"brier": None, "sesgo_prob": None}
        p = lista.loc[respuestas.index, "prob"]
        brier = float(((p - respuestas) ** 2).mean())
        sesgo = float(p.mean() - respuestas.mean())  # >0: el agente era optimista
        self._log("critica", ciclo, n_alcanzados=int(len(respuestas)),
                  brier=round(brier, 4), sesgo_prob=round(sesgo, 4))
        return {"brier": brier, "sesgo_prob": sesgo}

    # ---- APRENDIZAJE ---------------------------------------------------
    def aprender(self, lista, respuestas, ciclo):
        """APRENDIZAJE: anade lo observado y reentrena (si hubo respuestas)."""
        if len(respuestas) == 0:
            self._log("aprendizaje", ciclo, conversion=None, reentrenado=False,
                      n_entrenamiento=int(len(self.X)))
            return
        nuevos = lista.loc[respuestas.index].drop(columns="prob")
        self.X = pd.concat([self.X, nuevos])
        self.y = pd.concat([self.y, respuestas])
        self.modelo.fit(self.X, self.y)
        self._log("aprendizaje", ciclo, conversion=round(float(respuestas.mean()), 4),
                  reentrenado=True, n_entrenamiento=int(len(self.X)))

    # ---- CICLO COMPLETO ------------------------------------------------
    def ejecutar_ciclo(self, clientes, ciclo, ambiente):
        """Un ciclo completo. Un lote invalido no detiene al agente: se rechaza,
        se registra y el ciclo termina con estado 'rechazado'."""
        try:
            lote = self.percibir(clientes, ciclo)
        except LoteInvalido as e:
            return {"estado": "rechazado", "motivo": str(e), "ciclo": ciclo}
        lista = self.decidir(lote, ciclo)
        respuestas = self.actuar(lista, ciclo, ambiente)
        critica = self.criticar(lista, respuestas, ciclo)
        self.aprender(lista, respuestas, ciclo)
        return {"estado": "ok", "ciclo": ciclo, "lote": lote, "lista": lista,
                "probs": self.ultimas_probs, "respuestas": respuestas, "critica": critica}


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="ruta a bank-full.csv (si no, data/ o descarga de UCI)")
    ap.add_argument("--out", default="evidencias", help="carpeta de salida")
    ap.add_argument("--tandas", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.20, help="fraccion top a contactar")
    ap.add_argument("--fallo", type=float, default=0.10, help="tasa de contactos fallidos")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(a.data)
    print("Datos:", df.shape)

    X, y = preparar(df)
    X_train, X_test, y_train, y_test = dividir(X, y, a.seed)
    print(f"Train {X_train.shape} | Test {X_test.shape} | conversion {y.mean():.3f}")

    # 1) Comparar modelos
    filas, probs = [], {}
    for nombre, m in modelos_base().items():
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        probs[nombre] = p
        pred = (p >= 0.5).astype(int)
        filas.append({"modelo": nombre,
                      "auc_roc": round(roc_auc_score(y_test, p), 3),
                      "recall": round(recall_score(y_test, pred), 3),
                      "precision": round(precision_score(y_test, pred), 3),
                      "f1": round(f1_score(y_test, pred), 3)})
    res_modelos = pd.DataFrame(filas)
    res_modelos.to_csv(out / "resultados_modelos.csv", index=False)
    print("\n", res_modelos.to_string(index=False))

    # 2) Backtest de lift y curva de ganancia (boosting)
    r = pd.DataFrame({"p": probs["boosting"], "y": y_test.values})
    r = r.sort_values("p", ascending=False).reset_index(drop=True)
    base = r["y"].mean()
    print(f"\nConversion sin priorizar: {base:.3f}")
    for frac in (0.10, 0.20, 0.30):
        top = r["y"].iloc[: int(len(r) * frac)]
        print(f"Top {int(frac*100)}%: conversion {top.mean():.3f} | "
              f"mejora {top.mean()/base-1:+.0%} | captura {top.sum()/r['y'].sum():.0%}")
    cum = r["y"].cumsum() / r["y"].sum()
    xs = np.arange(1, len(r) + 1) / len(r)
    plt.figure(figsize=(6, 5))
    plt.plot(xs, cum, label="Agente")
    plt.plot([0, 1], [0, 1], "--", label="Sin priorizar")
    plt.xlabel("% clientes contactados"); plt.ylabel("% compradores capturados")
    plt.title("Curva de ganancia"); plt.legend()
    plt.savefig(out / "curva_ganancia.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 3) Simulacion del agente en ciclos
    sim = SimuladorBanco(X_test, y_test, a.tandas, a.fallo, a.seed)
    agente = CampaignTargetAgent(modelos_base()["boosting"], X_train, y_train, a.frac,
                                 out / "log_ejecucion.jsonl", out / "listas")
    ciclos = []
    for i in range(a.tandas):
        res = agente.ejecutar_ciclo(sim.llegan_clientes(i), i + 1, sim)
        if res["estado"] != "ok":
            ciclos.append({"ciclo": i + 1, "base": None, "agente": None, "mejora": res["motivo"]})
            continue
        b = y_test.loc[res["lote"].index].mean()  # solo para evaluar, el agente no lo ve
        conv = res["respuestas"].mean()
        ciclos.append({"ciclo": i + 1, "base": round(b, 3), "agente": round(conv, 3),
                       "mejora": f"{conv / b - 1:+.0%}"})
    res_ciclos = pd.DataFrame(ciclos)
    res_ciclos.to_csv(out / "resultados_simulacion.csv", index=False)
    print("\nSimulacion:\n", res_ciclos.to_string(index=False))
    print(f"\nEvidencias guardadas en {out.resolve()}")


if __name__ == "__main__":
    main()
