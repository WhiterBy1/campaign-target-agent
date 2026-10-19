#!/usr/bin/env python3
"""Experimento complementario (no es una medida del diseno): ¿mejora el agente con la experiencia?

Con el entrenamiento completo (36.168 clientes) cada ciclo anade ~330 respuestas (≈1 %), asi que el
reentrenamiento apenas se nota. Aqui el agente arranca con pocos datos (N0 clientes) y se compara
contra una copia CONGELADA (mismo modelo inicial, no reentrena) sobre los mismos lotes y contactos.
Si el agente aprende, su mejora y su AUC deben crecer con los ciclos frente al congelado.

    python experimento_aprendizaje.py            # N0 = 300, 20 tandas, 5 semillas
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from campaign_target_agent import (CampaignTargetAgent, SimuladorBanco, cargar_datos, dividir,
                                   modelos_base, preparar)


class AgenteCongelado(CampaignTargetAgent):
    """Control: percibe, decide, actua y critica igual, pero nunca reentrena."""
    def aprender(self, lista, respuestas, ciclo):
        self._log("aprendizaje", ciclo, conversion=None, reentrenado=False, n_entrenamiento=int(len(self.X)))


def correr(clase, n0, tandas, seed, X_train, y_train, X_test, y_test, ruta_log):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), n0, replace=False)
    sim = SimuladorBanco(X_test, y_test, tandas, 0.10, seed)   # mismo ambiente para ambos agentes
    ag = clase(modelos_base()["boosting"], X_train.iloc[idx], y_train.iloc[idx], 0.20, ruta_log)
    filas = []
    for i in range(tandas):
        r = ag.ejecutar_ciclo(sim.llegan_clientes(i), i + 1, sim)
        yl = y_test.loc[r["lote"].index]
        filas.append({"ciclo": i + 1, "mejora": r["respuestas"].mean() / yl.mean() - 1,
                      "auc": roc_auc_score(yl, r["probs"].loc[yl.index])})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=300)
    ap.add_argument("--tandas", type=int, default=20)
    ap.add_argument("--semillas", type=int, default=5)
    a = ap.parse_args()
    X, y = preparar(cargar_datos())
    X_train, X_test, y_train, y_test = dividir(X, y, 42)
    out = Path("resultados_fase3"); out.mkdir(exist_ok=True)
    tmp = out / "_tmp.jsonl"
    res = []
    for s in range(1, a.semillas + 1):
        for nombre, clase in (("aprende", CampaignTargetAgent), ("congelado", AgenteCongelado)):
            d = correr(clase, a.n0, a.tandas, s, X_train, y_train, X_test, y_test, tmp)
            d["agente"], d["semilla"] = nombre, s
            res.append(d)
            print("ok", nombre, s, flush=True)
    tmp.unlink()
    res = pd.concat(res)
    res.to_csv(out / "experimento_aprendizaje.csv", index=False)
    k = 5
    resumen = []
    for ag, g in res.groupby("agente"):
        for tramo, sel in (("ciclos 1-5", g.ciclo <= k), ("ciclos 16-20", g.ciclo > a.tandas - k)):
            m = g[sel].groupby("semilla")[["mejora", "auc"]].mean()
            resumen.append({"agente": ag, "tramo": tramo, "mejora_media": m.mejora.mean(), "mejora_sd": m.mejora.std(ddof=1),
                            "auc_medio": m.auc.mean(), "auc_sd": m.auc.std(ddof=1)})
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(out / "experimento_aprendizaje_resumen.csv", index=False)
    print(resumen.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
