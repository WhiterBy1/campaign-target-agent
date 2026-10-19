#!/usr/bin/env python3
"""Reevalua corridas.csv con la regla ORIGINAL de la corrida inicial en Colab: cada umbral
M1-M5 aplicado a TODOS los escenarios (sin aplicabilidad por escenario). Salida: resultados_fase3/resultados_regla_original.csv"""
import pandas as pd
r = pd.read_csv("resultados_fase3/corridas.csv")
U = {"M1": (">=", .20), "M2": (">=", .75), "M3": (">=", .50), "M4": ("<=", .10), "M5": (">=", 1.0)}
g = r.groupby("id", sort=False)
t = pd.DataFrame({"M1": g.M1.mean(), "M2": g.M2.mean(), "M3": g.M3.mean(), "M4": g.M1.std(ddof=1), "M5": g.M5.mean()})
ok = pd.DataFrame({m: (t[m] >= u if op == ">=" else t[m] <= u) for m, (op, u) in U.items()})
t["incumple"] = ok.apply(lambda f: ",".join(m for m in U if not f[m]) or "—", axis=1)
t.round(3).to_csv("resultados_fase3/resultados_regla_original.csv")
print(t.round(3).to_string())
