"""Interfaz Streamlit de CampaignTarget-Agent (evidencia de funcionamiento, opcion A).

    streamlit run app.py

Ejecuta el agente desde cero con los parametros elegidos, muestra cada ciclo
(percepcion -> decision -> accion -> critica -> aprendizaje) y el log con marcas de tiempo.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from campaign_target_agent import (CampaignTargetAgent, SimuladorBanco, cargar_datos, dividir,
                                   modelos_base, preparar)
from pruebas_fase3 import DANOS

st.set_page_config(page_title="CampaignTarget-Agent", page_icon="🎯", layout="wide")


@st.cache_data(show_spinner="Cargando y preparando bank-full.csv ...")
def datos():
    X, y = preparar(cargar_datos())
    return dividir(X, y, 42)


X_train, X_test, y_train, y_test = datos()

st.title("🎯 CampaignTarget-Agent")
st.caption("Agente de aprendizaje que prioriza a los clientes con mayor probabilidad de responder "
           "a una campaña de marketing directo. Datos: UCI Bank Marketing (ambiente simulado, "
           "estocástico y parcialmente observable). Sin LLM: scikit-learn + reglas fijas.")

with st.sidebar:
    st.header("Parámetros de campaña")
    preset = st.radio("Escenario", ["Típico (diseño)", "Falla: lote sin columna 'balance'",
                                    "Falla: lote vacío", "Falla: 90 % de contactos fallidos",
                                    "Personalizado"])
    base = {"frac": 0.20, "fallo": 0.10, "tandas": 5, "dano": "ninguno"}
    if preset.startswith("Falla: lote sin"):
        base["dano"] = "sin_columna"
    elif preset == "Falla: lote vacío":
        base["dano"] = "vacio"
    elif preset.startswith("Falla: 90"):
        base["fallo"] = 0.90
    custom = preset == "Personalizado"
    frac = st.slider("Fracción a contactar", 0.01, 1.0, base["frac"], 0.01, disabled=not custom)
    fallo = st.slider("Tasa de contactos fallidos", 0.0, 1.0, base["fallo"], 0.05, disabled=not custom)
    tandas = st.slider("Número de tandas (ciclos)", 2, 20, base["tandas"], disabled=not custom)
    opciones = ["ninguno"] + list(DANOS)
    dano = st.selectbox("Daño inyectado en el lote 2", opciones, opciones.index(base["dano"]),
                        disabled=not custom)
    seed = st.number_input("Semilla del ambiente", 0, 10_000, 1)
    ejecutar = st.button("▶ Ejecutar agente desde cero", type="primary")

tab_sim, tab_res, tab_ayuda = st.tabs(["Simulación", "Resultados de la Fase 3", "Cómo probar"])

with tab_sim:
    if not ejecutar:
        st.info("Elige un escenario en la barra lateral y pulsa **Ejecutar agente desde cero**.")
    else:
        frac_, fallo_, tandas_, dano_ = ((frac, fallo, tandas, dano) if custom
                                         else (base["frac"], base["fallo"], base["tandas"], base["dano"]))
        log_path = Path(tempfile.mkdtemp()) / "log_ejecucion.jsonl"
        sim = SimuladorBanco(X_test, y_test, tandas_, fallo_, int(seed))
        with st.spinner("Entrenando el modelo inicial (HistGradientBoosting) ..."):
            agente = CampaignTargetAgent(modelos_base()["boosting"], X_train, y_train, frac_, log_path)
        filas, barra = [], st.progress(0.0)
        for i in range(tandas_):
            lote = sim.llegan_clientes(i)
            if dano_ != "ninguno" and i + 1 == 2:
                lote = DANOS[dano_](lote)
            r = agente.ejecutar_ciclo(lote, i + 1, sim)
            if r["estado"] == "ok":
                b = float(y_test.loc[r["lote"].index].mean())
                c = float(r["respuestas"].mean()) if len(r["respuestas"]) else np.nan
                filas.append({"ciclo": i + 1, "estado": "ok", "clientes": len(r["lote"]),
                              "contactados": len(r["lista"]), "alcanzados": len(r["respuestas"]),
                              "conversión base": b, "conversión agente": c,
                              "mejora": c / b - 1 if b else np.nan,
                              "brier (crítico)": r["critica"]["brier"]})
            else:
                filas.append({"ciclo": i + 1, "estado": f"RECHAZADO · {r['motivo']}"})
            barra.progress((i + 1) / tandas_)
        df = pd.DataFrame(filas)
        ok = df[df.estado == "ok"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Ciclos aceptados", f"{len(ok)}/{tandas_}")
        if len(ok) and ok["mejora"].notna().any():
            c2.metric("Mejora media de conversión", f"{ok['mejora'].mean():+.0%}",
                      help="Meta SMART: ≥ +20 % frente a contactar sin priorizar")
            c3.metric("Conversión media del agente", f"{ok['conversión agente'].mean():.1%}",
                      f"base {ok['conversión base'].mean():.1%}")
        st.dataframe(df.style.format({"conversión base": "{:.3f}", "conversión agente": "{:.3f}",
                                      "mejora": "{:+.0%}", "brier (crítico)": "{:.3f}"}, na_rep="—"),
                     width="stretch", hide_index=True)
        st.subheader("Registro de ejecución (log_ejecucion.jsonl)")
        log = pd.DataFrame([json.loads(l) for l in open(log_path, encoding="utf-8")])
        st.dataframe(log, width="stretch", hide_index=True)
        st.download_button("Descargar log", open(log_path, "rb").read(), "log_ejecucion.jsonl")

with tab_res:
    carpeta = Path(__file__).parent / "resultados_fase3"
    if (carpeta / "tabla_resultados.csv").exists():
        st.markdown("**Tabla de resultados** (generada por `python pruebas_fase3.py`)")
        st.dataframe(pd.read_csv(carpeta / "tabla_resultados.csv").drop(columns=["escenarios", "evidencia"]),
                     width="stretch", hide_index=True)
        st.markdown("**Resultados por escenario**")
        st.dataframe(pd.read_csv(carpeta / "resultados_por_escenario.csv").round(3),
                     width="stretch", hide_index=True)
    else:
        st.warning("Aún no hay resultados: ejecuta `python pruebas_fase3.py`.")

with tab_ayuda:
    st.markdown("""
**Casos de prueba sugeridos**
1. *Escenario típico* (por defecto): 5 tandas, 20 % de contacto, 10 % de fallos. Esperado: mejora ≈ +200 %, 5/5 ciclos aceptados.
2. *Falla: lote sin columna `balance`*: el ciclo 2 se rechaza con `columnas_faltantes`, el agente sigue con los ciclos 3-5.
3. *Falla: lote vacío*: el ciclo 2 se rechaza con `lote_vacio`.
4. *Falla: 90 % de contactos fallidos*: el agente aprende de muy pocas respuestas; no se rompe.
5. *Personalizado*: `fracción = 1.0` contacta a todos y la mejora debe ser ≈ 0 (prueba nula).

El agente mantiene estado interno (el conjunto de entrenamiento crece en cada ciclo) y reentrena su modelo: es un agente de aprendizaje.
""")
