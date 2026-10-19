# CampaignTarget-Agent

Agente de **aprendizaje** que prioriza a los clientes con mayor probabilidad de responder a una campaña de marketing directo, para contactar solo a los más prometedores. Segundo Parcial [Agentes Inteligentes], Inteligencia Artificial, UTB (grupo PA 8).

- **Datos:** [Bank Marketing, UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`data/bank-full.csv`, 45.211 filas, objetivo `y`).
- **Meta SMART:** conversión al menos 20 % superior a una campaña sin priorización, sobre una muestra de prueba de 9.043 clientes.
- **Tecnología:** Python, pandas, scikit-learn. **No usa LLM**: las decisiones salen de un modelo estadístico y de reglas fijas.
- **Documento de diseño ajustado:** `Documento_Diseno_Ajustado_CampaignTarget.pdf` (matriz de decisión, tabla de cambios, sección 9 de desarrollo).

## Ciclo del agente

| Etapa | Método (`CampaignTargetAgent`) | Qué hace |
|---|---|---|
| Percepción (sensor) | `percibir()` | Recibe el lote de clientes y valida el esquema. Rechaza y registra lotes vacíos, con columnas faltantes, tipos inválidos o más de 20 % de datos faltantes. |
| Decisión | `decidir()` | Calcula la probabilidad de compra, ordena y se queda con el top 20 %. |
| Acción (actuador) | `actuar()` | Escribe `lista_ciclo_N.csv` y contacta a los elegidos a través del ambiente (aquí, el simulador). |
| Crítico | `criticar()` | Compara la probabilidad predicha con la respuesta real (Brier y sesgo). |
| Aprendizaje | `aprender()` | Añade las respuestas observadas y reentrena el modelo. |

`ejecutar_ciclo()` encadena las cinco etapas; un lote inválido no detiene al agente.

**Tipo de agente:** de aprendizaje (Russell y Norvig): elemento de desempeño + crítico + elemento de aprendizaje. No tiene generador de problemas.

**Qué razona el código y qué hacen las herramientas:** todo el razonamiento (validación, regla de decisión, crítico, ciclo de aprendizaje) es código propio. scikit-learn solo ajusta y evalúa el modelo (`fit`, `predict_proba`). No hay LLM, ni prompt de sistema.

## Ambiente simulado

`SimuladorBanco` genera el ambiente (no hay acceso a una plataforma real de campañas): entrega los clientes en tandas aleatorias, revela la respuesta real solo **después** de que el agente actúa y hace fallar el 10 % de los contactos (ambiente **estocástico** y **parcialmente observable**).

## Cómo ejecutarlo en otro equipo

```bash
git clone <URL-DEL-REPOSITORIO>
cd campaign-target-agent
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
python campaign_target_agent.py
```

Usa `data/bank-full.csv` (incluido) y, si no existe, lo descarga de UCI. Opciones: `--data ruta/bank-full.csv --tandas 5 --frac 0.20 --fallo 0.10 --seed 42 --out evidencias`.

Salidas en `evidencias/`: `resultados_modelos.csv`, `curva_ganancia.png`, `resultados_simulacion.csv`, `log_ejecucion.jsonl` (5 eventos por ciclo aceptado: percepcion, decision, accion, critica, aprendizaje) y `listas/lista_ciclo_N.csv`.

### Interfaz web

```bash
streamlit run app.py
```

Elige un escenario (típico o de falla), pulsa **Ejecutar agente desde cero** y mira cada ciclo y el log. Para publicarla: Streamlit Community Cloud → *New app* → este repositorio → `app.py`.

## Pruebas de la Fase 3

```bash
python pruebas_fase3.py                       # 15 escenarios x 5 repeticiones = 75 corridas (~8-15 min)
python pruebas_fase3.py --escenarios N1 F2    # subconjunto
python regla_original.py                   # reevalúa con la regla original (umbral en todos los escenarios)
python experimento_aprendizaje.py             # ¿mejora con la experiencia? (agente vs copia congelada)
```

- `umbrales_fase3.json`: fórmulas, umbrales y aplicabilidad por escenario, **fijados antes de ejecutar** (su SHA-256 queda en `resultados_fase3/resumen.json`).
- `resultados_fase3/`: `tabla_resultados.csv` (tabla obligatoria), `resultados_por_escenario.csv`, `corridas.csv`, `experimento_aprendizaje*.csv`, `resumen.json`.
- `logs_fase3/<escenario>_s<semilla>.jsonl`: log de cada corrida (evidencia trazable).
- `salida_pruebas_fase3.txt`: salida de consola de la corrida entregada.

Resultado (Python 3.12.10, scikit-learn 1.9.1): las seis medidas cumplen su umbral en todos los escenarios donde aplican (M1 mejora ≈ +213 % frente a ≥ +20 %; M2 AUC 0,804 frente a ≥ 0,75; M3 captura 62,5 % frente a ≥ 50 %; M4 desviación ≤ 0,085 frente a ≤ 0,10; M5 y M6 al 100 %). Análisis completo, incluidos los límites de la evidencia, en la sección 9.7 del documento.

> Los números pueden variar un poco con otra versión de scikit-learn (el Colab original dio +212 % y 62 % de captura).

## Diferencias frente al diseño de la Fase 1

| Cambio | Justificación |
|---|---|
| Se excluye `duration` | Solo se conoce después de la llamada: usarla sería fuga de información. |
| El actuador contacta a un simulador, no a un CRM | No hay acceso a un sistema real. |
| Ambiente simulado en tandas con 10 % de fallo | Introduce estocasticidad y evita que el agente vea las respuestas antes de decidir. |
| Se añade `criticar()` y 5 eventos de log por ciclo (antes 4) | El diseño describía un crítico que el código no tenía. |
| Validación de entradas en `percibir()` | Hacer probables las fallas esperadas (medida M6, nueva). |
| `class_weight="balanced"` | Las clases están desbalanceadas (88 % no, 12 % sí). |

## Limitaciones

- **Sesgo de retroalimentación:** el agente solo aprende de los clientes que contacta.
- **Backtest, no piloto:** datos históricos de 2008-2010 de una sola entidad portuguesa.
- **El reentrenamiento por ciclo aporta poco con la base completa** (cada ciclo añade ≈ 1 % de datos); solo se nota con una base inicial pequeña (`experimento_aprendizaje.py`).
- **Equidad y privacidad:** edad y estado civil entran al modelo sin auditoría de sesgo.
- En las repeticiones solo varía el ambiente; el modelo inicial y la partición 80/20 son fijos.
- La fracción de contacto (20 %) es fija y no se optimiza contra el costo de la llamada.

## Estructura

```
campaign-target-agent/
├── README.md · requirements.txt · BITACORA_IA.md · .gitignore
├── campaign_target_agent.py     # agente + simulador + programa principal
├── app.py                       # interfaz Streamlit
├── pruebas_fase3.py · umbrales_fase3.json · experimento_aprendizaje.py
├── data/bank-full.csv
├── evidencias/                  # salida de campaign_target_agent.py
├── resultados_fase3/ · logs_fase3/ · salida_pruebas_fase3.txt
└── Documento_Diseno_Ajustado_CampaignTarget.pdf
```

Datos: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 (CC BY 4.0).
