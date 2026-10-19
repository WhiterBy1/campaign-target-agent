# Bitácora de uso de IA

Equipo PA 8 · Segundo Parcial [Agentes Inteligentes] · Inteligencia Artificial, UTB.

La IA se usó como **asistente de desarrollo y redacción**. El agente entregado **no usa ningún LLM**: sus decisiones salen de scikit-learn y de reglas fijas escritas en el código.

| # | Fecha | Herramienta / modelo | Qué se pidió | Qué produjo | Qué verificó o corrigió el equipo |
|---|---|---|---|---|---|
| 1 | viernes, sabado | Gemini, con Google Colab |  del dataset en colab explorar, limpiar (sin `duration`), entrenar logística y boosting, backtest, clase del agente, simulador y log, borrador de matriz de la Fase 1, empaquetar `campaign_target_agent.py`/README/requirements, llenar la plantilla de diseño y diseñar la suite de pruebas | Código del ciclo percibir → decidir → actuar → aprender, simulador con 10 % de fallo, log JSONL, hoja con la matriz (A = 4,40; Agente B vacío), documento de diseño y suite de 12 escenarios; Gabriel corrió todo en Colab (60 corridas) | se ejecutó cada paso y pegó las salidas; el asistente reconoció que no pudo probar con datos reales y pidió validar en Colab. En la sesión 2 se reprodujeron sus cifras y se corrigieron las carencias descritas abajo |
| 2 | domingo| Claude Code (Claude Sonnet 5.5), sesión de la actividad | Revisar los avances, verificar las cifras y completar el entregable de la Fase 1, 2 y 3 | Auditoría del trabajo previo; reejecución del script original; mejora del agente (validación de entradas, crítico, CSV); suite de pruebas `pruebas_fase3.py` y `umbrales_fase3.json`; app Streamlit; diagramas Mermaid; generador del documento PDF; matriz completa con el Agente B | Se reejecutó el código original desde cero y se compararon las cifras (AUC 0,803 / 0,772 idénticos; mejora +213 % frente a +212 % del Colab por versión de librería). Los valores de los umbrales no cambiaron respecto a la corrida de Gabriel; la aplicabilidad por escenario y la medida M6 se definieron después de ver esa corrida (se declara en el documento, 9.7) y se fijaron antes de la corrida entregada (hash SHA-256 en `resultados_fase3/resumen.json`) |

## Hallazgos de la auditoría del trabajo previo

1. El primer borrador de la matriz daba 4,40 a CampaignTarget (Medibilidad 4, Complejidad 3), pero el PDF «Fase 1» traía 4,75; ambos puntajes superan al Agente B (3,15). La matriz solo puntuaba a CampaignTarget; el Agente B estaba vacío, por lo que la elección no era reconstruible. Se puntuó el Symptom Triage Agent con evidencia propia y se recalculó la sensibilidad.
2. El diseño describía un crítico y una salida CSV que no existían en el script del repositorio. Se implementaron.
3. La validación de entradas era inexistente: un lote con una columna faltante detenía el ciclo con una excepción. Ahora el agente lo rechaza y lo registra.
4. El bloque de pruebas del notebook guardado no traía salidas (Gabriel lo corrió en Colab y pegó los resultados en el chat) y aplicaba umbrales como M1 ≥ 20 % también a escenarios donde no tienen sentido (contactar al 100 %). Se reescribió con aplicabilidad declarada por escenario.
5. Las cifras del README y del diseño (AUC 0,803, +212 %, captura 62 %) son reales; se actualizaron a la reejecución (+213 %, 63 %).

## Decisiones que son del equipo (no de la IA)

- Elegir CampaignTarget-Agent frente a Symptom Triage Agent: la matriz lo respalda, pero la decisión y los puntajes deben ser revisados y asumidos por ambos integrantes.
- Qué opción de evidencia usar (despliegue o video) y grabar la demo con cámara y voz de ambos.
- Revisar cómo se incorporó la retroalimentación del docente (documento incompleto en la Tarea 2): el PDF ahora es autocontenido y trae el anexo del dataset como Anexo A.
