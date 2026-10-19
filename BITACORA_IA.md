# Bitácora de uso de IA

Equipo PA 8 · Segundo Parcial [Agentes Inteligentes] · Inteligencia Artificial, UTB.

La IA se usó como **asistente de desarrollo y redacción**. El agente entregado **no usa ningún LLM**: sus decisiones salen de scikit-learn y de reglas fijas escritas en el código.

| # | Fecha | Herramienta / modelo | Qué se pidió | Qué produjo | Qué verificó o corrigió el equipo |
|---|---|---|---|---|---|
| 1 | viernes, sabado | Gemini, con Google Colab |  del dataset en colab explorar, limpiar (sin `duration`), entrenar logística y boosting, backtest, clase del agente, simulador y log, borrador de matriz de la Fase 1, empaquetar `campaign_target_agent.py`/README/requirements, llenar la plantilla de diseño y diseñar la suite de pruebas | Código del ciclo percibir → decidir → actuar → aprender, simulador con 10 % de fallo, log JSONL, hoja con la matriz (A = 4,40; Agente B vacío), documento de diseño y suite de 12 escenarios; Gabriel corrió todo en Colab (60 corridas) | se ejecutó cada paso y pegó las salidas; el asistente reconoció que no pudo probar con datos reales y pidió validar en Colab. En la sesión 2 se reprodujeron sus cifras y se corrigieron las carencias descritas abajo |
| 2 | domingo| Claude Code (Claude Sonnet 5.5) | Revisar los avances, verificar las cifras y ayudar a comprobar el entregable de la Fase 1, 2 y 3 | Auditoría del trabajo previo; reejecución del script original; mejora del agente (validación de entradas, crítico, CSV); suite de pruebas `pruebas_fase3.py` y `umbrales_fase3.json`; junto con ayudar a deplegar la app Streamlit; diagramas Mermaid; matriz completa con el Agente B | Se reejecutó el código original desde cero y se compararon las cifras (AUC 0,803 / 0,772 idénticos; mejora +213 % frente a +212 % del Colab por versión de librería). Los valores de los umbrales no cambiaron respecto a la corrida de anterior; la aplicabilidad por escenario y la medida M6 se definieron después de ver esa corrida (se declara en el documento, 9.7) y se fijaron antes de la corrida entregada (hash SHA-256 en `resultados_fase3/resumen.json`) |


