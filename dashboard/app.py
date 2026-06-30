import time

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Segmentación Streaming EA3",
    layout="wide"
)

st.title("Segmentación de Usuarios de Streaming")
st.caption("Solución end-to-end con ETL de tres fuentes, KMeans, PCA, FastAPI, Streamlit y Docker.")


def obtener_datos_dashboard():
    url = "http://ml-service:8000/dashboard-data"

    for intento in range(1, 11):
        try:
            respuesta = requests.get(url, timeout=5)
            if respuesta.status_code == 200:
                return respuesta.json()
            st.warning(f"ML Service respondió con estado {respuesta.status_code}. Intento {intento}/10.")
        except requests.exceptions.RequestException:
            st.warning(f"Esperando respuesta del ML Service. Intento {intento}/10.")

        time.sleep(2)

    return None


payload = obtener_datos_dashboard()

if payload is None:
    st.error("No fue posible obtener datos desde el ML Service.")
    st.stop()

data = pd.DataFrame(payload["usuarios"])
centroides = pd.DataFrame(payload["centroides"])
perfil_segmentos = pd.DataFrame(payload["perfil_segmentos"])
reporte_etl = pd.DataFrame(payload["reporte_etl"])
metricas = payload["metricas"]

st.subheader("Métricas principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Silhouette Score", f"{metricas['silhouette_score']:.3f}")
with col2:
    st.metric("Clusters", metricas["n_clusters"])
with col3:
    st.metric("Usuarios", metricas["n_usuarios"])
with col4:
    st.metric("Varianza PCA", f"{metricas['varianza_total_pca']:.2%}")
with col5:
    st.metric("Fuentes integradas", metricas["fuentes_integradas"])

tab1, tab2, tab3 = st.tabs(["Vista ejecutiva", "Vista técnica", "Vista operativa"])

with tab1:
    st.subheader("Vista ejecutiva")

    st.write(
        "Esta vista resume la segmentación para apoyar decisiones comerciales "
        "en una plataforma de streaming."
    )

    st.subheader("Distribución de usuarios por segmento")
    distribucion = data["cluster"].value_counts().sort_index()
    st.bar_chart(distribucion)

    st.subheader("Perfil promedio de segmentos")
    st.dataframe(perfil_segmentos, use_container_width=True)

    st.subheader("Interpretación de negocio")
    st.info(
        "Los segmentos permiten identificar perfiles de usuarios según consumo, gasto, "
        "satisfacción, promociones, antigüedad y uso de la aplicación."
    )

with tab2:
    st.subheader("Vista técnica")

    st.write("Reporte de integración ETL")
    st.dataframe(reporte_etl, use_container_width=True)

    st.subheader("Método del codo")
    df_codo = pd.DataFrame({
        "k": metricas["valores_k"],
        "inercia": metricas["inercias"]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_codo["k"], df_codo["inercia"], marker="o")
    ax.set_title("Método del codo", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cantidad de clusters K")
    ax.set_ylabel("Inercia")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Visualización PCA")
    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in sorted(data["cluster"].unique()):
        subset = data[data["cluster"] == cluster]
        ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster}", alpha=0.7)

    ax.set_title("Visualización PCA de segmentos", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab3:
    st.subheader("Vista operativa")

    st.subheader("Usuarios segmentados")
    st.dataframe(data, use_container_width=True)

    st.subheader("Clusters según consumo y gasto")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        data["horas_consumo_mensual"],
        data["gasto_mensual"],
        c=data["cluster"],
        alpha=0.7,
        s=40
    )

    ax.scatter(
        centroides["horas_consumo_mensual"],
        centroides["gasto_mensual"],
        marker="X",
        s=250,
        edgecolor="black",
        linewidth=2
    )

    ax.set_xlabel("horas_consumo_mensual")
    ax.set_ylabel("gasto_mensual")
    ax.set_title("Clusters según consumo y gasto", fontsize=14, fontweight="bold")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Predicción de segmento")

    with st.form("form_prediccion"):
        horas_consumo_mensual = st.number_input("Horas de consumo mensual", value=40.0)
        gasto_mensual = st.number_input("Gasto mensual", value=12.0)
        cantidad_contenidos_vistos = st.number_input("Cantidad de contenidos vistos", value=30.0)
        sesiones_semana = st.number_input("Sesiones por semana", value=5.0)
        porcentaje_finalizacion = st.number_input("Porcentaje de finalización", value=70.0)
        tiempo_promedio_sesion_min = st.number_input("Tiempo promedio de sesión", value=45.0)
        cantidad_generos_consumidos = st.number_input("Cantidad de géneros consumidos", value=4.0)
        porcentaje_uso_promociones = st.number_input("Porcentaje uso promociones", value=20.0)
        antiguedad_cliente_meses = st.number_input("Antigüedad cliente meses", value=18.0)
        edad = st.number_input("Edad", value=30.0)
        dispositivos_registrados = st.number_input("Dispositivos registrados", value=2.0)
        porcentaje_uso_app_movil = st.number_input("Porcentaje uso app móvil", value=60.0)
        cantidad_perfiles_creados = st.number_input("Cantidad perfiles creados", value=2.0)
        interacciones_mensuales_soporte = st.number_input("Interacciones mensuales soporte", value=1.0)
        distancia_promedio_red_km = st.number_input("Distancia promedio red km", value=20.0)
        satisfaccion_usuario = st.number_input("Satisfacción usuario", value=80.0)
        reclamos_ultimos_3_meses = st.number_input("Reclamos últimos 3 meses", value=1.0)
        plan_streaming_codigo = st.number_input("Código plan streaming", value=2.0)
        dispositivo_principal_codigo = st.number_input("Código dispositivo principal", value=2.0)

        enviar = st.form_submit_button("Predecir segmento")

    if enviar:
        entrada = {
            "horas_consumo_mensual": horas_consumo_mensual,
            "gasto_mensual": gasto_mensual,
            "cantidad_contenidos_vistos": cantidad_contenidos_vistos,
            "sesiones_semana": sesiones_semana,
            "porcentaje_finalizacion": porcentaje_finalizacion,
            "tiempo_promedio_sesion_min": tiempo_promedio_sesion_min,
            "cantidad_generos_consumidos": cantidad_generos_consumidos,
            "porcentaje_uso_promociones": porcentaje_uso_promociones,
            "antiguedad_cliente_meses": antiguedad_cliente_meses,
            "edad": edad,
            "dispositivos_registrados": dispositivos_registrados,
            "porcentaje_uso_app_movil": porcentaje_uso_app_movil,
            "cantidad_perfiles_creados": cantidad_perfiles_creados,
            "interacciones_mensuales_soporte": interacciones_mensuales_soporte,
            "distancia_promedio_red_km": distancia_promedio_red_km,
            "satisfaccion_usuario": satisfaccion_usuario,
            "reclamos_ultimos_3_meses": reclamos_ultimos_3_meses,
            "plan_streaming_codigo": plan_streaming_codigo,
            "dispositivo_principal_codigo": dispositivo_principal_codigo
        }

        try:
            respuesta = requests.post("http://ml-service:8000/predict", json=entrada, timeout=5)
            resultado = respuesta.json()
            st.success(f"El usuario pertenece al cluster {resultado['cluster']}")
        except requests.exceptions.RequestException:
            st.error("No fue posible realizar la predicción.")
