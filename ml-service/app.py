"""Servicio ML para la segmentación de usuarios de streaming.

Expone un endpoint raíz, un endpoint de salud, un endpoint que consolida
los datos del entrenamiento para el dashboard y un endpoint de predicción
de cluster para nuevos usuarios.
"""

import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="Servicio ML - Segmentación de Usuarios Streaming",
    description="API para consultar métricas, segmentos y predicciones del modelo KMeans.",
    version="1.0"
)


class UsuarioEntrada(BaseModel):
    """Esquema de entrada para el endpoint ``/predict``.

    Contiene las variables (ya transformadas a códigos numéricos para
    las categóricas) necesarias para asignar un cluster a un usuario.
    """

    horas_consumo_mensual: float
    gasto_mensual: float
    cantidad_contenidos_vistos: float
    sesiones_semana: float
    porcentaje_finalizacion: float
    tiempo_promedio_sesion_min: float
    cantidad_generos_consumidos: float
    porcentaje_uso_promociones: float
    antiguedad_cliente_meses: float
    edad: float
    dispositivos_registrados: float
    porcentaje_uso_app_movil: float
    cantidad_perfiles_creados: float
    interacciones_mensuales_soporte: float
    distancia_promedio_red_km: float
    satisfaccion_usuario: float
    reclamos_ultimos_3_meses: float
    plan_streaming_codigo: float
    dispositivo_principal_codigo: float


def cargar_pickle(ruta):
    """Carga un objeto serializado desde disco usando pickle.

    Parameters
    ----------
    ruta : str
        Ruta relativa del archivo ``.pkl`` a deserializar.

    Returns
    -------
    object
        Objeto persistido (modelo, scaler, PCA o métricas).
    """
    with open(ruta, "rb") as archivo:
        return pickle.load(archivo)


@app.get("/")
def home():
    """Endpoint de verificación del servicio ML."""
    return {"mensaje": "Servicio ML funcionando"}


@app.get("/health")
def health():
    """Endpoint de salud para orquestación con Docker."""
    return {"estado": "ok"}


@app.get("/dashboard-data")
def dashboard_data():
    """Consolida los artefactos del entrenamiento para el dashboard.

    Returns
    -------
    dict
        Usuarios segmentados, centroides, perfil de segmentos,
        reporte ETL y métricas del modelo.
    """
    try:
        usuarios = pd.read_csv("data/transformados/usuarios_segmentados.csv")
        centroides = pd.read_csv("data/transformados/centroides.csv")
        perfil_segmentos = pd.read_csv("data/transformados/perfil_segmentos.csv")
        reporte_etl = pd.read_csv("data/transformados/reporte_etl.csv")
        metricas = cargar_pickle("models/metricas.pkl")

        return {
            "usuarios": usuarios.to_dict(orient="records"),
            "centroides": centroides.to_dict(orient="records"),
            "perfil_segmentos": perfil_segmentos.to_dict(orient="records"),
            "reporte_etl": reporte_etl.to_dict(orient="records"),
            "metricas": metricas
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se encontraron archivos generados por el entrenamiento: {error}"
        )


@app.post("/predict")
def predict(usuario: UsuarioEntrada):
    """Predice el segmento (cluster) de un usuario.

    Parameters
    ----------
    usuario : UsuarioEntrada
        Datos del usuario a clasificar.

    Returns
    -------
    dict
        Identificador del cluster asignado, por ejemplo ``{"cluster": 2}``.
    """
    try:
        modelo = cargar_pickle("models/modelo_kmeans.pkl")
        scaler = cargar_pickle("models/scaler.pkl")

        entrada = pd.DataFrame([usuario.dict()])
        entrada_escalada = scaler.transform(entrada)
        cluster = modelo.predict(entrada_escalada)[0]

        return {"cluster": int(cluster)}

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se encontró el modelo serializado: {error}"
        )
