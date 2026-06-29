"""Pipeline de integración ETL y entrenamiento del modelo de segmentación.

Este módulo integra tres fuentes de datos (CSV, PostgreSQL y API REST),
valida esquemas y calidad, entrena un modelo KMeans con selección óptima
del número de clusters mediante el método del codo, reduce dimensionalidad
con PCA y persiste todos los artefactos para el servicio ML.
"""

import os
import time
import pickle
import logging

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "streaming_clientes")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
API_SOURCE_URL = os.getenv("API_SOURCE_URL", "http://api-source:7000/usuarios-api")

CSV_PATH = "data/originales/usuarios_streaming.csv"
DATA_INTEGRADA_PATH = "data/transformados/data_usuarios.csv"
DATA_SEGMENTADA_PATH = "data/transformados/usuarios_segmentados.csv"
CENTROIDES_PATH = "data/transformados/centroides.csv"
ETL_REPORT_PATH = "data/transformados/reporte_etl.csv"

MODEL_PATH = "models/modelo_kmeans.pkl"
SCALER_PATH = "models/scaler.pkl"
PCA_PATH = "models/pca.pkl"
METRICAS_PATH = "models/metricas.pkl"


COLUMNAS_USUARIOS = [
    "id_cliente",
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses"
]

COLUMNAS_PERFIL = [
    "id_cliente",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km"
]

COLUMNAS_API = [
    "id_cliente",
    "plan_streaming",
    "satisfaccion_usuario",
    "reclamos_ultimos_3_meses",
    "dispositivo_principal"
]

COLUMNAS_MODELO = [
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
    "satisfaccion_usuario",
    "reclamos_ultimos_3_meses",
    "plan_streaming_codigo",
    "dispositivo_principal_codigo"
]


def crear_engine():
    """Crea un motor de conexión SQLAlchemy hacia PostgreSQL.

    Returns
    -------
    sqlalchemy.engine.Engine
        Motor de conexión configurado con las variables de entorno.
    """
    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def esperar_postgres(intentos=30, segundos=2):
    """Espera a que PostgreSQL esté disponible antes de continuar.

    Parameters
    ----------
    intentos : int, optional
        Número máximo de reintentos (por defecto 30).
    segundos : int, optional
        Segundos de espera entre reintentos (por defecto 2).

    Returns
    -------
    sqlalchemy.engine.Engine
        Motor de conexión una vez confirmada la disponibilidad.

    Raises
    ------
    ConnectionError
        Si no se logra conectar tras los reintentos definidos.
    """
    engine = crear_engine()

    for intento in range(1, intentos + 1):
        try:
            with engine.connect() as conexion:
                conexion.execute(text("SELECT 1"))
            logging.info("Conexión a PostgreSQL establecida correctamente.")
            return engine
        except OperationalError:
            logging.warning("PostgreSQL no está listo. Intento %s/%s", intento, intentos)
            time.sleep(segundos)

    raise ConnectionError("No fue posible conectar con PostgreSQL.")


def esperar_api(intentos=30, segundos=2):
    """Espera a que la API fuente externa esté disponible y devuelve su payload.

    Parameters
    ----------
    intentos : int, optional
        Número máximo de reintentos (por defecto 30).
    segundos : int, optional
        Segundos de espera entre reintentos (por defecto 2).

    Returns
    -------
    dict
        Respuesta JSON de ``GET /usuarios-api``.

    Raises
    ------
    ConnectionError
        Si no se logra una respuesta satisfactoria tras los reintentos.
    """
    for intento in range(1, intentos + 1):
        try:
            respuesta = requests.get(API_SOURCE_URL, timeout=5)
            if respuesta.status_code == 200:
                logging.info("API fuente externa disponible.")
                return respuesta.json()

            logging.warning("API respondió estado %s. Intento %s/%s", respuesta.status_code, intento, intentos)

        except requests.exceptions.RequestException:
            logging.warning("API externa no está lista. Intento %s/%s", intento, intentos)

        time.sleep(segundos)

    raise ConnectionError("No fue posible conectar con la API externa.")


def validar_columnas(data, columnas, nombre_fuente):
    """Valida que un DataFrame contenga las columnas esperadas.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame a validar.
    columnas : list[str]
        Lista de columnas obligatorias.
    nombre_fuente : str
        Nombre identificador de la fuente (para mensajes de error).

    Raises
    ------
    ValueError
        Si faltan columnas respecto a las esperadas.
    """
    faltantes = [col for col in columnas if col not in data.columns]
    if faltantes:
        raise ValueError(f"La fuente {nombre_fuente} no contiene las columnas requeridas: {faltantes}")


def validar_calidad(data, nombre_fuente, columna_id="id_cliente"):
    """Valida la calidad básica de una fuente de datos.

    Comprueba que el DataFrame no esté vacío, que la columna identificador
    no contenga nulos y que no haya identificadores duplicados.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame a validar.
    nombre_fuente : str
        Nombre identificador de la fuente.
    columna_id : str, optional
        Columna usada como identificador (por defecto ``id_cliente``).

    Raises
    ------
    ValueError
        Si la fuente está vacía, tiene nulos en el id o ids duplicados.
    """
    if data.empty:
        raise ValueError(f"La fuente {nombre_fuente} está vacía.")

    if data[columna_id].isna().any():
        raise ValueError(f"La fuente {nombre_fuente} tiene valores nulos en {columna_id}.")

    if data[columna_id].duplicated().any():
        raise ValueError(f"La fuente {nombre_fuente} tiene {columna_id} duplicados.")


def cargar_csv():
    """Carga y valida la fuente 1 (CSV de comportamiento de consumo).

    Returns
    -------
    pandas.DataFrame
        Datos de usuarios validados.
    """
    logging.info("Leyendo fuente 1: CSV usuarios_streaming.csv.")
    usuarios = pd.read_csv(CSV_PATH)
    validar_columnas(usuarios, COLUMNAS_USUARIOS, "usuarios_streaming.csv")
    validar_calidad(usuarios, "usuarios_streaming.csv")
    return usuarios


def cargar_postgres():
    """Carga y valida la fuente 2 (PostgreSQL ``perfil_usuarios``).

    Returns
    -------
    pandas.DataFrame
        Perfil de usuarios validado.
    """
    logging.info("Leyendo fuente 2: PostgreSQL perfil_usuarios.")
    engine = esperar_postgres()
    perfil = pd.read_sql("SELECT * FROM perfil_usuarios", engine)
    validar_columnas(perfil, COLUMNAS_PERFIL, "perfil_usuarios")
    validar_calidad(perfil, "perfil_usuarios")
    return perfil


def cargar_api():
    """Carga y valida la fuente 3 (API REST ``/usuarios-api``).

    Returns
    -------
    pandas.DataFrame
        Datos comerciales validados.
    """
    logging.info("Leyendo fuente 3: API REST usuarios-api.")
    payload = esperar_api()
    api_data = pd.DataFrame(payload["usuarios_api"])
    validar_columnas(api_data, COLUMNAS_API, "API REST usuarios_api")
    validar_calidad(api_data, "API REST usuarios_api")
    return api_data


def transformar_api(api_data):
    """Codifica las variables categóricas de la fuente API a numéricas.

    Parameters
    ----------
    api_data : pandas.DataFrame
        Datos crudos de la API.

    Returns
    -------
    pandas.DataFrame
        Datos con las columnas ``plan_streaming_codigo`` y
        ``dispositivo_principal_codigo`` añadidas.
    """
    plan_map = {
        "Basico": 1,
        "Estandar": 2,
        "Premium": 3,
        "Familiar": 4
    }

    dispositivo_map = {
        "Smart TV": 1,
        "Celular": 2,
        "Notebook": 3,
        "Tablet": 4,
        "Consola": 5
    }

    api_data["plan_streaming_codigo"] = api_data["plan_streaming"].map(plan_map)
    api_data["dispositivo_principal_codigo"] = api_data["dispositivo_principal"].map(dispositivo_map)

    validar_columnas(
        api_data,
        ["plan_streaming_codigo", "dispositivo_principal_codigo"],
        "API transformada"
    )

    return api_data


def integrar_fuentes():
    """Integra las tres fuentes (CSV + PostgreSQL + API) y genera el dataset analítico.

    Realiza ``inner join`` por ``id_cliente``, elimina nulos y persiste
    el dataset integrado y un reporte ETL.

    Returns
    -------
    pandas.DataFrame
        Dataset analítico integrado y limpio.

    Raises
    ------
    ValueError
        Si el dataset integrado queda vacío antes o después de limpiar nulos.
    """
    usuarios = cargar_csv()
    perfil = cargar_postgres()
    api_data = cargar_api()
    api_data = transformar_api(api_data)

    logging.info("Integrando CSV + PostgreSQL mediante id_cliente.")
    data = pd.merge(
        usuarios,
        perfil,
        on="id_cliente",
        how="inner"
    )

    logging.info("Integrando resultado con API REST mediante id_cliente.")
    data = pd.merge(
        data,
        api_data,
        on="id_cliente",
        how="inner"
    )

    if data.empty:
        raise ValueError("El dataset integrado quedó vacío.")

    data = data.dropna()

    if data.empty:
        raise ValueError("El dataset quedó vacío después de eliminar nulos.")

    data.to_csv(DATA_INTEGRADA_PATH, index=False)

    reporte = pd.DataFrame(
        [
            {"fuente": "CSV usuarios_streaming", "registros": len(usuarios), "columnas": len(usuarios.columns)},
            {"fuente": "PostgreSQL perfil_usuarios", "registros": len(perfil), "columnas": len(perfil.columns)},
            {"fuente": "API REST usuarios_api", "registros": len(api_data), "columnas": len(api_data.columns)},
            {"fuente": "Dataset integrado", "registros": len(data), "columnas": len(data.columns)}
        ]
    )

    reporte.to_csv(ETL_REPORT_PATH, index=False)

    logging.info("Dataset integrado guardado en %s", DATA_INTEGRADA_PATH)
    logging.info("Reporte ETL guardado en %s", ETL_REPORT_PATH)

    return data


def obtener_k_optimo(data_escalada):
    """Determina el número óptimo de clusters mediante el método del codo.

    Parameters
    ----------
    data_escalada : numpy.ndarray
        Datos ya escalados sobre los que evaluar la inercia.

    Returns
    -------
    tuple
        ``(k_optimo, valores_k, inercias)`` donde ``k_optimo`` es el número
        óptimo de clusters (con fallback a 4 si no se detecta codo).
    """
    valores_k = list(range(2, 9))
    inercias = []

    for k in valores_k:
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        modelo.fit(data_escalada)
        inercias.append(modelo.inertia_)

    knee = KneeLocator(
        valores_k,
        inercias,
        curve="convex",
        direction="decreasing"
    )

    if knee.elbow is None:
        logging.warning("No se detectó codo. Se usará k=4 como valor base.")
        k_optimo = 4
    else:
        k_optimo = int(knee.elbow)

    return k_optimo, valores_k, inercias


def describir_segmentos(data):
    """Genera el perfil promedio de cada segmento y lo persiste en CSV.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset con la columna ``cluster`` asignada.

    Returns
    -------
    pandas.DataFrame
        Resumen descriptivo por segmento.
    """
    perfil = data.groupby("cluster").agg(
        usuarios=("id_cliente", "count"),
        consumo_promedio=("horas_consumo_mensual", "mean"),
        gasto_promedio=("gasto_mensual", "mean"),
        satisfaccion_promedio=("satisfaccion_usuario", "mean"),
        reclamos_promedio=("reclamos_ultimos_3_meses", "mean"),
        antiguedad_promedio=("antiguedad_cliente_meses", "mean"),
        edad_promedio=("edad", "mean"),
        uso_app_movil=("porcentaje_uso_app_movil", "mean")
    ).round(2)

    perfil.to_csv("data/transformados/perfil_segmentos.csv")

    return perfil


def entrenar():
    """Ejecuta el pipeline completo: ETL, selección de k, entrenamiento y persistencia.

    Integra las fuentes, escala los datos, selecciona el ``k`` óptimo,
    entrena KMeans, aplica PCA para visualización, genera el perfil de
    segmentos y persiste modelo, scaler, PCA y métricas.
    """
    data = integrar_fuentes()

    X = data[COLUMNAS_MODELO]

    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X)

    k_optimo, valores_k, inercias = obtener_k_optimo(X_escalada)

    modelo = KMeans(
        n_clusters=k_optimo,
        random_state=42,
        n_init=10
    )

    clusters = modelo.fit_predict(X_escalada)
    silhouette = silhouette_score(X_escalada, clusters)

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_escalada)

    data["cluster"] = clusters
    data["pc1"] = componentes[:, 0]
    data["pc2"] = componentes[:, 1]

    centroides_originales = scaler.inverse_transform(modelo.cluster_centers_)
    centroides = pd.DataFrame(centroides_originales, columns=COLUMNAS_MODELO)
    centroides["cluster"] = range(k_optimo)

    metricas = {
        "silhouette_score": float(silhouette),
        "n_clusters": int(k_optimo),
        "n_usuarios": int(len(data)),
        "varianza_pc1": float(pca.explained_variance_ratio_[0]),
        "varianza_pc2": float(pca.explained_variance_ratio_[1]),
        "varianza_total_pca": float(pca.explained_variance_ratio_.sum()),
        "valores_k": valores_k,
        "inercias": [float(valor) for valor in inercias],
        "fuentes_integradas": 3
    }

    data.to_csv(DATA_SEGMENTADA_PATH, index=False)
    centroides.to_csv(CENTROIDES_PATH, index=False)
    describir_segmentos(data)

    with open(MODEL_PATH, "wb") as archivo:
        pickle.dump(modelo, archivo)

    with open(SCALER_PATH, "wb") as archivo:
        pickle.dump(scaler, archivo)

    with open(PCA_PATH, "wb") as archivo:
        pickle.dump(pca, archivo)

    with open(METRICAS_PATH, "wb") as archivo:
        pickle.dump(metricas, archivo)

    logging.info("Entrenamiento finalizado correctamente.")
    logging.info("Fuentes integradas: 3")
    logging.info("K óptimo: %s", k_optimo)
    logging.info("Silhouette Score: %.3f", silhouette)
    logging.info("Varianza PCA total: %.3f", metricas["varianza_total_pca"])


if __name__ == "__main__":
    entrenar()
