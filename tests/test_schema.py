"""Pruebas automatizadas de esquema y calidad de las fuentes de datos.

Cubre las tres fuentes integradas por el pipeline ETL:
- CSV de comportamiento de consumo (``data/raw/usuarios_streaming.csv``)
- PostgreSQL de perfil (``database/perfil_usuarios.csv``)
- API REST comercial (``api-source/datos_api_usuarios.csv``)
"""

import pandas as pd


COLUMNAS_CSV = [
    "id_cliente",
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
]

COLUMNAS_PERFIL = [
    "id_cliente",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
]

COLUMNAS_API = [
    "id_cliente",
    "plan_streaming",
    "satisfaccion_usuario",
    "reclamos_ultimos_3_meses",
    "dispositivo_principal",
]


def _verificar_esquema_y_calidad(data, columnas, nombre_fuente):
    """Verifica que la fuente tenga las columnas y calidad mínima esperadas.

    Parameters
    ----------
    data : pandas.DataFrame
        Datos de la fuente a validar.
    columnas : list[str]
        Columnas obligatorias.
    nombre_fuente : str
        Nombre de la fuente para mensajes de aserción.
    """
    assert not data.empty, f"La fuente {nombre_fuente} está vacía."

    for columna in columnas:
        assert columna in data.columns, (
            f"La fuente {nombre_fuente} no contiene la columna {columna}."
        )

    assert not data["id_cliente"].isna().any(), (
        f"La fuente {nombre_fuente} tiene nulos en id_cliente."
    )
    assert not data["id_cliente"].duplicated().any(), (
        f"La fuente {nombre_fuente} tiene id_cliente duplicados."
    )


def test_usuarios_streaming_schema():
    """Valida esquema y calidad de la fuente 1 (CSV de comportamiento)."""
    data = pd.read_csv("data/originales/usuarios_streaming.csv")
    _verificar_esquema_y_calidad(data, COLUMNAS_CSV, "usuarios_streaming.csv")


def test_perfil_usuarios_schema():
    """Valida esquema y calidad de la fuente 2 (perfil de PostgreSQL)."""
    data = pd.read_csv("database/perfil_usuarios.csv")
    _verificar_esquema_y_calidad(data, COLUMNAS_PERFIL, "perfil_usuarios.csv")


def test_api_usuarios_schema():
    """Valida esquema y calidad de la fuente 3 (API REST comercial)."""
    data = pd.read_csv("api-source/datos_api_usuarios.csv")
    _verificar_esquema_y_calidad(data, COLUMNAS_API, "datos_api_usuarios.csv")
