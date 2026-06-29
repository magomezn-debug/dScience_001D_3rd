# Segmentación de Usuarios de Streaming con Machine Learning

## Descripción del proyecto

Este proyecto implementa una solución completa de segmentación de clientes para una plataforma de streaming utilizando técnicas de aprendizaje no supervisado.

El objetivo es identificar grupos de usuarios con comportamientos similares a partir de su consumo, perfil y características comerciales, utilizando el algoritmo **KMeans** apoyado en reducción de dimensionalidad con **PCA**.

La solución integra:

- Una fuente de datos en formato **CSV** (comportamiento de consumo).
- Una segunda fuente de datos almacenada en **PostgreSQL** (perfil del usuario).
- Una tercera fuente expuesta mediante **API REST** (información comercial).
- Un **pipeline ETL** automatizado que integra y valida las tres fuentes.
- Entrenamiento de un modelo de **clustering** con selección óptima de `k`.
- Exposición del modelo mediante una **API REST** (FastAPI).
- **Dashboard** interactivo para análisis de resultados (Streamlit).
- Contenerización completa utilizando **Docker** y **Docker Compose**.

## Arquitectura de la solución

```text
   CSV Streaming          PostgreSQL          API REST
   (comportamiento)       (perfil)            (comercial)
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                      +----------------+
                      |  Integración   |
                      |   de datos ETL |
                      +----------------+
                              |
                              v
                      +----------------+
                      |     KMeans     |
                      | + StandardScaler|
                      | +     PCA       |
                      +----------------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
            FastAPI                    Streamlit
         Servicio ML                   Dashboard
```

## Tecnologías utilizadas

### Lenguaje
- Python 3.11

### Machine Learning
- Scikit-learn
- KMeans
- StandardScaler
- PCA
- Silhouette Score
- Método del codo mediante `KneeLocator`

### Datos
- Pandas
- PostgreSQL
- SQLAlchemy

### Backend
- FastAPI
- Uvicorn

### Visualización
- Streamlit
- Matplotlib

### Infraestructura
- Docker
- Docker Compose
- Variables de entorno para configuración externa

## Estructura del proyecto

```text
segmentacion-usuarios-streaming/
│
├── docker-compose.yml
├── README.md
│
├── database/
│   ├── init.sql
│   └── perfil_usuarios.csv
│
├── data/
│   ├── originales/
│   │   └── usuarios_streaming.csv
│   └── transformados/
│       ├── data_usuarios.csv
│       ├── usuarios_segmentados.csv
│       ├── centroides.csv
│       ├── perfil_segmentos.csv
│       └── reporte_etl.csv
│
├── etl/
│   ├── README.md
│   └── EA3_notebook_apoyo.ipynb
│
├── ml-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py
│   ├── app.py
│   └── models/
│       ├── modelo_kmeans.pkl
│       ├── scaler.pkl
│       ├── pca.pkl
│       └── metricas.pkl
│
├── api-source/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── datos_api_usuarios.csv
│
├── dashboard/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── tests/
│   ├── README.md
│   ├── requirements.txt
│   └── test_schema.py
│
├── docs/
│   ├── api.md
│   ├── arquitectura.md
│   └── manual_usuario.md
│
└── repo/
    └── evidencia_git.md
```

## Fuentes de datos

El proyecto integra **tres fuentes de datos diferentes**, cumpliendo el requisito de la rúbrica.

### Fuente 1: CSV
Archivo: `data/originales/usuarios_streaming.csv`

Contiene información asociada al comportamiento de consumo de los usuarios.

Variables principales:
- horas de consumo mensual
- gasto mensual
- cantidad de contenidos vistos
- sesiones por semana
- porcentaje de finalización
- antigüedad del cliente

### Fuente 2: PostgreSQL
Base de datos: `streaming_clientes`
Tabla: `perfil_usuarios`

Contiene información del perfil del usuario:
- edad
- dispositivos registrados
- uso de app móvil
- perfiles creados
- interacciones con soporte

### Fuente 3: API REST
Endpoint: `GET /usuarios-api` (servicio `api-source`)

Contiene información comercial complementaria:
- plan de streaming
- satisfacción del usuario
- reclamos
- dispositivo principal

## Pipeline de Machine Learning

El proceso ejecutado por `ml-service/train.py` realiza:

1. Lectura del archivo CSV (fuente 1).
2. Conexión a PostgreSQL y extracción desde `perfil_usuarios` (fuente 2).
3. Consumo de la API REST `/usuarios-api` (fuente 3).
4. Validación de esquemas y calidad de cada fuente.
5. Integración de las tres fuentes mediante `id_cliente`.
6. Generación del dataset analítico `data_usuarios.csv`.
7. Normalización de variables con `StandardScaler`.
8. Evaluación del número óptimo de clusters con el método del codo (`KneeLocator`).
9. Entrenamiento del modelo `KMeans`.
10. Reducción de dimensionalidad con `PCA` para visualización.
11. Perfilamiento de cada segmento.
12. Persistencia del modelo, scaler, PCA y métricas.

### Serialización

El ML Service guarda en `ml-service/models/`:
- `modelo_kmeans.pkl`
- `scaler.pkl`
- `pca.pkl`
- `metricas.pkl`

## Ejecución del proyecto

### Requisitos
Tener instalado:
- Docker
- Docker Compose

### Levantar la solución
Desde la raíz del proyecto:

```bash
docker compose up --build
```

Esto levanta cuatro servicios:

| Servicio | Puerto | Descripción |
|---|---|---|
| PostgreSQL | 5432 | Base de datos `streaming_clientes` |
| API fuente externa | 7000 | API REST con datos comerciales |
| ML Service | 8000 | FastAPI con modelo KMeans |
| Dashboard | 8501 | Streamlit interactivo |

## Acceso a los servicios

### API Machine Learning
Abrir:
```text
http://localhost:8000
```
Respuesta esperada:
```json
{"mensaje": "Servicio ML funcionando"}
```

### API fuente externa
Abrir:
```text
http://localhost:7000
```

### Dashboard
Abrir:
```text
http://localhost:8501
```

El dashboard muestra primero las métricas principales:
- Silhouette Score
- número de clusters
- cantidad de usuarios
- varianza PCA
- fuentes integradas

Luego contiene vistas diferenciadas por audiencia:
- **Vista ejecutiva**: indicadores de negocio y tamaño de segmentos.
- **Vista técnica**: métricas del modelo, varianza PCA y método del codo.
- **Vista operativa**: perfil detallado de cada segmento.

## Endpoint de predicción

Servicio:
```text
POST /predict
```

Ejemplo de entrada:
```json
{
  "horas_consumo_mensual": 45,
  "gasto_mensual": 30,
  "cantidad_contenidos_vistos": 18,
  "sesiones_semana": 5,
  "porcentaje_finalizacion": 0.75,
  "tiempo_promedio_sesion_min": 50,
  "cantidad_generos_consumidos": 4,
  "porcentaje_uso_promociones": 0.2,
  "antiguedad_cliente_meses": 24,
  "edad": 35,
  "dispositivos_registrados": 3,
  "porcentaje_uso_app_movil": 0.4,
  "cantidad_perfiles_creados": 2,
  "interacciones_mensuales_soporte": 1,
  "distancia_promedio_red_km": 5,
  "satisfaccion_usuario": 4.2,
  "reclamos_ultimos_3_meses": 0,
  "plan_streaming_codigo": 3,
  "dispositivo_principal_codigo": 1
}
```

Respuesta:
```json
{"cluster": 2}
```

## Detener servicios

Para detener los contenedores:
```bash
docker compose down
```

Para eliminar también los volúmenes:
```bash
docker compose down -v
```

## Para cambios de código

Reconstruir sin caché y levantar:
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

Para verificar el contenido de la tabla en PostgreSQL:
```bash
docker exec -it streaming_database psql -U admin -d streaming_clientes
```
Luego, para ver las tablas:
```sql
\dt
```

## Objetivo del proyecto

Construir una solución analítica completa que permita transformar datos provenientes de múltiples fuentes en información accionable para la toma de decisiones mediante técnicas de Machine Learning no supervisado, con un pipeline ETL robusto, validación de esquemas, manejo avanzado de errores y despliegue reproducible en Docker.

La lectura se realiza con `rb`, es decir, read binary.

## Documentación

La documentación está en:

```text
docs/
```

Incluye:

- arquitectura
- documentación API
- manual de usuario

## Tests

Pruebas básicas en:

```text
tests/
```

## Git

La carpeta `repo/` contiene una guía de evidencias que deben completar con capturas o links reales del repositorio.
