import pandas as pd
from fastapi import FastAPI


app = FastAPI(
    title="API Fuente Externa - Usuarios Streaming",
    description="Fuente REST complementaria para la integración ETL.",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "mensaje": "API fuente externa funcionando"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok"
    }


@app.get("/usuarios-api")
def usuarios_api():
    data = pd.read_csv("datos_api_usuarios.csv")

    return {
        "usuarios_api": data.to_dict(orient="records")
    }
