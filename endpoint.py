from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd

import importlib, sys
m = importlib.import_module("query_model_results")
print("Dir(query_model_results):", dir(m), file=sys.stderr)
from query_model_results import query_model_api

app = FastAPI(title="Open-Meteo Climate API", version="1.0.0")

class ClimateSummary(BaseModel):
    records: int
    columns: List[str]
    datetime_min: Optional[str]
    datetime_max: Optional[str]
    temperature_mean_c: Optional[float]

class ClimateResponse(BaseModel):
    summary: ClimateSummary
    sample: Optional[list] = None  # primeras N filas opcional

@app.get("/climate", response_model=ClimateResponse)
def get_climate(
    latitude: float = Query(52.52, description="Latitud, default Berlin"),
    longitude: float = Query(13.41, description="Longitud, default Berlin"),
    timeout: int = Query(30, ge=1, le=120, description="Timeout en segundos"),
    sample_rows: int = Query(0, ge=0, le=200, description="Cantidad de filas de muestra a retornar (0 = ninguna)")
):
    try:
        df: pd.DataFrame = get_climate_data_from_api(latitude=latitude, longitude=longitude, timeout=timeout)

        summary = ClimateSummary(
            records=len(df),
            columns=list(df.columns),
            datetime_min=df["datetime"].min().isoformat() if "datetime" in df.columns and len(df) else None,
            datetime_max=df["datetime"].max().isoformat() if "datetime" in df.columns and len(df) else None,
            temperature_mean_c=float(df["temperature_2m"].mean()) if "temperature_2m" in df.columns and len(df) else None,
        )

        sample = None
        if sample_rows > 0 and len(df) > 0:
            # Convertir a tipos serializables
            df_out = df.copy()
            if "datetime" in df_out.columns:
                df_out["datetime"] = df_out["datetime"].astype(str)
            sample = df_out.head(sample_rows).to_dict(orient="records")

        return ClimateResponse(summary=summary, sample=sample)

    except Exception as e:
        # Propaga un error HTTP 502 con el mensaje original
        raise HTTPException(status_code=502, detail=f"Error obteniendo datos: {str(e)}")