from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="API Mail Classifier",
    description="API for classifying emails using a pre-trained model.",
    version="1.0.0"
)

class EmailData(BaseModel):
    subject: str = Field(..., example="Important Meeting Tomorrow")
@app.post("/predict")
def predict(getData: EmailData):
    
    model01C = joblib.load( 'models/modelRandomForest.pkl') # Carga del modelo.
    vectorizer = joblib.load( 'models/vectorizer.pkl') # Carga del vectorizador.
    X = vectorizer.transform([getData.subject])
    predRandom = model01C.predict(X)
    return {"label": str(predRandom.tolist()[0])}


#uvicorn app:app --reload
#python -m uvicorn app:app --reload --host 0.0.0.0 --port 8002 --app-dir "C:\Users\Juan Pablo\inspecciones_en_campo\src\data\models"
#uvicorn app:app --reload --host 0.0.0.0 --port 8000
#pm2 start "sudo $(which python) -m uvicorn app:app --host 0.0.0.0 --port 8006 --ssl-keyfile /etc/letsencrypt/live/visor.inn.com.co/privkey.pem --ssl-certfile /etc/letsencrypt/live/visor.inn.com.co/fullchain.pem" --name api-inspecciones-campo
#uvicorn app:app --reload --app-dir src/data/models
#uvicorn app:app --reload --host 0.0.0.0 --port 8001 --app-dir src/data/models
#uvicorn app:app --reload --host 0.0.0.0 --app-dir src/data/models
#uvicorn app:app --reload --host 192.168.10.151 --port 8000 --app-dir src/data/models