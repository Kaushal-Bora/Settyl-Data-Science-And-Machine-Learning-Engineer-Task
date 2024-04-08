from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import uvicorn
from pydantic import BaseModel
import numpy as np
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
model = load_model("bestWeights.keras")
le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True, fix_imports=True)
with open("max_length.npy", "rb") as f:
    max_length = np.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


class ExternalStatus(BaseModel):
    externalStatus: str


app = FastAPI()


@app.post("/predict")
def predict(external_status: ExternalStatus):
    X = [external_status.externalStatus]
    print(f"Input external status: {X}")

    X_seq = tokenizer.texts_to_sequences(X)
    print(f"Tokenized input: {X_seq}")

    X_pad = pad_sequences(X_seq, maxlen=max_length, padding="post")
    print(f"Padded input: {X_pad}")

    X_pad = np.expand_dims(X_pad, axis=-1)
    print(f"Final input shape: {X_pad.shape}")

    y_pred = model.predict(X_pad)
    print(f"Model prediction: {y_pred}")

    predicted_label = le.inverse_transform([np.argmax(y_pred[0])])[0]
    print(f"Predicted internal status: {predicted_label}")

    return {"internalStatus": predicted_label}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="My FastAPI App",
        version="1.0.0",
        description="API for predicting internal status",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
######added print statements for readability and debugging##########
