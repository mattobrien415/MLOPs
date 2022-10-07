from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests

app = FastAPI(title='Serverless Lambda FastAPI')

@app.get("/", )
async def root():
    response = requests.get('http:/face-bokeh-container:8001/')
    return response
