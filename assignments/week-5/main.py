from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests
from fastapi.encoders import jsonable_encoder

# Let's generate a new FastAPI app
# Generate a FastAPI instance called `app` with the title 'Triton Health Check'
# https://fastapi.tiangolo.com/


#Call your get function for a health Check
#to receive both (face-bokeh and face-emotion



app = FastAPI(title='Serverless Lambda FastAPI')

@app.get("/", )
async def root():
    response_bokeh = requests.get('http://face-bokeh-container:8000/')
    respose_emotion = requests.get('http://face-emotion:8000/')
    return [response_bokeh, respose_emotion]
