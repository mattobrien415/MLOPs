from fastapi import FastAPI, File, UploadFile
import aiofiles

import detect



app = FastAPI()
#The root path will be used as the health check endpoint
#@app.get("/")
#async def root():
#    return {"health_check": "ok"}

from fastapi.responses import FileResponse

some_file_path = "im.jpeg"
app = FastAPI()


@app.post("/")
async def post_endpoint(in_file: UploadFile=File(...)):
    # ...
    async with aiofiles.open('inference/images/horses.jpg', 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write

    return {"Result": "OK"}

@app.get("/")
async def mainx():
    print(type(detect.detect))
    detect.detect()
    return FileResponse('runs/detect/horses.jpg')
