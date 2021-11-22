from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from typing import List
from PIL import ImageColor
from eval_image import main
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/snapshot")
async def create_upload_files(color: str = Form(default = "#FFFFFF"), style: str = Form(default = "style1"), files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "./"
    print(color)
    print(style)
    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename + ".jpg"), "wb") as fp:
            fp.write(contents)
        print(file.filename+ ".jpg")
        main(file.filename+ ".jpg", style, color)

    return "output_img.jpg"