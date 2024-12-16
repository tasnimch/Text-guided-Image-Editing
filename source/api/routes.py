from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io
from source.service.image_editor import ImageEditorService
from configuration.config import Config

app = FastAPI()


image_editor_service = ImageEditorService()

@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "Welcome to the Image Editor. Use the /edit-image/ endpoint to upload an image and an instruction."})

@app.post("/edit-image/")
async def edit_image(instruction: str = Form(...), file: UploadFile = File(None), url: str = Form(None)):
    try:
        image = await image_editor_service.process_image(instruction, file, url)        
        return FileResponse(image)
    except ValueError as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"message": "Cannot identify image file"}, status_code=500)
