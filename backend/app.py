from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded file
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the video
    input_path = f"uploads/{file.filename}"
    output_path = f"processed/{file.filename}"
    process_video(input_path, output_path)

    return {"message": "Processing complete", "output_video": output_path}
