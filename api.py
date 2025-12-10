import os
import shutil
import json
from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import inference
from read_result import phan_tich_ket_qua

app = FastAPI(title="Lung Nodule Malignancy Prediction API")

# CHÚ Ý: thư mục templates nằm cùng cấp với file main.py
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # request ở đây là fastapi.Request, không liên quan gì tới thư viện requests
    return templates.TemplateResponse("index.html", {"request": request})


def build_inputs_from_tabular_row(row: Dict) -> Tuple[Dict, Dict]:
    # y chang như đã gửi: chuyển row -> clinical_data + locations_data
    raw_gender = str(row.get("Gender", "")).upper()
    if raw_gender in ["F", "FEMALE"]:
        gender = "FEMALE"
    elif raw_gender in ["M", "MALE"]:
        gender = "MALE"
    else:
        gender = "UNKNOWN"

    try:
        age = int(row.get("Age_at_StudyDate"))
    except (TypeError, ValueError):
        age = None

    clinical_data = {
        "gender": gender,
        "age": age,
        "smoking_status": "Unknown",
        "clinical_category": None,
        "regional_nodes_category": None,
        "metastasis_category": None,
    }

    patient_id = str(row.get("PatientID", ""))
    study_date = str(row.get("StudyDate", ""))

    lesion_id = row.get("LesionID", 1)
    annotation_id = row.get("AnnotationID")

    if annotation_id:
        name = str(annotation_id)
    else:
        name = f"{patient_id}_{lesion_id}_{study_date}"

    try:
        coord_x = float(row["CoordX"])
        coord_y = float(row["CoordY"])
        coord_z = float(row["CoordZ"])
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Thiếu trường toạ độ trong dữ liệu: {e}",
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Giá trị CoordX/CoordY/CoordZ không phải số.",
        )

    point = [coord_x, coord_y, coord_z]

    locations_data = {
        "name": "Points of interest",
        "type": "Multiple points",
        "points": [
            {"name": name, "point": point}
        ],
        "version": {"major": 1, "minor": 0},
    }

    return clinical_data, locations_data


@app.post("/predict")
async def predict_malignancy(
    image_file: UploadFile = File(..., description="File ảnh CT phổi (.mha / .mhd)"),

    PatientID: str = Form(...),
    SeriesInstanceUID: str = Form(...),
    StudyDate: str = Form(...),

    CoordX: float = Form(...),
    CoordY: float = Form(...),
    CoordZ: float = Form(...),

    LesionID: int = Form(...),
    AnnotationID: str = Form(None),
    NoduleID: int = Form(None),
    label: int = Form(None),

    Age_at_StudyDate: int = Form(...),
    Gender: str = Form(...)
):
    if not image_file.filename.endswith((".mha", ".mhd")):
        raise HTTPException(
            status_code=400,
            detail="Chỉ chấp nhận file định dạng .mha hoặc .mhd",
        )

    os.makedirs("uploads/input", exist_ok=True)
    img_path = "uploads/input/" + image_file.filename

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)

    try:
        patient_row: Dict = {
            "PatientID": PatientID,
            "SeriesInstanceUID": SeriesInstanceUID,
            "StudyDate": StudyDate,
            "CoordX": CoordX,
            "CoordY": CoordY,
            "CoordZ": CoordZ,
            "LesionID": LesionID,
            "AnnotationID": AnnotationID,
            "NoduleID": NoduleID,
            "label": label,
            "Age_at_StudyDate": Age_at_StudyDate,
            "Gender": Gender,
        }

        clinical_data, locations_data = build_inputs_from_tabular_row(patient_row)

        data = inference.run(
            locations_data,
            clinical_data,
            Path(img_path),
            patient_row.get('LesionID'),
            patient_row.get('SeriesInstanceUID'),
            mode="3D",
            model_name="videomae",
        )

        return {
                "status": "success",
                "data": data
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý nội bộ: {str(e)}",
        )
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    import uvicorn
    # Chạy server tại localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)