import shutil
import tempfile
import json
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from typing import Dict
import inference
from read_result import phan_tich_ket_qua

# --- CẤU HÌNH API FASTAPI ---

app = FastAPI(title="Lung Nodule Malignancy Prediction API")

@app.get("/")
async def root():
    return {"message": "Chào mừng đến với API dự đoán ung thư nốt phổi!"}

@app.post("/predict")
async def predict_malignancy(
    image_file: UploadFile = File(..., description="File ảnh CT phổi (.mha)"),
    nodule_locations: UploadFile = File(..., description="File JSON chứa tọa độ nodule"),
    clinical_info: UploadFile = File(..., description="File JSON chứa thông tin lâm sàng (Tuổi, Giới tính)")
):
    """
    API nhận vào file ảnh .mha và tọa độ, trả về xác suất ung thư.
    """
    
    # 1. Validate định dạng file ảnh
    if not image_file.filename.endswith(('.mha', '.mhd')):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file định dạng .mha hoặc .mhd")

    # 2. Tạo thư mục tạm để lưu file ảnh (SimpleITK thường cần đường dẫn file vật lý)
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp_img:
    #     shutil.copyfileobj(image_file.file, tmp_img)
    #     tmp_img_path = tmp_img.name
    img_path = "uploads/input/" + image_file.filename
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    

    try:
        # 3. Đọc nội dung JSON
        try:
            locations_content = await nodule_locations.read()
            locations_data = json.loads(locations_content)
            
            clinical_content = await clinical_info.read()
            clinical_data = json.loads(clinical_content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="File locations hoặc clinical info không phải là JSON hợp lệ.")

        # 4. Gọi hàm dự đoán từ module inference
        results = inference.run(
            locations_data,
            clinical_data,
            Path(img_path),
            mode="3D",
            model_name="finetune-hiera"
        )
        
        # 5. Đọc kết quả từ file kết quả
        with open('results/results.json', 'r') as f:
            data_from_file = json.load(f)
            phan_tich_ket_qua(data_from_file)
        return data_from_file

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý nội bộ: {str(e)}")
    
    finally:
        # 5. Dọn dẹp file tạm
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    import uvicorn
    # Chạy server tại localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)