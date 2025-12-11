import requests
from pathlib import Path

url = "http://localhost:8000/api/v1/predict/lesion"

# Đường dẫn file MHA
mha_path = Path(
    r"C:/Users/dxduc/Downloads/luna25_images_01_1/1.2.840.113654.2.55.112678153129052512356539493326784217907.mha"
)

def main():
    if not mha_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {mha_path}")

    # 1) PHẦN FILE: key phải là "image_file"
    files = {
        "image_file": (
            "scan.mha",               # tên gửi lên (có thể dùng mha_path.name)
            open(mha_path, "rb"),     # stream nhị phân
            "application/octet-stream"
        )
    }

    # 2) PHẦN CÁC FIELD FORM (trùng TÊN với FastAPI)
    data = {
        "PatientID": "122427",
        "SeriesInstanceUID": "1.2.840.113654.2.55.112678153129052512356539493326784217907",
        "StudyDate": "19990102",

        "CoordX": 107.99,
        "CoordY": -123.84,
        "CoordZ":  -261.55,

        "LesionID": 1,
        "AnnotationID": "1",
        "NoduleID": 1,
        "label": 1,

        "Age_at_StudyDate": 55,
        "Gender": "F",
    }

    print(f">>> Đang gửi request tới: {url}")
    resp = requests.post(url, files=files, data=data)

    print("Status Code:", resp.status_code)
    print("Headers:", resp.headers)

    try:
        print("JSON response:")
        print(resp.json())
    except Exception:
        print("Raw response:")
        print(resp.text)


if __name__ == "__main__":
    main()
