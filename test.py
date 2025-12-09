import requests

url = "http://localhost:8000/predict"

files = [
    ('image_file', ('scan.mha', open(r'test/input/images/chest-ct/1.2.840.113654.2.55.294281779470566559919697495520361195429.mha', 'rb'), 'application/octet-stream')),
    ('nodule_locations', ('loc.json', open(r'test/input/nodule-locations.json', 'rb'), 'application/json')),
    ('clinical_info', ('clinic.json', open(r'test/input/clinical-information-lung-ct.json', 'rb'), 'application/json'))
]

response = requests.post(url, files=files)
# print(response.json())
print("Status Code:", response.status_code)
print("Nội dung trả về từ server:", response.text)