import json

# Giả sử đây là dữ liệu trả về từ API (hoặc đọc từ file)
# json_data = """
# {
#     "name": "Points of interest",
#     "type": "Multiple points",
#     "points": [
#         {
#             "name": "123456_1_19990102",
#             "point": [
#                 -64.78,
#                 64.5,
#                 -52.46
#             ],
#             "probability": 0.8920893669128418
#         },
#         {
#             "name": "123456_2_19990102",
#             "point": [
#                 -78.49,
#                 79.86,
#                 -152.83
#             ],
#             "probability": 0.8920893669128418
#         }
#     ],
#     "version": {
#         "major": 1,
#         "minor": 0
#     }
# }
# """

def phan_tich_ket_qua(data_input):
    # Nếu data_input là chuỗi string, cần parse sang Dict
    if isinstance(data_input, str):
        data = json.loads(data_input)
    else:
        data = data_input

    print(f"--- KẾT QUẢ PHÂN TÍCH ({len(data['points'])} nốt phổi) ---")
    print(f"{'ID NỐT':<25} | {'TỌA ĐỘ (X, Y, Z)':<25} | {'TỈ LỆ UNG THƯ':<15} | {'ĐÁNH GIÁ'}")
    print("-" * 85)

    for nodule in data['points']:
        # 1. Lấy ID
        name = nodule['name']
        
        # 2. Lấy tọa độ và làm tròn cho đẹp
        coords = nodule['point']
        coords_str = f"[{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]"
        
        # 3. Lấy xác suất và chuyển sang phần trăm
        prob = nodule['probability']
        prob_percent = f"{prob * 100:.2f}%"
        
        # 4. Đánh giá dựa trên ngưỡng (ví dụ > 0.5 là nguy cơ cao)
        status = "NGUY CƠ CAO 🔴" if prob > 0.5 else "LÀNH TÍNH 🟢"

        # In ra dòng kết quả
        print(f"{name:<25} | {coords_str:<25} | {prob_percent:<15} | {status}")

# --- CHẠY THỬ ---
# if __name__ == "__main__":
#     # Trường hợp 1: Đọc từ biến string ở trên
#     phan_tich_ket_qua(json_data)

    # Trường hợp 2: Nếu bạn muốn đọc từ file 'ket_qua.json'
    # with open('ket_qua.json', 'r') as f:
    #     data_from_file = json.load(f)
    #     phan_tich_ket_qua(data_from_file)