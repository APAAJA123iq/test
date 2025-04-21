# from ultralytics import YOLO
# from ultralytics.yolo.v11.detect.predict import DetectionPredictor
# import cv2

# model=YOLO("yolov11onnxs.pt")

# results= model.predict(source="0", show=True)

# print(results)

# import cv2
# import torch
# from ultralytics import YOLO

# # Load model YOLOv11
# # Ganti dengan path model Anda
# model= YOLO('yolo11s.pt')
# model = YOLO("C:/Users/User/Videos/Workspace Python/best (9).pt")
# #model = YOLO("C:/Users/User/Videos/Workspace Python/best (3).pt")
# #model = YOLO("C:/Users/User/Videos/Workspace Python/best (5).pt")

# # Inisialisasi webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 untuk webcam utama

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Jalankan deteksi objek
#     results = model(frame)

#     # Gambar bounding box pada frame
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0].item()
#             cls = int(box.cls[0].item())
#             label = f"{model.names[cls]} {conf:.2f}"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Tampilkan hasil
#     cv2.imshow("YOLOv11 Webcam", frame)

#     # Tekan 'q' untuk keluar
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import torch
# import time  # Untuk menghitung FPS
# from ultralytics import YOLO

# # =============================
# # 1. Load model YOLO
# # =============================
# model = YOLO("C:/Users/User/Videos/Workspace Python/best (9).pt")

# # =============================
# # 2. Inisialisasi kamera
# # =============================
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# # =============================
# # 3. Mulai loop deteksi
# # =============================
# while cap.isOpened():
#     start_time = time.time()  # Catat waktu awal frame

#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Deteksi objek
#     results = model(frame)

#     # Gambar bounding box dan label
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0].item()
#             cls = int(box.cls[0].item())
#             label = f"{model.names[cls]} {conf:.2f}"

#             # Gambar kotak dan label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # =============================
#     # 4. Hitung dan tampilkan FPS
#     # =============================
#     end_time = time.time()  # Catat waktu akhir frame
#     fps = 1 / (end_time - start_time)  # Hitung FPS

#     # Tampilkan teks FPS di pojok kiri atas
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     # Warna biru, ukuran font 1, tebal 2

#     # =============================
#     # 5. Tampilkan frame
#     # =============================
#     cv2.imshow("YOLOv11 Webcam", frame)

#     # Tekan 'q' untuk keluar dari loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # =============================
# # 6. Bersihkan resource
# # =============================
# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import torch
# import time
# from ultralytics import YOLO

# # =============================
# # 1. Load model YOLO
# # =============================
# model = YOLO("C:/Users/User/Videos/Workspace Python/best321.pt")


# # =============================
# # 2. Inisialisasi kamera
# # =============================
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # =============================
# # 3. Inisialisasi penghitung
# # =============================
# count_normal = 0  # Kelas 0 = Botol Normal
# count_rusak = 0   # Kelas 1 = Botol Rusak

# # =============================
# # 4. Loop deteksi
# # =============================
# while cap.isOpened():
#     start_time = time.time()  # Waktu awal untuk FPS

#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Deteksi objek
#     results = model(frame)

#     # Gambar bounding box dan hitung jumlah deteksi
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0].item()
#             cls = int(box.cls[0].item())
#             label = f"{model.names[cls]} {conf:.2f}"

#             # Hitung deteksi botol
#             if cls == 0:
#                 count_normal += 1
#             elif cls == 1:
#                 count_rusak += 1

#             # Gambar kotak dan label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # =============================
#     # 5. Hitung dan tampilkan FPS
#     # =============================
#     end_time = time.time()
#     fps = 1 / (end_time - start_time)

#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # =============================
#     # 6. Tampilkan jumlah deteksi
#     # =============================
#     cv2.putText(frame, f"Normal: {count_normal}", (10, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.putText(frame, f"Rusak: {count_rusak}", (10, 110),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # =============================
#     # 7. Tampilkan frame
#     # =============================
#     cv2.imshow("YOLOv11 Webcam", frame)

#     # Tekan 'q' untuk keluar dari loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # =============================
# # 8. Bersihkan resource
# # =============================
# cap.release()
# cv2.destroyAllWindows()

# # =============================
# # 9. Simpan hasil
# # =============================

# # Simpan hasil ke file teks
# with open("hasil_deteksi.txt", "w") as f:
#     f.write(f"Jumlah botol normal: {count_normal}\n")
#     f.write(f"Jumlah botol rusak : {count_rusak}\n")

# print("Hasil deteksi disimpan ke 'hasil_deteksi.txt'")

# with cuda

import cv2
import torch
import time
from ultralytics import YOLO

# =============================
# 1. Load model YOLO
# =============================
model = YOLO("C:/Users/User/Videos/Workspace Python/best321.pt")

# Pindahkan model ke GPU jika tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model berjalan di: {device.upper()}")

# =============================
# 2. Inisialisasi kamera
# =============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# =============================
# 3. Inisialisasi penghitung
# =============================
count_normal = 0  # Kelas 0 = Botol Normal
count_rusak = 0   # Kelas 1 = Botol Rusak

# =============================
# 4. Loop deteksi
# =============================
while cap.isOpened():
    start_time = time.time()  # Waktu awal untuk FPS

    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek (opsional: kirim frame ke GPU jika model di GPU)
    results = model(frame)

    # Gambar bounding box dan hitung jumlah deteksi
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"

            # Hitung deteksi botol
            if cls == 0:
                count_normal += 1
            elif cls == 1:
                count_rusak += 1

            # Gambar kotak dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # =============================
    # 5. Hitung dan tampilkan FPS
    # =============================
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # =============================
    # 6. Tampilkan jumlah deteksi
    # =============================
    cv2.putText(frame, f"Normal: {count_normal}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Rusak: {count_rusak}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # =============================
    # 7. Tampilkan frame
    # =============================
    cv2.imshow("YOLOv11 Webcam", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# 8. Bersihkan resource
# =============================
cap.release()
cv2.destroyAllWindows()

# =============================
# 9. Simpan hasil
# =============================

# Simpan hasil ke file teks
with open("hasil_deteksi.txt", "w") as f:
    f.write(f"Jumlah botol normal: {count_normal}\n")
    f.write(f"Jumlah botol rusak : {count_rusak}\n")

print("Hasil deteksi disimpan ke 'hasil_deteksi.txt'")





