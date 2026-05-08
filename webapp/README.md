# Waveform ML Studio

เว็บสำหรับงาน waveform regression ที่ใช้ **TCN encoder เป็น feature extractor** แล้วส่ง embedding เข้า **AutoGluon Tabular** เพื่อ train/predict พร้อม dashboard และ waveform analysis gallery

## Features
- Drag & drop CSV
- Preview ตารางข้อมูล
- Train model จากหน้าเว็บ
- แสดง metrics เช่น MAE / RMSE / fast precision / fast recall
- สร้าง plots อัตโนมัติ
- Predict ไฟล์ใหม่จากหน้าเว็บ
- สร้างโฟลเดอร์ `backend/data/analysis/<job_id>/...` เก็บรูป waveform prediction ราย sample
- เปิดดูรูป analysis จากหน้าเว็บได้เลย

## Expected CSV shape
อย่างน้อยควรมีคอลัมน์ลักษณะนี้:
- `wave_id`
- `wait_time_ms`
- `wave_0, wave_1, ..., wave_999`
- meta columns อื่น ๆ ใส่เพิ่มได้

## Run backend
```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Backend จะรันที่ `http://localhost:5000`

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend จะรันที่ `http://localhost:5173`

## Key folders
- `backend/data/uploads/` ไฟล์ CSV ที่อัปโหลด
- `backend/data/models/<job_id>/` model artifacts
- `backend/data/results/<job_id>/` csv และ summary
- `backend/data/plots/<job_id>/` loss curve / scatter / histogram
- `backend/data/analysis/<job_id>/validation/` waveform images ของ validation set
- `backend/data/analysis/<job_id>/predict/` waveform images หลัง predict

## Notes
- starter นี้รองรับ CSV เป็นหลักก่อน
- ถ้า dataset ใหญ่มาก ควรเปลี่ยนจาก thread เป็น job queue เช่น Celery / RQ
- ถ้า waveform ไม่ได้ใช้ prefix `wave_` ให้เปลี่ยนจากหน้าเว็บได้
