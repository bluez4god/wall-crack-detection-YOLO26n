from pathlib import Path
from io import BytesIO
from hashlib import sha256
from threading import Lock
import time

import streamlit as st
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title='Wall Crack Detection', page_icon='🧱', layout='centered')

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'best.pt'

@st.cache_resource
def load_model(path, modified):
    torch.set_num_threads(2)
    return YOLO(path), Lock()


def analyze(photo_bytes, threshold, model, model_lock):
    with Image.open(BytesIO(photo_bytes)) as source:
        if source.width * source.height > 24_000_000:
            raise ValueError('Resolusi terlalu besar. Gunakan foto maksimal 24 megapiksel.')
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((3000, 3000))
    start = time.perf_counter()
    with model_lock:
        result = model.predict(image, conf=threshold, imgsz=640, device='cpu', verbose=False)[0]
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        annotated = Image.fromarray(result.plot()[:, :, ::-1])
    elapsed = time.perf_counter() - start
    buffer = BytesIO()
    annotated.save(buffer, format='PNG')
    return image, annotated, scores, elapsed, buffer.getvalue()


st.title('Wall Crack Detection')
st.caption('YOLO26 • Ambil foto dan deteksi retakan secara otomatis')
st.write('Arahkan kamera ke dinding, ambil foto, dan lihat hasil analisis.')

with st.sidebar:
    st.header('Pengaturan')
    confidence = st.slider('Confidence threshold', 0.05, 0.95, 0.25, 0.05)
    st.caption('Ambang rendah dapat membantu mendeteksi retakan kecil, tetapi juga meningkatkan deteksi keliru.')
    st.write('Model: YOLO26n | CPU | 640 px')

if not MODEL_PATH.is_file():
    st.error('Model tidak ditemukan. Pastikan models/best.pt tersedia di repository.')
    st.stop()

mode = st.radio('Sumber gambar', ['📷 Ambil Foto', '📁 Upload Foto'], horizontal=True)

if mode == '📷 Ambil Foto':
    st.subheader('Kamera')
    st.caption('Izinkan kamera. Di HP, gunakan ikon ganti kamera jika tersedia untuk memilih kamera belakang.')
    photo = st.camera_input('Foto permukaan dinding', resolution='720p', key='wall_camera')
    st.caption('Tekan Take Photo untuk menganalisis. Gunakan Clear photo untuk mengambil foto baru. Jika kamera tidak tersedia, gunakan Upload Foto.')
else:
    st.subheader('Upload Foto')
    photo = st.file_uploader('Pilih foto dinding', type=['jpg', 'jpeg', 'png', 'webp'], key='wall_upload')

if photo is None:
    st.session_state.pop('wall_result', None)
    st.info('Ambil atau pilih foto untuk memulai deteksi.')
    st.stop()

photo_bytes = photo.getvalue()
if len(photo_bytes) > 20 * 1024 * 1024:
    st.error('Ukuran foto terlalu besar. Gunakan foto maksimal 20 MB.')
    st.stop()

# Simpan hanya hasil foto terakhir dalam sesi pengguna, bukan cache bersama.
result_key = (mode, sha256(photo_bytes).hexdigest(), confidence, MODEL_PATH.stat().st_mtime)
saved = st.session_state.get('wall_result')
if saved is None or saved['key'] != result_key:
    st.session_state.pop('wall_result', None)
    try:
        with st.spinner('Menganalisis foto dinding...'):
            model, model_lock = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            result = analyze(photo_bytes, confidence, model, model_lock)
        saved = {'key': result_key, 'data': result}
        st.session_state['wall_result'] = saved
    except Exception as exc:
        st.error(f'Analisis gagal: {exc}')
        st.stop()

original, annotated, scores, elapsed, png_bytes = saved['data']

st.divider()
st.subheader('Hasil Deteksi')
if scores:
    st.success('Retakan terdeteksi oleh model.')
else:
    st.info('Tidak ada retakan terdeteksi pada ambang confidence ini. Ini bukan bukti bahwa dinding bebas retak.')

st.image(annotated, caption='Hasil analisis YOLO26n', width='stretch')
a, b = st.columns(2)
a.metric('Jumlah bounding box', len(scores))
b.metric('Confidence tertinggi', f'{max(scores):.1%}' if scores else '—')
st.caption(f'Waktu inferensi dan anotasi (termasuk antrean): {elapsed:.2f} detik. Tidak termasuk waktu pengiriman foto atau pemuatan awal model.')

if scores:
    with st.expander('Detail deteksi'):
        st.dataframe([{'Deteksi': i + 1, 'Kelas': 'crack', 'Confidence': f'{score:.1%}'} for i, score in enumerate(scores)], hide_index=True)

with st.expander('Lihat foto asli'):
    st.image(original, width='stretch')

st.download_button('⬇ Unduh hasil deteksi', data=png_bytes, file_name='wall_crack_result.png', mime='image/png')
st.caption('Foto dikirim ke server untuk analisis. Aplikasi ini hanya alat bantu visual, bukan penilaian keamanan bangunan.')
