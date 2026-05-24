
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import torch
import cv2
import numpy as np
from PIL import Image
from model import MultiTaskFaceNet
from torchvision import transforms

# --- 1. ПАРАҚША БАПТАУЛАРЫ ---
st.set_page_config(page_title="AI Face Analyzer", layout="wide")

# Камераның тұрақтылығы үшін STUN сервері (MacBook M2 үшін маңызды)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# --- 2. МОДЕЛЬДІ ЖҮКТЕУ ---
@st.cache_resource
def load_model():
    # MacBook M2-де жылдам жұмыс істеу үшін MPS қолданамыз
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiTaskFaceNet().to(device)

    model_path = "best_face_model_m1.pth"

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Модельді жүктеу кезінде қате шықты: {e}")
        return None, device


model, device = load_model()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 3. КАДРДЫ ӨҢДЕУ ФУНКЦИЯСЫ ---
def process_frame(img_array):
    draw_img = img_array.copy()
    gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        roi = draw_img[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)

        input_tensor = tf(roi_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            age_p, gen_p = model(input_tensor)

            age = int(age_p.item() * 100)

            gender_idx = torch.argmax(gen_p, dim=1).item()
            gender = "Male" if gender_idx == 1 else "Female"

        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), color, 3)
        label = f"{gender}, {age}y"
        cv2.putText(draw_img, label, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return draw_img


# --- 4. STREAMLIT ИНТЕРФЕЙСІ ---

st.title("🤖 Нейрондық желілер негізінде бейнелерді өңдеу")

st.info("""
Бұл жүйе нейрондық желілер негізінде бейнелерді өңдеу арқылы адамның жасын және жынысын автоматты түрде анықтауға арналған. 
Жүйе нақты уақыт режимінде бейнекамера арқылы немесе жүктелген бейне арқылы жұмыс істейді. 
Бір уақытта бірнеше адамды анықтай алады.
""")

st.write(f"Жұмыс істеп тұрған құрылғы: **{device}**")


# --- ЖҮЙЕ МҮМКІНДІКТЕРІ ---
st.subheader("🔍 Жүйенің мүмкіндіктері")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    • Нақты уақыт режимінде анықтау  
    • Бірнеше адамды анықтау  
    • Адамның жасын анықтау  
    """)

with col2:
    st.markdown("""
    • Адамның жынысын анықтау  
    • Бейне арқылы талдау  
    • Автоматты нейрондық талдау  
    """)


# --- ПАЙДАЛАНУ НҰСҚАУЛЫҒЫ ---
st.subheader("📌 Пайдалану нұсқаулығы")

st.markdown("""
① Режимді таңдаңыз  
② Бейнекамера немесе бейне файлын таңдаңыз  
③ Бастау батырмасын басыңыз  
④ Жүйе автоматты түрде талдау жүргізеді  
⑤ Нәтиже экранда көрсетіледі  
""")


# --- Режим таңдау ---
mode = st.sidebar.selectbox(
    "Режимді таңдаңыз:",
    ["нақты уақыт режимі (бейнекамера арқылы)", "бейне файлын жүктеу"]
)

if mode == "нақты уақыт режимі (бейнекамера арқылы)":
    st.subheader("🎥 Тікелей бейне")

    st.caption("Бейнекамера арқылы нақты уақыт режимінде адамның жасын және жынысын анықтау")

    webrtc_streamer(
        key="face-analyzer",
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=lambda frame: av.VideoFrame.from_ndarray(
            process_frame(frame.to_ndarray(format="bgr24")), format="bgr24"
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    st.subheader("📁 Бейне файлын жүктеу")

    st.caption("Жүктелген бейне арқылы адамның жасын және жынысын анықтау")

    uploaded_file = st.file_uploader("Бейнені таңдаңыз...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with st.spinner('Нейрожелі бейнені өңдеуде...'):
            result_img = process_frame(image)

        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Талдау нәтижесі")
