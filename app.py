import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from gtts import gTTS

@st.cache_resource
def carregar_modelo():
    return load_model("model/keras_model.h5", compile=False)
model = carregar_modelo()

with open("model/labels.txt", "r", encoding="utf-8") as f:
    class_names = [linha.strip().split(" ", 1)[1] for linha in f.readlines()]

dados_residuos = {
    "Plastico": {
        "emoji": "🟥",
        "lixeira": "Vermelha",
        "imagem": "assets/plastico.png",
        "tempo": "até 450 anos",
        "orientacao": "Descarte plásticos limpos na lixeira vermelha."
    },
    "Papel": {
        "emoji": "🟦",
        "lixeira": "Azul",
        "imagem": "assets/papel.png",
        "tempo": "3 a 6 meses",
        "orientacao": "Descarte papéis secos e limpos na lixeira azul."
    },
    "Metal": {
        "emoji": "🟨",
        "lixeira": "Amarela",
        "imagem": "assets/metal.png",
        "tempo": "100 a 500 anos",
        "orientacao": "Descarte metais na lixeira amarela."
    },
    "Vidro": {
        "emoji": "🟩",
        "lixeira": "Verde",
        "imagem": "assets/vidro.png",
        "tempo": "mais de 4.000 anos",
        "orientacao": "Descarte vidros na lixeira verde, com cuidado se estiverem quebrados."
    }
}
def classificar_imagem(imagem):
    imagem = imagem.convert("RGB")
    imagem = imagem.resize((224, 224))

    img_array = np.asarray(imagem)

    normalized_image = (img_array.astype(np.float32)/ 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    prediction = model.predict(data, verbose=0)
    indice = np.argmax(prediction)
    confianca = float(prediction[0][indice])
    return class_names[indice], confianca


st.set_page_config(
    page_title="EcoScan",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ EcoScan")
st.subheader("Identificação Inteligente de Resíduos")
st.write(
        """
    Envie uma imagem para identificar 
    automaticamente materiais recicláveis e descobrir 
    o descarte correto.
"""
    )

st.markdown("---")
st.info(
        "Ajude o meio ambiente identificando resíduos recicláveis."
    )


#foto = st.camera_input("Tirar Foto")

arquivo = st.file_uploader(
    "Envie uma foto para analisar",
    type=["jpeg", "png", "jpg"]
)

imagem_selecionada = arquivo

#if foto is not None:
#   imagem_selecionada = foto
#elif arquivo is not None:
#   imagem_selecionada = arquivo

if imagem_selecionada is not None:

    imagem = Image.open(imagem_selecionada)
    st.image(
        imagem,
        caption="Imagem selecionada",
        use_container_width=True
    )

    classe, confianca = classificar_imagem(imagem)
    st.markdown("Resultado da análise")
    info = dados_residuos[classe]

    st.markdown("## Resultado da análise")

    st.success(f"{info['emoji']} Material identificado: {classe}")
    st.write(f"**Confiança:** {confianca * 100:.2f}%")
    st.write(f"**Lixeira correta:** {info['lixeira']}")
    st.write(f"**Tempo estimado de decomposição:** {info['tempo']}")
    st.image(info["imagem"], caption=f"Lixeira {info['lixeira']}", 
    width=180)
    st.info(info["orientacao"])
    texto_audio = (
        f"Material identificado como {classe},"
        f"Utilize a lixeira {info['lixeira']} para o descarte correto."
    )
    tts = gTTS(text=texto_audio, lang="pt")
    tts.save("audio.mp3")
    audio_file = open("audio.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")
else:
    st.info("Envie uma imagem")
