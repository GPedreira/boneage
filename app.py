import streamlit as st
import gdown
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd # Adicionado para manipulação de dados
import io
import pydicom
import os
import matplotlib.pyplot as plt # Para o histograma

# ============================================================================
# 1. DEFINIÇÃO DA ARQUITETURA DO MODELO
# ============================================================================

class BoneAgeModel(nn.Module):
    """
    Modelo baseado em EfficientNet com:
    - Attention mechanism
    - Gender como feature adicional
    - Regressão para idade óssea
    """
    
    def __init__(self, backbone='efficientnet_b3', pretrained=False, use_gender=True):
        super(BoneAgeModel, self).__init__()
        
        self.use_gender = use_gender
        
        if backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights=None)
            n_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone} não suportado")
        
        self.attention = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        input_size = n_features + (1 if use_gender else 0)
        
        self.regressor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout importante para o MC Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout importante para o MC Dropout
            nn.Linear(256, 1)
        )
        
    def forward(self, x, gender=None):
        features = self.backbone(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        features = features * attention_weights
        
        if self.use_gender and gender is not None:
            gender_unsqueezed = gender.unsqueeze(1) if gender.dim() == 1 else gender
            features = torch.cat([features, gender_unsqueezed], dim=1)
        
        bone_age = self.regressor(features)
        
        return bone_age.squeeze(-1)

# ============================================================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO
# ============================================================================

def get_valid_transforms(img_size=500):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# ============================================================================
# 3. LÓGICA DE MC DROPOUT E CARREGAMENTO
# ============================================================================

@st.cache_resource
def carrega_modelo(drive_url, model_path="modelo_idade_ossea.pth"):
    try:
        if not os.path.exists(model_path):
            with st.spinner(f"Baixando modelo... Isso pode demorar um pouco."):
                gdown.download(drive_url, model_path, quiet=False)
        
        model = BoneAgeModel(pretrained=False, use_gender=True)
        # map_location='cpu' é vital para deploy gratuito (Streamlit Cloud)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model
    
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def processa_imagem(uploaded_file, gender_float, img_size=500):
    image_data = uploaded_file.read()
    try:
        if uploaded_file.name.lower().endswith('.dcm'):
            dcm = pydicom.dcmread(io.BytesIO(image_data))
            image_array = dcm.pixel_array
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            display_image = image_array 
        else:
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_array = np.array(image_pil)
            display_image = image_pil 

    except Exception as e:
        st.error(f"Erro ao ler o arquivo de imagem: {e}")
        return None, None, None

    transform = get_valid_transforms(img_size)
    augmented = transform(image=image_array)
    image_tensor = augmented['image'].unsqueeze(0) 
    gender_tensor = torch.tensor([gender_float], dtype=torch.float32)

    return image_tensor, gender_tensor, display_image

def enable_dropout(m):
    """Função auxiliar para ativar o Dropout durante a inferência (eval)"""
    if type(m) == nn.Dropout:
        m.train()

def predicao_mc_dropout(model, image_tensor, gender_tensor, n_samples=30):
    """
    Executa a inferência N vezes com Dropout ativado para estimar incerteza.
    """
    model.eval() # Coloca Batch Norm e outros em modo de avaliação
    model.apply(enable_dropout) # Força APENAS o Dropout a ficar em modo de treino
    
    predictions = []
    
    # Barra de progresso visual
    progress_bar = st.progress(0)
    
    with torch.no_grad():
        for i in range(n_samples):
            pred = model(image_tensor, gender_tensor).cpu().item()
            predictions.append(pred)
            progress_bar.progress((i + 1) / n_samples)
            
    progress_bar.empty() # Remove a barra ao terminar
    
    return np.array(predictions)

# ============================================================================
# 4. APLICAÇÃO STREAMLIT
# ============================================================================

def main():
    st.set_page_config(page_title="Preditor de Idade Óssea + Incerteza", page_icon="🦴")
    
    st.title("🦴 Idade Óssea com Incerteza (MC Dropout)")
    st.markdown("""
    Esta aplicação utiliza **Monte Carlo Dropout** para estimar não apenas a idade, 
    mas a **confiança** do modelo. O modelo realiza múltiplas previsões variando os neurônios ativos.
    """)

    # URL DO MODELO
    DRIVE_FILE_ID = "1_vmkoI_Z9VVvgO6LY2V0KCDBtxJJCbCq" 
    drive_url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'

    model = carrega_modelo(drive_url)

    if model:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader('Upload Raio-X (PNG, JPG, DICOM)', type=['png', 'jpg', 'jpeg', 'dcm'])
        with col2:
            gender_str = st.radio("Gênero:", ('Masculino', 'Feminino'), horizontal=True)
            
            # Configuração de amostras para MC Dropout
            n_samples = st.slider("Amostras MC (Simulações):", min_value=10, max_value=100, value=30, step=10, 
                                  help="Quanto maior, mais precisa a estimativa de incerteza, mas mais lento.")

        if uploaded_file is not None:
            if st.button('Calcular com Incerteza'):
                
                gender_float = 1.0 if gender_str == 'Masculino' else 0.0
                image_tensor, gender_tensor, display_image = processa_imagem(uploaded_file, gender_float)
                
                if image_tensor is not None:
                    col_img, col_result = st.columns([1, 1.5])
                    
                    with col_img:
                        st.image(display_image, caption="Raio-X", use_column_width=True)
                    
                    with col_result:
                        st.write("### Analisando...")
                        
                        # --- PREDIÇÃO COM MC DROPOUT ---
                        try:
                            # Recebe lista de N predições
                            preds = predicao_mc_dropout(model, image_tensor, gender_tensor, n_samples=n_samples)
                            
                            # Estatísticas
                            media_meses = np.mean(preds)
                            std_meses = np.std(preds) # Desvio Padrão (Incerteza)
                            
                            media_anos = media_meses / 12.0
                            std_anos = std_meses / 12.0
                            
                            # --- EXIBIÇÃO DOS RESULTADOS ---
                            
                            st.success("Análise Finalizada!")
                            
                            # Métrica Principal
                            st.metric(
                                label="Idade Óssea Estimada (Média)", 
                                value=f"{media_meses:.1f} meses",
                                delta=f"± {std_meses:.1f} meses (Incerteza)",
                                delta_color="inverse" # Cinza se for neutro, ou mude conforme preferir
                            )
                            
                            st.write(f"**Em anos:** {media_anos:.1f} anos (± {std_anos:.1f} anos)")
                            
                            # Intervalo de Confiança (95% - aprox 2 desvios padrão)
                            lower = media_meses - (2 * std_meses)
                            upper = media_meses + (2 * std_meses)
                            st.info(f"**Intervalo de Confiança (95%):** Entre {lower:.1f} e {upper:.1f} meses")
                            
                            # --- GRÁFICO DE DISTRIBUIÇÃO ---
                            st.write("---")
                            st.write("#### Distribuição das Predições (Histograma)")
                            
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.hist(preds, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
                            ax.axvline(media_meses, color='red', linestyle='dashed', linewidth=1, label=f'Média: {media_meses:.1f}')
                            ax.set_xlabel('Idade (Meses)')
                            ax.set_ylabel('Frequência')
                            ax.set_title(f'Variação nas {n_samples} simulações')
                            ax.legend()
                            ax.grid(axis='y', alpha=0.5)
                            
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Erro na inferência: {e}")

if __name__ == "__main__":
    main()
