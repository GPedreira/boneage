import streamlit as st
import gdown
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import io
import pydicom  # Para ler arquivos DICOM
import os

# ============================================================================
# 1. DEFINIÇÃO DA ARQUITETURA DO MODELO
# (Copiado do nosso script de treino - Necessário para recarregar o modelo)
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
        
        # Backbone (usamos pretrained=False pois carregaremos nossos próprios pesos)
        if backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights=None) # 'weights=None' é o novo padrão
            n_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone} não suportado")
        
        # Attention module
        self.attention = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        # Regressão final
        input_size = n_features + (1 if use_gender else 0)
        
        self.regressor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, gender=None):
        features = self.backbone(x)
        attention_weights = torch.softmax(self.attention(features), dim=1)
        features = features * attention_weights
        
        if self.use_gender and gender is not None:
            # Garante que 'gender' tenha as dimensões corretas [B, 1]
            gender_unsqueezed = gender.unsqueeze(1) if gender.dim() == 1 else gender
            features = torch.cat([features, gender_unsqueezed], dim=1)
        
        bone_age = self.regressor(features)
        
        # Correção do bug do batch_size=1
        return bone_age.squeeze(-1)

# ============================================================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO
# (Copiado do nosso script de treino)
# ============================================================================

def get_valid_transforms(img_size=500):
    """Transformações de validação (sem augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# ============================================================================
# 3. FUNÇÕES DO STREAMLIT
# ============================================================================

@st.cache_resource
def carrega_modelo(drive_url, model_path="modelo_idade_ossea.pth"):
    """
    Baixa o modelo .pth do Google Drive e o carrega no PyTorch.
    """
    try:
        # Verifica se o arquivo já existe para não baixar de novo
        if not os.path.exists(model_path):
            with st.spinner(f"Baixando modelo de {drive_url}... Isso pode demorar um pouco."):
                gdown.download(drive_url, model_path, quiet=False)
        
        # Instanciar a arquitetura do modelo
        model = BoneAgeModel(pretrained=False, use_gender=True)
        
        # Carregar os pesos salvos (state_dict)
        # Importante: map_location='cpu' garante que funcione no Streamlit (que não usa GPU)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Colocar o modelo em modo de avaliação (desliga dropout, etc.)
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.error("Verifique se o link do Google Drive está correto e com permissão 'Qualquer pessoa com o link'.")
        return None

def processa_imagem(uploaded_file, gender_float, img_size=500):
    """
    Lê um arquivo (PNG, JPG, DICOM), aplica transformações e retorna tensores.
    """
    image_data = uploaded_file.read()
    
    # 1. Ler a imagem (DICOM ou Padrão)
    try:
        if uploaded_file.name.lower().endswith('.dcm'):
            # É DICOM
            dcm = pydicom.dcmread(io.BytesIO(image_data))
            image_array = dcm.pixel_array
            
            # Converter para 3 canais (RGB) se for monocromático
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # Normalizar pixels DICOM (que não são 0-255) para 0-255 (uint8)
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            display_image = image_array # Para mostrar na tela
            
        else:
            # É JPG ou PNG
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_array = np.array(image_pil)
            display_image = image_pil # Para mostrar na tela

    except Exception as e:
        st.error(f"Erro ao ler o arquivo de imagem: {e}")
        return None, None, None

    # 2. Aplicar transformações do Albumentations
    transform = get_valid_transforms(img_size)
    augmented = transform(image=image_array)
    image_tensor = augmented['image']
    
    # 3. Preparar tensores para o modelo
    # Adicionar dimensão de batch (B, C, H, W)
    image_tensor = image_tensor.unsqueeze(0) 
    
    # Criar tensor para o gênero
    gender_tensor = torch.tensor([gender_float], dtype=torch.float32)

    return image_tensor, gender_tensor, display_image


def main():
    st.set_page_config(page_title="Preditor de Idade Óssea", page_icon="🦴")
    st.title("Calculadora de Idade Óssea 🦴")
    st.write("Faça o upload de um Raio-X da mão (PNG, JPG ou DICOM) e selecione o gênero para estimar a idade óssea.")

    # ========================================================================
    # TODO: SUBSTITUA A URL ABAIXO PELA SUA
    # 1. Faça upload do seu melhor modelo (.pth) no Google Drive
    # 2. Clique em "Compartilhar" -> "Qualquer pessoa com o link"
    # 3. Copie o link. Se o link for:
    #    https://drive.google.com/file/d/1bbgv3F0KR8xUIpMthWDhHvh0KAbuqFGd/view?usp=sharing
    #    Pegue apenas o 'ID_DO_ARQUIVO' e cole na URL abaixo.
    # ========================================================================
    
    # URL do modelo salvo (exemplo, use o seu ID real)
    # Este é o ID do 'test_bone_age_model.pth' que deu MAE 8.34
    DRIVE_FILE_ID = "1bbgv3F0KR8xUIpMthWDhHvh0KAbuqFGd" 
    drive_url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'

    model = carrega_modelo(drive_url)

    if model:
        # --- UI de Inputs ---
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                'Selecione a imagem do Raio-X', 
                type=['png', 'jpg', 'jpeg', 'dcm']
            )
        with col2:
            gender_str = st.radio(
                "Selecione o Gênero do Paciente:", 
                ('Masculino', 'Feminino'), 
                horizontal=True
            )

        if uploaded_file is not None:
            # Botão para iniciar a predição
            if st.button('Calcular Idade Óssea'):
                
                # Converter gênero para o formato float que o modelo espera
                gender_float = 1.0 if gender_str == 'Masculino' else 0.0
                
                # Processar a imagem e o gênero
                image_tensor, gender_tensor, display_image = processa_imagem(uploaded_file, gender_float)
                
                if image_tensor is not None:
                    # Mostrar a imagem que foi carregada
                    st.image(display_image, caption="Imagem Carregada", use_column_width=True)
                    
                    with st.spinner('Analisando a imagem...'):
                        try:
                            # --- PREDIÇÃO ---
                            with torch.no_grad(): # Desativa o cálculo de gradiente
                                pred_meses = model(image_tensor, gender_tensor).cpu().item()
                            
                            pred_anos = pred_meses / 12.0
                            
                            st.success("Análise concluída!")
                            
                            # --- Mostrar Resultado ---
                            st.metric(
                                label="Idade Óssea Predita", 
                                value=f"{pred_meses:.1f} meses",
                                delta=f"~ {pred_anos:.1f} anos",
                                delta_color="off" # Apenas informativo
                            )
                            
                        except Exception as e:
                            st.error(f"Erro durante a predição: {e}")

if __name__ == "__main__":
    main()
