"""
RecycleNet Streamlit app.

Fixed dark theme, responsive layout, desktop nav plus mobile hamburger menu,
class grouping, recycling info and About Us page.
"""

from __future__ import annotations

import io
import os
import uuid

import streamlit as st
from PIL import Image

from utils import CLASS_EMOJIS, CLASS_NAMES, get_device, load_model, predict


st.set_page_config(page_title="RecycleNet ♻️", page_icon="♻️", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
ALL_CLASSES = CLASS_NAMES

CLASS_LABELS = {
    "es": {
        "Glass": "Vidrio",
        "Paper": "Papel",
        "Cardboard": "Cartón",
        "Plastic": "Plástico",
        "Metal": "Metal",
        "Battery": "Pila",
        "Biological": "Orgánico",
        "Textile": "Textil",
    },
    "en": {
        "Glass": "Glass",
        "Paper": "Paper",
        "Cardboard": "Cardboard",
        "Plastic": "Plastic",
        "Metal": "Metal",
        "Battery": "Battery",
        "Biological": "Biological",
        "Textile": "Textile",
    },
    "pt": {
        "Glass": "Vidro",
        "Paper": "Papel",
        "Cardboard": "Papelão",
        "Plastic": "Plástico",
        "Metal": "Metal",
        "Battery": "Pilha",
        "Biological": "Orgânico",
        "Textile": "Têxtil",
    },
}

CLASS_META = {
    "Glass": {
        "icon": "🪟",
        "tip": {
            "es": "Botellas y frascos de vidrio. Enjuagalos antes. No incluyas vidrio roto ni espejos.",
            "en": "Glass bottles and jars. Rinse before recycling. Do not include broken glass or mirrors.",
            "pt": "Garrafas e potes de vidro. Enxágue antes. Não inclua vidro quebrado nem espelhos.",
        },
    },
    "Paper": {
        "icon": "📄",
        "tip": {
            "es": "Diarios, revistas, cuadernos y hojas limpias. Evitá papel con restos de comida o grasa.",
            "en": "Newspapers, magazines, notebooks and clean sheets. Avoid paper with food or grease residue.",
            "pt": "Jornais, revistas, cadernos e folhas limpas. Evite papel com restos de comida ou gordura.",
        },
    },
    "Cardboard": {
        "icon": "📦",
        "tip": {
            "es": "Cajas y packaging. Aplastá las cajas para reducir volumen y sacá cintas si podés.",
            "en": "Boxes and packaging. Flatten them to reduce volume and remove tape when possible.",
            "pt": "Caixas e embalagens. Achate para reduzir volume e remova fitas quando possível.",
        },
    },
    "Plastic": {
        "icon": "🧴",
        "tip": {
            "es": "Botellas PET y envases. Enjuagalos y aplastá las botellas para ahorrar espacio.",
            "en": "PET bottles and containers. Rinse them and flatten bottles to save space.",
            "pt": "Garrafas PET e embalagens. Enxágue e achate as garrafas para economizar espaço.",
        },
    },
    "Metal": {
        "icon": "🥫",
        "tip": {
            "es": "Latas de aluminio y acero. Aplastá las latas para ahorrar espacio y facilitar el transporte.",
            "en": "Aluminum and steel cans. Flatten cans to save space and make transport easier.",
            "pt": "Latas de alumínio e aço. Achate as latas para economizar espaço e facilitar o transporte.",
        },
    },
    # "Trash": {
    #     "icon": "🗑️",
    #     "tip": {
    #         "es": "Residuo no reciclable. Intentá reducirlo comprando productos con menos packaging.",
    #         "en": "Non-recyclable waste. Try to reduce it by buying products with less packaging.",
    #         "pt": "Resíduo não reciclável. Tente reduzi-lo comprando produtos com menos embalagem.",
    #     },
    # },
    "Battery": {
        "icon": "🔋",
        "tip": {
            "es": "Residuo peligroso. Llevá las pilas a puntos de recolección especiales.",
            "en": "Hazardous waste. Take batteries to special collection points.",
            "pt": "Resíduo perigoso. Leve as pilhas a pontos de coleta especiais.",
        },
    },
    "Biological": {
        "icon": "🍂",
        "tip": {
            "es": "Restos de comida, hojas y material orgánico. Se pueden compostar en casa.",
            "en": "Food scraps, leaves and organic material. Can be composted at home.",
            "pt": "Restos de comida, folhas e material orgânico. Podem ser compostados em casa.",
        },
    },
    "Textile": {
        "icon": "👕",
        "tip": {
            "es": "Ropa en buen estado: doná. Dañada: contenedor de reciclaje textil.",
            "en": "Good condition clothing: donate. Damaged: textile recycling container.",
            "pt": "Roupa em bom estado: doe. Danificada: contêiner de reciclagem têxtil.",
        },
    },
}

TRANSLATIONS = {
    "es": {
        "app_title": "RecycleNet",
        "app_subtitle": "",
        "nav_classify": "Clasificar",
        "nav_recycling": "Reciclaje",
        "nav_about": "About us",
        "nav_config": "Configurar clases",
        "lang_label": "Idioma",
        "classify_title": "Clasificar residuos",
        "classify_subtitle": "Subí una foto para identificar el tipo de residuo y ver cómo se recicla.",
        "tab_upload": "📂 Subir imagen",
        "tab_camera": "📷 Cámara",
        "upload_label": "Seleccioná una imagen (JPG, PNG, WEBP)",
        "cam_label": "O tomá una foto con la cámara",
        "loading": "Analizando imagen...",
        "no_model": "⚠️ No se encontró model.pth en prod/. Copiá el modelo entrenado para activar predicciones.",
        "result_label": "Clase detectada",
        "result_conf": "Confianza",
        "result_dist": "Distribución de clases",
        "result_tip": "¿Cómo reciclarlo?",
        "group_detail": "Dentro del grupo",
        "btn_again": "🔄 Volver a clasificar",
        "btn_config": "⚙️ Configurar clases de reciclaje",
        "btn_back_config": "← Volver",
        "config_title": "Configurar clases de reciclaje",
        "config_subtitle": "Agrupá clases del modelo en una sola etiqueta. Las clases que no selecciones quedan separadas como categorías individuales.",
        "config_hint_title": "Configurar clases de reciclaje",
        "config_hint_body": "Ejemplo: Vidrio + Plástico → Reciclaje. Si una clase pertenece a un grupo, no se puede repetir en otro.",
        "config_hint_empty": "Todavía no configuraste grupos. Podés crear uno desde acá.",
        "config_hint_ready": "Tenés grupos activos. Podés editarlos o crear nuevos.",
        "config_add": "+ Agregar grupo",
        "config_save": "Guardar y volver",
        "config_reset": "Restablecer",
        "group_name": "Nombre del grupo",
        "group_classes": "Clases incluidas",
        "ungrouped": "Clases sin agrupar",
        "why_title": "Importancia del reciclaje",
        "how_title": "Cómo separar",
        "mendoza_title": "Reciclaje en Mendoza",
        "info_why": "El reciclaje reduce residuos en basurales, disminuye emisiones de CO₂ y ahorra recursos naturales.",
        "info_how": "Separá por material, enjuagá envases cuando haga falta, aplastá cajas y botellas para ahorrar espacio, mantené pilas y residuos peligrosos aparte.",
        "info_mendoza": "La ciudad de Mendoza promueve activamente el reciclaje a través de puntos verdes distribuidos en barrios y plazas.",
        "impact_1_num": "60%",
        "impact_1_desc": "de los residuos podría reciclarse",
        "impact_2_num": "95%",
        "impact_2_desc": "de energía se ahorra reciclando aluminio",
        "impact_3_num": "17",
        "impact_3_desc": "árboles por tonelada de papel reciclado",
        "about_title": "About Us",
        "about_intro": "RecycleNet es un proyecto universitario de Redes Neuronales Profundas de la UTN Facultad Regional Mendoza.",
        "about_project": "Nace como respuesta a una necesidad concreta de la ciudad: se promueve el reciclaje, pero a veces clasificar correctamente los residuos no es tan simple.",
        "about_recycling": "También incluye una explicación más completa sobre la importancia del reciclaje, cómo separar residuos y buenas prácticas para reducir el impacto ambiental.",
        "about_repo": "Repositorio del proyecto",
        "team_title": "Integrantes",
        "team_badge": "LinkedIn",
        "repo_label": "Abrir repo",
    },
    "en": {
        "app_title": "RecycleNet",
        "nav_classify": "Classify",
        "nav_recycling": "Recycling",
        "nav_about": "About us",
        "nav_config": "Configure classes",
        "lang_label": "Language",
        "classify_title": "Classify waste",
        "classify_subtitle": "Upload a photo to identify the waste type and learn how to recycle it.",
        "tab_upload": "📂 Upload image",
        "tab_camera": "📷 Camera",
        "upload_label": "Select an image (JPG, PNG, WEBP)",
        "cam_label": "Or take a photo with the camera",
        "loading": "Analyzing image...",
        "no_model": "⚠️ model.pth was not found in prod/. Copy the trained model to enable predictions.",
        "result_label": "Detected class",
        "result_conf": "Confidence",
        "result_dist": "Class distribution",
        "result_tip": "How should it be recycled?",
        "group_detail": "Inside the group",
        "btn_again": "🔄 Classify again",
        "btn_config": "⚙️ Configure recycling classes",
        "btn_back_config": "← Back",
        "config_title": "Configure recycling classes",
        "config_subtitle": "Group model classes under one label. Unselected classes remain as separate categories.",
        "config_hint_title": "Configure recycling classes",
        "config_hint_body": "Example: Glass + Plastic → Recyclable. If a class belongs to a group, it cannot be repeated in another one.",
        "config_hint_empty": "You have not configured any groups yet. You can create one here.",
        "config_hint_ready": "You already have active groups. You can edit them or create new ones.",
        "config_add": "+ Add group",
        "config_save": "Save and return",
        "config_reset": "Reset",
        "group_name": "Group name",
        "group_classes": "Included classes",
        "ungrouped": "Ungrouped classes",
        "why_title": "Why recycling matters",
        "how_title": "How to sort",
        "mendoza_title": "Recycling in Mendoza",
        "info_why": "Recycling reduces landfill waste, lowers CO₂ emissions and saves natural resources.",
        "info_how": "Sort by material, rinse containers when needed, flatten boxes and bottles to save space, keep batteries and hazardous waste separate.",
        "info_mendoza": "The city of Mendoza actively promotes recycling through green points across neighborhoods and parks.",
        "impact_1_num": "60%",
        "impact_1_desc": "of waste could be recycled",
        "impact_2_num": "95%",
        "impact_2_desc": "energy is saved by recycling aluminum",
        "impact_3_num": "17",
        "impact_3_desc": "trees per ton of recycled paper",
        "about_title": "About Us",
        "about_intro": "RecycleNet is a university deep neural networks project from UTN Facultad Regional Mendoza.",
        "about_project": "It comes from a real local need: recycling is encouraged in the city, but classifying waste correctly is not always easy.",
        "about_recycling": "It also includes a deeper explanation of why recycling matters, how to sort waste, and which habits reduce environmental impact.",
        "about_repo": "Project repository",
        "team_title": "Team",
        "team_badge": "LinkedIn",
        "repo_label": "Open repo",
    },
    "pt": {
        "app_title": "RecycleNet",
        "nav_classify": "Classificar",
        "nav_recycling": "Reciclagem",
        "nav_about": "About us",
        "nav_config": "Configurar classes",
        "lang_label": "Idioma",
        "classify_title": "Classificar resíduos",
        "classify_subtitle": "Envie uma foto para identificar o tipo de resíduo e ver como reciclar.",
        "tab_upload": "📂 Enviar imagem",
        "tab_camera": "📷 Câmera",
        "upload_label": "Selecione uma imagem (JPG, PNG, WEBP)",
        "cam_label": "Ou tire uma foto com a câmera",
        "loading": "Analisando imagem...",
        "no_model": "⚠️ model.pth não foi encontrado em prod/. Copie o modelo treinado para ativar as previsões.",
        "result_label": "Classe detectada",
        "result_conf": "Confiança",
        "result_dist": "Distribuição de classes",
        "result_tip": "Como reciclar?",
        "group_detail": "Dentro do grupo",
        "btn_again": "🔄 Classificar novamente",
        "btn_config": "⚙️ Configurar classes de reciclagem",
        "btn_back_config": "← Voltar",
        "config_title": "Configurar classes de reciclagem",
        "config_subtitle": "Agrupe classes do modelo em uma única etiqueta. As classes não selecionadas ficam separadas.",
        "config_hint_title": "Configurar classes de reciclagem",
        "config_hint_body": "Exemplo: Vidro + Plástico → Reciclável. Se uma classe pertence a um grupo, ela não pode se repetir em outro.",
        "config_hint_empty": "Você ainda não configurou grupos. Pode criar um aqui.",
        "config_hint_ready": "Você já tem grupos ativos. Pode editá-los ou criar novos.",
        "config_add": "+ Adicionar grupo",
        "config_save": "Salvar e voltar",
        "config_reset": "Redefinir",
        "group_name": "Nome do grupo",
        "group_classes": "Classes incluídas",
        "ungrouped": "Classes sem grupo",
        "why_title": "Importância da reciclagem",
        "how_title": "Como separar",
        "mendoza_title": "Reciclagem em Mendoza",
        "info_why": "A reciclagem reduz resíduos em aterros, diminui emissões de CO₂ e economiza recursos naturais.",
        "info_how": "Separe por material, enxágue embalagens quando necessário, achate caixas e garrafas para economizar espaço, mantenha pilhas e resíduos perigosos separados.",
        "info_mendoza": "A cidade de Mendoza promove ativamente a reciclagem por meio de pontos verdes distribuídos em bairros e praças.",
        "impact_1_num": "60%",
        "impact_1_desc": "dos resíduos poderiam ser reciclados",
        "impact_2_num": "95%",
        "impact_2_desc": "de energia é economizada ao reciclar alumínio",
        "impact_3_num": "17",
        "impact_3_desc": "árvores por tonelada de papel reciclado",
        "about_title": "About Us",
        "about_intro": "RecycleNet é um projeto universitário de Redes Neuronais Profundas da UTN Faculdade Regional Mendoza.",
        "about_project": "Ele surge de uma necessidade real da cidade: a reciclagem é incentivada, mas classificar corretamente os resíduos nem sempre é simples.",
        "about_recycling": "Este app também traz uma explicação mais completa sobre a importância da reciclagem, como separar resíduos e boas práticas para reduzir o impacto ambiental.",
        "about_repo": "Repositório do projeto",
        "team_title": "Equipe",
        "team_badge": "LinkedIn",
        "repo_label": "Abrir repositório",
    },
}

LANG_OPTIONS = ["es", "en", "pt"]


def ss(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def t(key: str) -> str:
    return TRANSLATIONS[st.session_state.lang].get(key, key)


def class_label(cls: str) -> str:
    return CLASS_LABELS[st.session_state.lang].get(cls, cls)


def new_group() -> dict[str, object]:
    return {"id": uuid.uuid4().hex, "name": "", "classes": []}


def normalize_groups(groups: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    used: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id") or uuid.uuid4().hex)
        name = str(group.get("name") or "").strip()
        selected: list[str] = []
        for cls in group.get("classes", []):
            if cls in ALL_CLASSES and cls not in used and cls not in selected:
                selected.append(cls)
                used.add(cls)
        normalized.append({"id": group_id, "name": name, "classes": selected})
    return normalized


def grouped_classes() -> list[str]:
    return [cls for group in st.session_state.groups for cls in group["classes"]]


def ungrouped_classes() -> list[str]:
    grouped = set(grouped_classes())
    return [cls for cls in ALL_CLASSES if cls not in grouped]


def resolve_display(predicted: str, probs: dict[str, float]) -> tuple[str, bool, list[str], float]:
    for group in st.session_state.groups:
        if predicted in group["classes"] and group["name"]:
            classes = list(group["classes"])
            confidence = sum(probs.get(cls, 0.0) for cls in classes)
            return str(group["name"]), True, classes, confidence
    return predicted, False, [], probs.get(predicted, 0.0)


def go(page: str, reset_result: bool = False) -> None:
    st.session_state.page = page
    if reset_result:
        st.session_state.result = None


def render_css() -> None:
    bg = "#090b0f"
    bg2 = "#121720"
    bg3 = "#1a202c"
    text = "#f8fbff"
    text2 = "#c4cfdd"
    text3 = "#9aa7b8"
    border = "rgba(255,255,255,0.12)"
    accent = "#54d6b2"
    accent2 = "#2ea7ff"
    accent_soft = "rgba(84,214,178,0.16)"
    gradient = "linear-gradient(135deg, rgba(84,214,178,0.34), rgba(46,167,255,0.16))"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background:
    radial-gradient(circle at top right, rgba(84,214,178,0.18), transparent 30%),
    radial-gradient(circle at left bottom, rgba(46,167,255,0.10), transparent 28%),
    {bg} !important;
  color: {text} !important;
  font-family: 'Manrope', sans-serif !important;
  color-scheme: dark;
}}

#MainMenu, header, footer {{ display: none !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}

.block-container {{
  padding-top: 1rem !important;
  padding-bottom: 1.8rem !important;
  max-width: none !important;
  width: 100% !important;
  padding-left: clamp(1rem, 2vw, 2rem) !important;
  padding-right: clamp(1rem, 2vw, 2rem) !important;
}}

h1, h2, h3, h4, h5, h6, p, div, span, label, button, input, textarea {{
  font-family: 'Manrope', sans-serif !important;
}}

.rn-brand {{
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  font-size: 1.08rem;
  font-weight: 900;
  color: {text};
  letter-spacing: -0.03em;
}}

.rn-brand span:first-child {{
  width: 2rem;
  height: 2rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: {gradient};
  box-shadow: 0 10px 30px rgba(22,160,133,0.22);
}}

.rn-hero,
.rn-card,
.rn-side,
.rn-badge,
.rn-info-card,
.rn-team-card,
.rn-impact-card,
.rn-config-group,
.rn-hero-panel {{
  background: {bg2};
  border: 1px solid {border};
  border-radius: 22px;
  box-shadow: 0 14px 36px rgba(0,0,0,0.12);
}}

.rn-hero {{
  padding: 1.1rem 1.2rem;
  margin: 0 0 1rem 0;
  background:
    linear-gradient(135deg, rgba(84,214,178,0.18), rgba(46,167,255,0.06)),
    {bg2};
}}

.rn-hero h1 {{
  margin: 0;
  font-size: clamp(1.75rem, 3vw, 2.75rem);
  font-weight: 900;
  letter-spacing: -0.05em;
  color: {text};
}}

.rn-hero p {{
  margin: 0.45rem 0 0;
  color: {text2};
  line-height: 1.55;
  font-size: 0.98rem;
}}

.rn-section-title {{
  margin: 0.9rem 0 0.35rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  color: {text};
}}

.rn-side {{
  position: sticky;
  top: 1rem;
  padding: 1rem;
}}

.rn-side-title {{
  font-weight: 900;
  font-size: 1rem;
  margin: 0 0 0.35rem;
  color: {text};
}}

.rn-side p {{
  margin: 0;
  color: {text2};
  line-height: 1.55;
  font-size: 0.95rem;
}}

.rn-chip-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.8rem;
}}

.rn-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.72rem;
  border-radius: 999px;
  background: {bg3};
  border: 1px solid {border};
  color: {text2};
  font-size: 0.82rem;
  font-weight: 700;
}}

.rn-chip.solid {{
  background: {accent};
  color: #fff;
  border-color: {accent};
}}

.rn-callout {{
  margin-top: 0.8rem;
  padding: 0.9rem 1rem;
  border-radius: 18px;
  background: {accent_soft};
  border: 1px solid rgba(84,214,178,0.24);
}}

.rn-callout strong {{ display:block; color:{text}; margin-bottom:0.35rem; }}
.rn-callout span {{ color:{text2}; line-height:1.45; }}

.rn-badge {{
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 1rem 1.1rem;
  margin-bottom: 0.9rem;
}}

.rn-badge-icon {{
  width: 3rem;
  height: 3rem;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: {gradient};
  font-size: 1.45rem;
  flex-shrink: 0;
}}

.rn-badge-label {{
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.74rem;
  font-weight: 900;
  color: {text3};
  margin-bottom: 0.18rem;
}}

.rn-badge-value {{
  font-size: clamp(1.25rem, 3vw, 1.85rem);
  font-weight: 900;
  color: {text};
  line-height: 1.15;
}}

.rn-badge-value small {{ color: {text2}; font-weight: 700; }}

.rn-prob-row {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
  font-size: 0.92rem;
}}

.rn-prob-cls {{
  width: 8.2rem;
  flex-shrink: 0;
  color: {text2};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.rn-prob-bar-wrap {{
  flex: 1;
  height: 7px;
  background: {bg3};
  border-radius: 999px;
  overflow: hidden;
}}

.rn-prob-bar {{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, {accent}, {accent2});
}}

.rn-prob-bar.muted {{ opacity: 0.45; }}

.rn-prob-pct {{
  width: 3.6rem;
  text-align: right;
  color: {text3};
  font-size: 0.84rem;
  flex-shrink: 0;
}}

.rn-tip {{
  background: {bg2};
  border: 1px solid {border};
  border-left: 4px solid {accent};
  border-radius: 18px;
  padding: 0.95rem 1rem;
  margin-top: 0.95rem;
  color: {text2};
  line-height: 1.6;
}}

.rn-tip-title {{
  font-weight: 900;
  color: {text};
  margin-bottom: 0.35rem;
}}

.rn-info-grid,
.rn-team-grid,
.rn-impact,
.rn-hero-grid {{
  display: grid;
  gap: 0.75rem;
}}

.rn-info-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.rn-team-grid {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
.rn-impact {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin: 0.95rem 0 1.1rem; }}
.rn-hero-grid {{ grid-template-columns: 1.45fr 0.95fr; margin-bottom: 0.95rem; }}

.rn-info-card,
.rn-team-card,
.rn-impact-card,
.rn-config-group,
.rn-hero-panel {{ padding: 1rem; }}

.rn-info-card h4,
.rn-team-card h4 {{
  margin: 0 0 0.45rem;
  color: {text};
  font-size: 1rem;
  font-weight: 900;
}}

.rn-info-card p,
.rn-team-card p,
.rn-hero-panel p {{
  margin: 0;
  color: {text2};
  line-height: 1.55;
  font-size: 0.94rem;
}}

.stButton > button,
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stRadio [role="radiogroup"],
.stFileUploader button {{
  color: {text} !important;
}}

.stButton > button,
div[data-testid="stButton"] > button {{
  background: {bg2};
  border: 1px solid {border};
  border-radius: 14px;
  box-shadow: 0 8px 18px rgba(0,0,0,0.10);
}}

.stButton > button:hover,
div[data-testid="stButton"] > button:hover {{
  border-color: {accent};
  color: {text};
  background: {bg3};
  box-shadow: 0 10px 22px rgba(0,0,0,0.16);
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stRadio"] [role="radiogroup"] {{
  background: {bg2} !important;
  border-color: {border} !important;
  color: {text} !important;
}}

div[data-testid="stSelectbox"] svg,
div[data-testid="stRadio"] svg {{ fill: {text2} !important; }}

div[data-testid="stPopover"] > button {{
  width: auto !important;
  min-width: 0 !important;
  padding-left: 0.8rem !important;
  padding-right: 0.8rem !important;
  margin-left: auto !important;
}}

div[data-testid="stMain"] [data-testid="stAlert"] {{
  background: #ffffff !important;
  border: 1px solid rgba(15,23,42,0.18) !important;
  color: #111111 !important;
}}

div[data-testid="stMain"] [data-testid="stAlert"] * {{
  color: #111111 !important;
}}

div[data-testid="stMain"] [data-testid="stAlert"] svg {{
  fill: #111111 !important;
}}

a,
a:visited {{
  color: {accent};
}}

a:hover {{
  color: {accent2};
}}

.rn-team-card {{ text-align: center; }}

.rn-avatar {{
  width: 3rem;
  height: 3rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  margin: 0 auto 0.7rem;
  background: {gradient};
  color: {text};
  font-weight: 900;
  letter-spacing: 0.03em;
}}

.rn-team-link,
.rn-repo-link {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 0.75rem;
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  text-decoration: none;
  border: 1px solid {accent};
  color: {accent};
  font-size: 0.82rem;
  font-weight: 900;
}}

.rn-repo-link {{
  background: {accent};
  color: #071018;
}}

.rn-impact-card {{ text-align: center; }}
.rn-impact-num {{ font-size: 1.75rem; font-weight: 900; color: {text}; letter-spacing: -0.03em; }}
.rn-impact-desc {{ margin-top: 0.25rem; color: {text2}; font-size: 0.87rem; line-height: 1.45; }}

.rn-config-grid {{ display: grid; gap: 0.75rem; }}

.rn-config-group h4 {{
  margin: 0 0 0.75rem;
  font-size: 1rem;
  color: {text};
}}

.rn-footer-space {{ height: 0.8rem; }}

@media (max-width: 980px) {{
  .rn-team-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .rn-hero-grid {{ grid-template-columns: 1fr; }}
}}

@media (max-width: 820px) {{
  .block-container {{
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    padding-top: 0.75rem !important;
  }}
  div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stSelectbox"] {{
    display: none !important;
  }}
  div[data-testid="stRadio"] {{ display: none !important; }}
  .rn-info-grid,
  .rn-team-grid,
  .rn-impact {{ grid-template-columns: 1fr; }}
  .rn-prob-cls {{ width: 7rem; }}
  .rn-side {{ position: static; }}
}}

@media (min-width: 821px) {{
  div[data-testid="stPopover"] {{ display: none !important; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_nav() -> None:
  left, middle, right = st.columns([1.5, 3.2, 0.9], gap="small", vertical_alignment="center")

  with left:
    st.markdown(
      f"<div class='rn-brand'><span>♻️</span><span>{t('app_title')}</span></div>",
      unsafe_allow_html=True,
    )
    st.caption(t("app_subtitle"))

  with middle:
    nav_choice = st.radio(
      "Navigation",
      options=[t("nav_classify"), t("nav_recycling"), t("nav_about"), t("nav_config")],
      horizontal=True,
      label_visibility="collapsed",
      index=["classify", "recycling", "about", "config"].index(st.session_state.page),
    )
    mapping = {
      t("nav_classify"): "classify",
      t("nav_recycling"): "recycling",
      t("nav_about"): "about",
      t("nav_config"): "config",
    }
    new_page = mapping[nav_choice]
    if new_page != st.session_state.page:
      st.session_state.page = new_page
      st.rerun()

  with right:
    new_lang = st.selectbox(
      t("lang_label"),
      options=LANG_OPTIONS,
      index=LANG_OPTIONS.index(st.session_state.lang),
      format_func=lambda value: {"es": "🇦🇷 Español", "en": "🇺🇸 English", "pt": "🇧🇷 Português"}[value],
      label_visibility="collapsed",
      key="lang_select_desktop",
    )
    if new_lang != st.session_state.lang:
      st.session_state.lang = new_lang
      st.rerun()

    with st.popover("☰"):
      mobile_lang = st.selectbox(
        t("lang_label"),
        options=LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.lang),
        format_func=lambda value: {"es": "🇦🇷 Español", "en": "🇺🇸 English", "pt": "🇧🇷 Português"}[value],
        label_visibility="visible",
        key="lang_select_mobile",
      )
      if mobile_lang != st.session_state.lang:
        st.session_state.lang = mobile_lang
        st.rerun()

      st.markdown("---")

      if st.button(t("nav_classify"), key="mobile_nav_classify", use_container_width=True):
        go("classify")
        st.rerun()
      if st.button(t("nav_recycling"), key="mobile_nav_recycling", use_container_width=True):
        go("recycling")
        st.rerun()
      if st.button(t("nav_about"), key="mobile_nav_about", use_container_width=True):
        go("about")
        st.rerun()
      if st.button(t("nav_config"), key="mobile_nav_config", use_container_width=True):
        go("config")
        st.rerun()


def render_team_cards() -> None:
    team = [
        ("Antonella Aldao", "AA", "https://www.linkedin.com/in/antonellaaldao/"),
        ("Ignacio Berridy", "IB", "https://www.linkedin.com/in/ignacioberridy/"),
        ("Carlos Gitto", "CG", "https://www.linkedin.com/in/carlosgitto/"),
        ("Diandra Malca", "DM", "https://www.linkedin.com/in/diandra-malca-8099791b3/"),
        ("Ignacio Ramos", "IR", "https://www.linkedin.com/in/ignacio-ramos-developer/"),
    ]

    cols = st.columns(len(team), gap="small")
    for col, (name, initials, url) in zip(cols, team):
        with col:
            st.markdown(
                f"""
                <div class='rn-team-card'>
                  <div class='rn-avatar'>{initials}</div>
                  <h4>{name}</h4>
                  <p>{t('team_badge')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"[LinkedIn ↗]({url})")


def render_recycling_cards() -> None:
    cols = st.columns(2, gap="small")
    for index, cls in enumerate(ALL_CLASSES):
        tip = CLASS_META[cls]["tip"][st.session_state.lang]
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class='rn-info-card'>
                  <h4>{CLASS_META[cls]['icon']} {class_label(cls)}</h4>
                  <p>{tip}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result_bars(probs: dict[str, float], limit_classes: list[str] | None = None) -> None:
    items = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    if limit_classes is not None:
        allowed = set(limit_classes)
        items = [item for item in items if item[0] in allowed]

    rows = []
    top_class = items[0][0] if items else None
    for cls, pct in items:
        muted = "" if cls == top_class else " muted"
        rows.append(
            f"""
            <div class='rn-prob-row'>
              <span class='rn-prob-cls'>{CLASS_EMOJIS.get(cls, '')} {class_label(cls)}</span>
              <div class='rn-prob-bar-wrap'><div class='rn-prob-bar{muted}' style='width:{pct:.2f}%'></div></div>
              <span class='rn-prob-pct'>{pct:.1f}%</span>
            </div>
            """
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


def render_classify_page() -> None:
    st.markdown(
        f"""
        <div class='rn-hero'>
          <h1>{t('classify_title')}</h1>
          <p>{t('classify_subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.65, 0.85], gap="large", vertical_alignment="top")

    # with right:
    #     st.markdown(
    #         f"""
    #         <div class='rn-side'>
    #           <div class='rn-side-title'>{t('config_hint_title')}</div>
    #           <p>{t('config_hint_body')}</p>
    #           <div class='rn-callout'>
    #             <strong>{t('config_hint_title')}</strong>
    #             <span>{t('config_hint_ready') if st.session_state.groups else t('config_hint_empty')}</span>
    #           </div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    #     st.button(t("btn_config"), use_container_width=True, key="open_config_side", on_click=go, args=("config", False))

    #     if st.session_state.groups:
    #         st.caption(f"{len(st.session_state.groups)} grupo(s) activos · {len(grouped_classes())} clase(s) agrupadas")
    #         st.markdown(
    #             "<div class='rn-chip-row'>"
    #             + "".join(
    #                 f"<span class='rn-chip'>{group['name'] or t('group_name')}</span>"
    #                 for group in st.session_state.groups
    #             )
    #             + "</div>",
    #             unsafe_allow_html=True,
    #         )

    # with right:
    #     st.markdown(
    #         f"""
    #         <div class='rn-side'>
    #         <div class='rn-side-title'>{t('config_hint_title')}</div>
    #         <p>{t('config_hint_body')}</p>
    #         <div class='rn-callout'>
    #             <strong>{t('config_hint_title')}</strong>
    #             <span>{t('config_hint_ready') if st.session_state.groups else t('config_hint_empty')}</span>
    #         </div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    #     st.button(
    #     t("btn_config"),
    #     use_container_width=True,
    #     key="open_config_side",
    #     on_click=go,
    #     args=("config", False),
    # )

    with right:
        st.markdown(
            f"""
            <div class='rn-side'>
                <div class='rn-side-title'>{t('config_hint_title')}</div>
                <p>{t('config_hint_body')}</p>
                <div class='rn-callout'>
                    <strong>{t('config_hint_title')}</strong>
                    <span>{t('config_hint_ready') if st.session_state.groups else t('config_hint_empty')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            t("btn_config"),
            use_container_width=True,
            key="open_config_side",
            on_click=go,
            args=("config", False),
        )

        # ==========================================
        # RESUMEN COMPACTO DE GRUPOS
        # ==========================================

        has_real_groups = any(len(g["classes"]) > 0 for g in st.session_state.groups)

        if has_real_groups or ungrouped_classes():

            st.markdown("##### 📦 Grupos configurados")

            # grupos
            for group in st.session_state.groups:

                if not group["classes"]:
                    continue

                group_name = group["name"].strip()

                if not group_name:
                    group_name = " + ".join(
                        class_label(cls)
                        for cls in group["classes"]
                    )

                classes_text = ", ".join(
                    class_label(cls)
                    for cls in group["classes"]
                )

                st.caption(f"📂 {group_name}: {classes_text}")

            # individuales
            for cls in ungrouped_classes():

                st.caption(
                    f"{CLASS_EMOJIS.get(cls,'')} {class_label(cls)}: Clase individual"
                )
    with left:
        if not MODEL_READY:
            st.warning(t("no_model"))

        if st.session_state.result is None:
            tab_upload, tab_camera = st.tabs([t("tab_upload"), t("tab_camera")])
            uploaded_file = None
            camera_file = None

            with tab_upload:
                uploaded_file = st.file_uploader(
                    t("upload_label"),
                    type=["jpg", "jpeg", "png", "webp"],
                    label_visibility="collapsed",
                )

            with tab_camera:
                camera_file = st.camera_input(t("cam_label"), label_visibility="collapsed")

            source = uploaded_file or camera_file
            if source is not None and MODEL_READY:
                with st.spinner(t("loading")):
                    image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")
                    model, device = get_model()
                    predicted, confidence, all_probs = predict(model, image, device)

                st.session_state.result = {
                    "predicted": predicted,
                    "confidence": confidence,
                    "probs": all_probs,
                    "image_bytes": source.getvalue(),
                }
                st.rerun()

        else:
            result = st.session_state.result
            predicted = result["predicted"]
            all_probs = result["probs"]
            image_bytes = result.get("image_bytes")
            meta = CLASS_META[predicted]
            display, is_group, group_classes, display_conf = resolve_display(predicted, all_probs)

            preview_col, info_col = st.columns([1, 1.25], gap="medium", vertical_alignment="top")
            with preview_col:
                if image_bytes:
                    st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True)

            with info_col:
                st.markdown(
                    f"""
                    <div class='rn-badge'>
                      <div class='rn-badge-icon'>{meta['icon']}</div>
                      <div>
                        <div class='rn-badge-label'>{t('result_label')}</div>
                        <div class='rn-badge-value'>{display}{f" <small>({predicted})</small>" if is_group else ""}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(t("result_conf"), f"{display_conf:.1f}%")
                st.progress(max(min(display_conf / 100.0, 1.0), 0.0))

                st.markdown(f"<p class='rn-section-title'>{t('result_dist')}</p>", unsafe_allow_html=True)
                render_result_bars(all_probs)

                if is_group and group_classes:
                    st.markdown(f"<p class='rn-section-title'>{t('group_detail')}</p>", unsafe_allow_html=True)
                    render_result_bars(all_probs, group_classes)

                tip = meta["tip"][st.session_state.lang]
                st.markdown(
                    f"""
                    <div class='rn-tip'>
                      <div class='rn-tip-title'>ℹ️ {t('result_tip')}</div>
                      <div>{tip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='rn-footer-space'></div>", unsafe_allow_html=True)
            action_col_1, action_col_2 = st.columns(2, gap="small")
            with action_col_1:
                st.button(t("btn_again"), use_container_width=True, key="classify_again_button", on_click=go, args=("classify", True))
            with action_col_2:
                st.button(t("btn_config"), use_container_width=True, key="open_config_bottom", on_click=go, args=("config", False))


def render_config_page() -> None:
    st.markdown(
        f"""
        <div class='rn-hero'>
          <h1>{t('config_title')}</h1>
          <p>{t('config_subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def add_group() -> None:
        st.session_state.groups.append(new_group())

    st.button(t("config_add"), key="add_group_button", on_click=add_group)

    if not st.session_state.groups:
        st.info(t("config_hint_empty"))

    for group_index, group in enumerate(st.session_state.groups):
        with st.container(border=True):
            header_left, header_right = st.columns([5, 1], gap="small", vertical_alignment="center")
            with header_left:
                new_name = st.text_input(
                    t("group_name"),
                    value=group["name"],
                    key=f"group_name_{group['id']}",
                    placeholder=t("group_name"),
                )
                st.session_state.groups[group_index]["name"] = new_name.strip()

            with header_right:
                if st.button("✕", key=f"delete_group_{group['id']}", use_container_width=True):
                    st.session_state.groups.pop(group_index)
                    st.rerun()

            st.markdown(f"**{t('group_classes')}**")
            current_classes = set(group["classes"])
            grouped_now = set(grouped_classes())
            cols = st.columns(3)

            for class_index, cls in enumerate(ALL_CLASSES):
                selected_elsewhere = cls in grouped_now and cls not in current_classes
                with cols[class_index % 3]:
                    checked = st.checkbox(
                        f"{CLASS_EMOJIS.get(cls, '')} {class_label(cls)}",
                        value=cls in current_classes,
                        key=f"group_{group['id']}_class_{cls}",
                        disabled=selected_elsewhere,
                    )
                    if checked and cls not in current_classes:
                        for other_index, other_group in enumerate(st.session_state.groups):
                            if other_index != group_index and cls in other_group["classes"]:
                                other_group["classes"].remove(cls)
                        st.session_state.groups[group_index]["classes"].append(cls)
                        st.rerun()
                    if not checked and cls in current_classes:
                        st.session_state.groups[group_index]["classes"].remove(cls)
                        st.rerun()

    ungrouped = ungrouped_classes()
    st.markdown(f"**{t('ungrouped')}:**")
    st.markdown(
        "<div class='rn-chip-row'>"
        + "".join(f"<span class='rn-chip'>{CLASS_EMOJIS.get(cls, '')} {class_label(cls)}</span>" for cls in ungrouped)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='rn-footer-space'></div>", unsafe_allow_html=True)
    back_col, save_col, reset_col = st.columns(3, gap="small")
    with back_col:
        st.button(t("btn_back_config"), use_container_width=True, on_click=go, args=("classify", False))
    with save_col:
        st.button(t("config_save"), use_container_width=True, type="primary", on_click=go, args=("classify", False))
    with reset_col:
        st.button(t("config_reset"), use_container_width=True, on_click=lambda: st.session_state.__setitem__("groups", []))


def render_recycling_page() -> None:
    st.markdown(
        f"""
        <div class='rn-hero'>
          <h1>{t('nav_recycling')}</h1>
          <p>{t('why_title')} · {t('how_title')} · {t('mendoza_title')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {t('why_title')}")
    st.markdown(t("info_why"))
    st.markdown(
        f"""
        <div class='rn-impact'>
          <div class='rn-impact-card'><div class='rn-impact-num'>{t('impact_1_num')}</div><div class='rn-impact-desc'>{t('impact_1_desc')}</div></div>
          <div class='rn-impact-card'><div class='rn-impact-num'>{t('impact_2_num')}</div><div class='rn-impact-desc'>{t('impact_2_desc')}</div></div>
          <div class='rn-impact-card'><div class='rn-impact-num'>{t('impact_3_num')}</div><div class='rn-impact-desc'>{t('impact_3_desc')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {t('how_title')}")
    st.markdown(t("info_how"))
    render_recycling_cards()

    st.markdown(f"### {t('mendoza_title')}")
    st.markdown(t("info_mendoza"))


def render_about_page() -> None:
    st.markdown(
        f"""
        <div class='rn-hero'>
          <h1>{t('about_title')}</h1>
          <p>{t('about_intro')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='rn-hero-grid'>
          <div class='rn-hero-panel'>
            <h3 class='rn-section-title'>UTN Facultad Regional Mendoza</h3>
            <p>{t('about_project')}</p>
          </div>
          <div class='rn-hero-panel'>
            <h3 class='rn-section-title'>{t('about_recycling')}</h3>
            <p>{t('about_recycling')}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {t('team_title')}")
    render_team_cards()

    st.markdown(f"### {t('about_repo')}")
    st.markdown("[" + t('repo_label') + " ↗](https://github.com/NachoBerridy/recyclenet/tree/dev)")


def get_model():
    device = get_device()
    model = load_model(MODEL_PATH, device)
    return model, device


MODEL_READY = os.path.exists(MODEL_PATH)

ss("lang", "es")
ss("dark", True)
ss("page", "classify")
ss("result", None)
ss("groups", [])

st.session_state.groups = normalize_groups(st.session_state.groups)

render_css()
render_nav()

if st.session_state.page == "classify":
    render_classify_page()
elif st.session_state.page == "recycling":
    render_recycling_page()
elif st.session_state.page == "about":
    render_about_page()
elif st.session_state.page == "config":
    render_config_page()
