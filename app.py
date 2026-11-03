import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# 실행 위치와 관계없이 app.py가 있는 폴더를 프로젝트 기준으로 사용함
PROJECT_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="금속 표면 불량 유형 분류 시스템", layout="wide")

# 클래스 코드에 대응하는 한국어명과 간단한 결함 설명 지정함
CLASS_DETAILS = {
    "Cr": ("균열", "금속 표면에 발생한 균열 형태의 결함"),
    "In": ("개재물", "비금속 개재물 등 이물질과 관련된 결함"),
    "Pa": ("패치", "불규칙한 패치 형태의 표면 결함"),
    "PS": ("표면 패임", "작은 구멍이나 패임 형태의 결함"),
    "RS": ("압연 스케일", "압연 과정의 산화 스케일 관련 결함"),
    "Sc": ("스크래치", "선형으로 나타나는 긁힘 결함"),
}


# 모델과 정보 및 가중치를 공유하는 Grad-CAM 연결을 한 번만 불러옴
@st.cache_resource(show_spinner="분류 모델을 불러오는 중입니다...")
def load_resources():
    with (PROJECT_DIR / "models/model_info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    model = tf.keras.models.load_model(PROJECT_DIR / "models/cnn_model.keras", compile=False)

    # 10단계와 동일하게 기존 계층을 새 입력에 연결함
    last_conv = model.get_layer("last_conv")
    cam_input = tf.keras.Input(shape=(*info["image_size"], info["input_channels"]))
    x = cam_input
    for layer in model.layers[:-1]:
        x = layer(x)
        if layer is last_conv:
            feature_maps = x
    cam_model = tf.keras.Model(cam_input, [feature_maps, x])
    return model, info, cam_model, model.layers[-1]


# 원본 미리보기는 유지하고 예측용 이미지만 흑백 변환 및 정규화함
def preprocess_image(image, image_size):
    processed = image.convert("L").resize(tuple(image_size), Image.Resampling.BILINEAR)
    pixels = np.array(processed).astype(np.float32) / 255.0
    return processed, pixels[np.newaxis, ..., np.newaxis]


# 10단계와 동일한 예측 클래스 logit 기준 Grad-CAM 계산함
def make_gradcam(image_batch, class_index, cam_model, output_layer):
    with tf.GradientTape() as tape:
        feature_maps, dense_features = cam_model(
            tf.convert_to_tensor(image_batch, dtype=tf.float32), training=False
        )
        # 기존 Softmax와 가중치를 수정하지 않고 출력층의 logit 계산함
        logits = tf.matmul(dense_features, output_layer.kernel) + output_layer.bias
        class_score = logits[:, class_index]

    # 공간 평균 기울기로 채널별 가중치를 구하고 양의 기여도 합산함
    gradients = tape.gradient(class_score, feature_maps)
    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    # 0으로 나누지 않도록 정규화하고 입력 이미지 크기로 확대함
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], image_batch.shape[1:3])
    return tf.squeeze(heatmap, axis=-1).numpy()


# 위젯 key를 갱신하여 업로드와 분석 상태만 초기화하고 모델 캐시는 유지함
def reset_analysis():
    st.session_state.upload_generation += 1
    for key in ["upload_signature", "analysis_results", "prediction_cache", "gradcam_cache", "selected_image", "analysis_errors"]:
        st.session_state.pop(key, None)


# 확정된 세이지 그린 팔레트를 헤더와 기본 컴포넌트에 일관되게 적용함
st.markdown("""
<style>
:root {
    --sage-ivory: #F8F9F6; --sage-white: #FFFFFF; --sage-light: #E7EFE8;
    --sage-accent: #87A99B; --sage-deep: #497765; --sage-text: #253E35;
    --sage-muted: #59685F; --sage-border: #D4DED7;
}
.stApp { background: var(--sage-ivory); color: var(--sage-text); }
[data-testid="stHeader"] { background: var(--sage-ivory); }
[data-testid="stMainBlockContainer"] { max-width: 1320px; padding-top: 4.5rem; padding-bottom: 2rem; }
h1, h2, h3, label { color: var(--sage-text); }
[data-testid="stCaptionContainer"] { color: var(--sage-muted); }
.sage-header { background: var(--sage-light); border: 1px solid var(--sage-border); border-radius: 16px; padding: 22px 28px; margin-bottom: 20px; }
.sage-eyebrow { color: var(--sage-deep); font-size: 12px; font-weight: 700; letter-spacing: 0.12em; margin: 0 0 6px; }
.sage-header h1 { font-size: clamp(1.5rem, 2.6vw, 2.1rem); line-height: 1.4; word-break: keep-all; padding: 0; margin: 0 0 6px; }
.sage-subtitle { color: var(--sage-text); font-size: 14px; line-height: 1.65; margin: 0; }
.sage-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.sage-badge { background: var(--sage-white); color: var(--sage-deep); border: 1px solid var(--sage-border); border-radius: 20px; padding: 4px 11px; font-size: 12px; font-weight: 600; }
.st-key-upload_card, .st-key-detail_card, .st-key-probability_card, .st-key-gradcam_card { background: var(--sage-white); border-color: var(--sage-border); border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] { background: var(--sage-light); border: 1px dashed var(--sage-accent); border-radius: 10px; }
button[kind="primary"] { background: var(--sage-deep); border-color: var(--sage-deep); color: white; border-radius: 9px; }
button[kind="primary"]:hover:not(:disabled) { background: #3E6656; border-color: #3E6656; color: white; }
button[kind="primary"]:disabled { background: var(--sage-deep); border-color: var(--sage-deep); opacity: 0.45; }
button[kind="secondary"] { background: var(--sage-white); color: var(--sage-text); border-color: var(--sage-border); border-radius: 9px; }
button[kind="secondary"]:hover { background: var(--sage-light); color: var(--sage-text); border-color: var(--sage-accent); }
[data-testid="stMetric"] { background: var(--sage-light); border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricValue"] { color: var(--sage-deep); }
/* 이미지와 확률 그래프를 원래 비율로 축소하고 공통 높이 제한 제거함 */
[data-testid="stImage"] img { max-width: 100%; height: auto; object-fit: contain; }
/* 화면 너비 대신 Grad-CAM 카드의 실제 가용 너비로 배치 전환함 */
.st-key-gradcam_card { container-type: inline-size; container-name: gradcam; }
@container gradcam (max-width: 480px) {
    .st-key-gradcam_card > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { flex-direction: column; flex-wrap: nowrap; }
    .st-key-gradcam_card > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100%; flex: 0 0 auto; min-width: 0; }
}
[data-testid="stDataFrame"] { border: 1px solid var(--sage-border); border-radius: 10px; overflow: hidden; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: var(--sage-white); border-color: var(--sage-border); color: var(--sage-text); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within { border-color: var(--sage-deep); }
[data-testid="stExpander"] { background: var(--sage-white); border-radius: 12px; border-color: var(--sage-border); }
[data-testid="stExpander"] summary { color: var(--sage-text); }
[data-testid="stExpander"] summary:hover { background: var(--sage-light); color: var(--sage-text); }
.sage-empty { background: var(--sage-light); border: 1px solid var(--sage-border); border-radius: 10px; color: var(--sage-text); padding: 12px 16px; margin: 12px 0; }
@media (max-width: 640px) {
    .sage-header { padding: 18px; }
    .sage-header h1 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# 세션별 업로드 구분값과 예측 및 선택 이미지의 Grad-CAM 보관함 초기화함
if "upload_generation" not in st.session_state:
    st.session_state.upload_generation = 0
for key, default in [("analysis_results", []), ("prediction_cache", {}), ("gradcam_cache", {}), ("analysis_errors", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# 기존 리소스 로드 함수를 사용하고 실패 시 화면에 오류 안내함
resources = None
model_info = None
try:
    resources = load_resources()
    model, model_info, cam_model, output_layer = resources
    classes = sorted(model_info["classes"], key=lambda item: item["label"])
except Exception:
    resources = None

# 실제 JSON 정확도를 배지에 표시하고 세이지 헤더 카드 구성함
accuracy_badge = f"테스트 정확도 {model_info['test_accuracy'] * 100:.2f}%" if model_info is not None else "테스트 정확도 정보 없음"
st.markdown(f"""
<section class="sage-header">
    <p class="sage-eyebrow">AI VISION INSPECTION</p>
    <h1>금속 표면 불량 유형 분류 시스템</h1>
    <p class="sage-subtitle">CNN 딥러닝 기반 금속 표면 결함 분석 및 Grad-CAM 시각화</p>
    <div class="sage-badges">
        <span class="sage-badge">6종 불량 분류</span>
        <span class="sage-badge">{accuracy_badge}</span>
        <span class="sage-badge">Grad-CAM 분석</span>
    </div>
</section>
""", unsafe_allow_html=True)
if resources is None:
    st.error("저장된 모델 또는 모델 정보를 불러오지 못했습니다. 프로젝트의 모델 파일 준비 상태를 확인해 주세요.")

# 여러 파일을 받아도 업로드만으로 예측하지 않도록 버튼과 구분함
with st.container(border=True, key="upload_card"):
    st.subheader("이미지 업로드")
    uploaded_files = st.file_uploader(
        "분석할 금속 표면 이미지 선택", type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True, key=f"uploads_{st.session_state.upload_generation}",
    )
    st.caption(f"업로드 {len(uploaded_files)}장 / 최대 20장 · PNG, JPG, JPEG, BMP")
    if len(uploaded_files) > 20:
        st.warning("한 번에 최대 20장까지 분류할 수 있습니다. 일부 파일을 제거해 주세요.")

    # 파일 이름과 내용으로 구분하여 선택 변경 시 이전 결과가 남지 않게 함
    upload_keys = [f"{file.name}:{hashlib.sha256(file.getbuffer()).hexdigest()}" for file in uploaded_files]
    upload_signature = tuple(upload_keys)
    if upload_signature != st.session_state.get("upload_signature"):
        st.session_state.upload_signature = upload_signature
        st.session_state.analysis_results = []
        st.session_state.analysis_errors = []
        st.session_state.pop("selected_image", None)
        # 그대로 남아 있는 파일의 계산 결과만 보관하여 재예측 줄임
        st.session_state.prediction_cache = {key: value for key, value in st.session_state.prediction_cache.items() if key in upload_keys}
        st.session_state.gradcam_cache = {key: value for key, value in st.session_state.gradcam_cache.items() if key in upload_keys}

    # 분류는 유효한 업로드에서만 허용하고 초기화는 항상 사용 가능하게 함
    start_column, reset_column, _ = st.columns([1.4, 1.1, 4])
    start_clicked = start_column.button(
        "분류 시작", type="primary", width="stretch",
        disabled=not uploaded_files or len(uploaded_files) > 20 or resources is None,
    )
    reset_column.button("초기화", on_click=reset_analysis, width="stretch")

# 분류 버튼을 눌렀을 때 파일별로 독립 예측하고 결과를 세션에 보관함
if start_clicked:
    results = []
    errors = []
    with st.spinner("업로드한 이미지를 분류하는 중입니다..."):
        for uploaded_file, cache_key in zip(uploaded_files, upload_keys):
            if cache_key in st.session_state.prediction_cache:
                results.append(st.session_state.prediction_cache[cache_key])
                continue
            try:
                # 기존 전처리와 model.predict를 그대로 사용함
                uploaded_file.seek(0)
                with Image.open(uploaded_file) as image:
                    image.load()
                    processed_image, image_batch = preprocess_image(image, model_info["image_size"])
                    # 큰 원본 전체를 보관하지 않고 비율을 유지한 미리보기만 보관함
                    preview = image.copy()
                    preview.thumbnail((320, 240))
                probabilities = model.predict(image_batch, verbose=0)[0]
                predicted_label = int(np.argmax(probabilities))
                predicted_class = classes[predicted_label]
                korean_name, description = CLASS_DETAILS[predicted_class["code"]]

                # 파일명과 클래스 및 실제 확률과 전처리 이미지를 함께 저장함
                result = {
                    "key": cache_key, "filename": uploaded_file.name,
                    "predicted_label": predicted_label, "class_name": predicted_class["name"],
                    "korean_name": korean_name, "description": description,
                    "confidence": float(probabilities[predicted_label]),
                    "probabilities": probabilities, "processed_image": processed_image,
                    "image_batch": image_batch, "preview": preview,
                }
                st.session_state.prediction_cache[cache_key] = result
                results.append(result)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                errors.append(f"{uploaded_file.name}: 이미지를 열거나 분석할 수 없습니다.")
            except Exception:
                errors.append(f"{uploaded_file.name}: 이미지 분류를 완료하지 못했습니다.")
    st.session_state.analysis_results = results
    st.session_state.analysis_errors = errors
    st.session_state.pop("selected_image", None)

# 일부 파일에 오류가 있어도 정상 파일의 결과는 유지함
for message in st.session_state.analysis_errors:
    st.warning(message)
results = st.session_state.analysis_results
if not results:
    st.markdown('<div class="sage-empty">이미지를 선택한 뒤 ‘분류 시작’을 누르면 분석 결과가 표시됩니다.</div>', unsafe_allow_html=True)
else:
    # 단일 및 다중 이미지 모두 같은 전체 결과 표로 표시함
    st.subheader("전체 이미지 분류 결과")
    summary = pd.DataFrame([
        {"파일명": result["filename"], "예측 결과": f"{result['korean_name']} ({result['class_name']})", "예측 확률 (%)": result["confidence"] * 100}
        for result in results
    ])
    st.dataframe(
        summary.style.set_properties(**{"background-color": "#FFFFFF", "color": "#253E35"}), hide_index=True, width="stretch",
        column_config={"예측 확률 (%)": st.column_config.NumberColumn(format="%.1f%%")},
    )

    # 파일명이 같아도 순번으로 구분하며 선택 변경 시 전체 결과는 유지함
    st.subheader("선택 이미지 상세 분석")
    selected_index = st.selectbox(
        "상세 분석할 이미지", options=range(len(results)), key="selected_image",
        format_func=lambda index: f"{index + 1}. {results[index]['filename']}",
    )
    selected = results[selected_index]

    # 원본 미리보기와 강조된 예측 클래스 및 확률을 두 열로 표시함
    with st.container(border=True, key="detail_card"):
        preview_column, result_column = st.columns([1, 1.3])
        with preview_column:
            st.image(selected["preview"], caption=selected["filename"], width=280)
        with result_column:
            st.caption("예측 결과")
            st.subheader(f"{selected['korean_name']} ({selected['class_name']})")
            st.metric("예측 확률", f"{selected['confidence'] * 100:.1f}%")
            st.write(selected["description"])

    # 확률과 Grad-CAM을 두 열에 배치하여 세로 스크롤 줄임
    probability_column, gradcam_column = st.columns(2, gap="large")
    with probability_column, st.container(border=True, key="probability_card"):
        st.subheader("클래스별 예측 확률")
        probabilities = selected["probabilities"]
        probability_order = np.argsort(-probabilities)
        class_labels = [classes[int(i)]["name"] for i in probability_order]
        colors = ["#497765" if int(i) == selected["predicted_label"] else "#87A99B" for i in probability_order]
        # 그래프 배경과 글씨 및 격자선도 확정된 팔레트로 통일함
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors="#253E35")
        ax.xaxis.label.set_color("#253E35")
        ax.spines[["bottom", "left"]].set_color("#D4DED7")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#E0E6E1", linewidth=0.7)
        bars = ax.barh(class_labels, probabilities[probability_order] * 100, color=colors)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color="#253E35")
        ax.set(xlabel="Prediction probability (%)", xlim=(0, 114))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(f"확률 합계 {probabilities.sum() * 100:.1f}% · 진한 녹색: 최고 확률 클래스")

    with gradcam_column, st.container(border=True, key="gradcam_card"):
        st.subheader("Grad-CAM 시각화")
        try:
            # 선택한 이미지의 Grad-CAM만 계산하고 다시 선택할 때 재사용함
            if selected["key"] not in st.session_state.gradcam_cache:
                with st.spinner("선택 이미지의 Grad-CAM을 계산하는 중입니다..."):
                    heatmap = make_gradcam(selected["image_batch"], selected["predicted_label"], cam_model, output_layer)
                    grayscale_rgb = np.repeat(selected["image_batch"][0], 3, axis=-1)
                    heatmap_rgb = plt.get_cmap("jet")(heatmap)[..., :3]
                    st.session_state.gradcam_cache[selected["key"]] = np.clip(0.6 * grayscale_rgb + 0.4 * heatmap_rgb, 0, 1)
            original_column, overlay_column = st.columns(2)
            original_column.image(selected["processed_image"], caption="전처리 원본", width=220)
            overlay_column.image(st.session_state.gradcam_cache[selected["key"]], caption="Grad-CAM", width=220)
        except Exception:
            st.warning("분류 결과는 표시됐지만 Grad-CAM 이미지를 생성하지 못했습니다.")
        st.caption("Grad-CAM은 모델이 참고한 영역을 시각화하며, 실제 결함 위치를 정확하게 검출한 결과는 아닙니다.")

# 저장된 JSON 기반 모델 정보를 기본적으로 접어두고 필요할 때 표시함
with st.expander("모델 정보 및 테스트 성능", expanded=False):
    if model_info is not None:
        accuracy_column, f1_column = st.columns(2)
        accuracy_column.metric("테스트 정확도", f"{model_info['test_accuracy'] * 100:.2f}%")
        f1_column.metric("테스트 Macro F1-score", f"{model_info['test_macro_f1']:.4f}")
        st.table(pd.DataFrame({
            "항목": ["모델", "데이터셋", "분류 클래스", "입력 이미지"],
            "내용": [model_info["model_name"], model_info["dataset_name"],
                     f"{len(model_info['classes'])}개 · " + " / ".join(item["code"] for item in model_info["classes"]),
                     f"{model_info['image_size'][0]}×{model_info['image_size'][1]} · 흑백 {model_info['input_channels']}채널"],
        }).set_index("항목"))
    else:
        st.write("모델 정보를 표시할 수 없습니다.")

# 사용 범위와 해석 한계는 접기 영역에 간결하게 유지함
with st.expander("모델 사용 범위와 한계", expanded=False):
    st.markdown("""
    - NEU 데이터셋 기반의 6종 금속 표면 결함 분류 모델입니다.
    - 정상 제품은 별도로 판별하지 못하며, 다른 소재를 포함한 입력도 6개 클래스 중 하나로 분류합니다.
    - 높은 Softmax 확률이 실제 판정 신뢰도를 보장하지 않습니다.
    - Grad-CAM은 정확한 결함 위치 탐지 기능이 아닙니다.
    - 실제 제조 현장 검증을 거치지 않은 **학습용 프로토타입**입니다.
    """)


