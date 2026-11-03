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
st.set_page_config(page_title="금속 표면 불량 유형 분류 시스템", layout="centered")

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


# 프로젝트 제목과 사용 목적을 간결하게 표시함
st.title("금속 표면 불량 유형 분류 시스템")
st.caption("Metal Surface Defect Classification")
st.write("금속 표면 이미지를 업로드하면 6가지 불량 유형의 예측 확률과 Grad-CAM을 확인할 수 있습니다.")

# 모델 로드 실패 시에도 업로드 안내와 사용 한계 화면은 유지함
resources = None
model_info = None
try:
    resources = load_resources()
    model, model_info, cam_model, output_layer = resources
except Exception:
    st.error("저장된 모델 또는 모델 정보를 불러오지 못했습니다. 프로젝트의 모델 파일 준비 상태를 확인해 주세요.")

# 업로드 파일의 지원 형식을 제한하고 원본 이미지 미리보기 표시함
st.subheader("이미지 업로드")
uploaded_file = st.file_uploader("금속 표면 이미지 선택", type=["png", "jpg", "jpeg", "bmp"])
original_image = None
if uploaded_file is None:
    st.info("PNG, JPG, JPEG 또는 BMP 이미지를 선택해 주세요.")
else:
    try:
        with Image.open(uploaded_file) as image:
            image.load()
            original_image = image.copy()
        st.image(original_image, caption="업로드 원본 이미지", width="stretch")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("이미지를 열 수 없습니다. 정상적인 PNG, JPG, JPEG 또는 BMP 파일을 선택해 주세요.")

# 유효한 이미지와 모델이 준비된 경우에만 예측 수행함
if original_image is not None and resources is not None:
    try:
        processed_image, image_batch = preprocess_image(original_image, model_info["image_size"])
        with st.spinner("불량 유형을 분석하는 중입니다..."):
            probabilities = model.predict(image_batch, verbose=0)[0]
        predicted_label = int(np.argmax(probabilities))
        classes = sorted(model_info["classes"], key=lambda item: item["label"])
        predicted_class = classes[predicted_label]
        korean_name, description = CLASS_DETAILS[predicted_class["code"]]
    except Exception:
        st.error("이미지 분석을 완료하지 못했습니다. 다른 이미지로 다시 시도해 주세요.")
    else:
        # 예측 유형과 실제 Softmax 확률 및 짧은 설명 표시함
        st.subheader("불량 유형 예측 결과")
        st.success(f"예측 결과: {korean_name} ({predicted_class['name']})")
        st.metric("예측 확률", f"{probabilities[predicted_label] * 100:.1f}%")
        st.write(description)

        # 여섯 클래스의 확률을 백분율로 비교하고 최고값 막대 강조함
        st.subheader("클래스별 예측 확률")
        class_labels = [f"{item['name']} ({item['code']})" for item in classes]
        colors = ["#2878b5" if i == predicted_label else "#bdc9d5" for i in range(len(classes))]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(class_labels, probabilities * 100, color=colors)
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
        ax.set(title="Class Prediction Probabilities", xlabel="Prediction probability (%)", ylabel="Defect type", xlim=(0, 112))
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(f"클래스별 확률 합계: {probabilities.sum() * 100:.1f}% · 파란색 막대는 최고 확률 클래스입니다.")

        # 원본 구조가 보이도록 정규화 이미지 위에 컬러 히트맵을 겹침
        st.subheader("Grad-CAM 판단 근거")
        try:
            heatmap = make_gradcam(image_batch, predicted_label, cam_model, output_layer)
            grayscale_rgb = np.repeat(image_batch[0], 3, axis=-1)
            heatmap_rgb = plt.get_cmap("jet")(heatmap)[..., :3]
            overlay = np.clip(0.6 * grayscale_rgb + 0.4 * heatmap_rgb, 0, 1)
            left, right = st.columns(2)
            left.image(processed_image, caption="전처리된 원본 이미지", width="stretch")
            right.image(overlay, caption="Grad-CAM 오버레이", width="stretch")
            st.caption("빨간색에 가까울수록 해당 이미지에서 상대적으로 높은 양의 기여도를 나타냅니다.")
        except Exception:
            st.warning("분류 결과는 표시됐지만 Grad-CAM 이미지를 생성하지 못했습니다.")
        st.info("Grad-CAM은 모델이 예측할 때 참고한 영역을 시각화한 것으로, 실제 결함 위치를 정확하게 검출한 결과는 아닙니다.")

# 성능 수치를 하드코딩하지 않고 저장된 JSON 값으로 모델 정보 표시함
st.divider()
st.subheader("모델 정보 및 테스트 성능")
if model_info is not None:
    accuracy_column, f1_column = st.columns(2)
    accuracy_column.metric("테스트 정확도", f"{model_info['test_accuracy'] * 100:.2f}%")
    f1_column.metric("테스트 Macro F1-score", f"{model_info['test_macro_f1']:.4f}")
    st.table(pd.DataFrame({
        "항목": ["모델", "데이터셋", "분류 클래스", "입력 이미지"],
        "내용": [model_info["model_name"], model_info["dataset_name"],
                 f"{len(model_info['classes'])}개", f"{model_info['image_size'][0]}×{model_info['image_size'][1]} · 흑백 {model_info['input_channels']}채널"],
    }).set_index("항목"))
else:
    st.write("모델 정보를 표시할 수 없습니다.")

# 학습 범위와 실제 사용 시 해석 한계를 화면 하단에 안내함
st.subheader("모델 사용 범위와 한계")
st.markdown("""
- NEU 데이터셋의 6가지 금속 표면 불량 유형을 분류하도록 학습했습니다.
- 정상 제품 이미지는 학습하지 않아, 정상 이미지나 다른 소재도 6개 결함 중 하나로 예측할 수 있습니다.
- 높은 Softmax 확률이 실제 현장 판정의 신뢰도를 보증하지 않습니다.
- Grad-CAM은 결함 위치를 정확하게 검출하는 기능이 아닙니다.
- 실제 제조 현장의 품질 판정을 대체하지 않는 **학습용 프로토타입**입니다.
""")
