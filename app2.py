import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from streamlit_drawable_canvas import st_canvas

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.set_page_config(page_title="0 to 999 Number Recognition", layout="centered")

MODEL_FILE = "best_digit_model.keras"

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 8px;
    white-space: nowrap;
}
.sub-text {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-bottom: 28px;
}
.result-box {
    padding: 16px;
    border-radius: 12px;
    background-color: #111827;
    border: 1px solid #374151;
    margin-top: 15px;
}
.small-note {
    font-size: 14px;
    color: #bbbbbb;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Train model if not found
# -------------------------------
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10
    )
    datagen.fit(x_train)

    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=64),
        validation_data=(x_test, y_test_cat),
        epochs=12,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    if os.path.exists(MODEL_FILE):
        return load_model(MODEL_FILE)

    model.save(MODEL_FILE)
    return model


@st.cache_resource
def load_digit_model():
    return load_model(MODEL_FILE)
    


# -------------------------------
# Preprocess single digit
# -------------------------------
def preprocess_single_digit(digit_img_array):
    coords = np.argwhere(digit_img_array > 0)

    if coords.size == 0:
        blank = np.zeros((28, 28), dtype=np.float32)
        return Image.fromarray((blank * 255).astype(np.uint8)), blank.reshape(1, 28, 28, 1)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = digit_img_array[y_min:y_max + 1, x_min:x_max + 1].astype(np.uint8)
    cropped_img = Image.fromarray(cropped, mode="L")

    cropped_img = ImageOps.expand(cropped_img, border=8, fill=0)
    cropped_img = cropped_img.filter(ImageFilter.GaussianBlur(0.5))
    cropped_img.thumbnail((20, 20))

    new_img = Image.new("L", (28, 28), 0)
    paste_x = (28 - cropped_img.size[0]) // 2
    paste_y = (28 - cropped_img.size[1]) // 2
    new_img.paste(cropped_img, (paste_x, paste_y))

    final_array = np.array(new_img).astype("float32") / 255.0
    final_array = final_array.reshape(1, 28, 28, 1)

    return new_img, final_array


# -------------------------------
# Segment multiple digits
# -------------------------------
def segment_digits_from_canvas(canvas_image):
    img = Image.fromarray(canvas_image.astype("uint8"), mode="RGBA").convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    arr = np.where(arr > 100, 255, 0).astype(np.uint8)

    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return [], arr

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    col_sum = np.sum(arr > 0, axis=0)
    non_empty_cols = np.where(col_sum > 0)[0]

    if len(non_empty_cols) == 0:
        return [], arr

    groups = []
    start = non_empty_cols[0]
    prev = non_empty_cols[0]

    min_gap = 10

    for c in non_empty_cols[1:]:
        if c - prev > min_gap:
            groups.append((start, prev))
            start = c
        prev = c
    groups.append((start, prev))

    digit_images = []
    for left, right in groups:
        digit_crop = arr[:, left:right + 1]

        coords_digit = np.argwhere(digit_crop > 0)
        if coords_digit.size == 0:
            continue

        yy_min, xx_min = coords_digit.min(axis=0)
        yy_max, xx_max = coords_digit.max(axis=0)
        digit_crop = digit_crop[yy_min:yy_max + 1, xx_min:xx_max + 1]

        if digit_crop.shape[0] < 8 or digit_crop.shape[1] < 4:
            continue

        digit_images.append(digit_crop)

    return digit_images, arr


# -------------------------------
# Predict full number up to 999
# -------------------------------
def predict_number_from_canvas(model, canvas_image):
    digit_crops, processed_binary = segment_digits_from_canvas(canvas_image)

    if len(digit_crops) == 0:
        return None, [], [], processed_binary, "First, draw a number on the canvas."

    if len(digit_crops) > 3:
        return None, [], [], processed_binary, "Write a number only from 0 to 999. Too many digits were detected."

    predicted_digits = []
    processed_digit_images = []
    confidences = []

    for crop in digit_crops:
        processed_pil, processed_array = preprocess_single_digit(crop)
        pred = model.predict(processed_array, verbose=0)[0]
        digit = int(np.argmax(pred))
        conf = float(np.max(pred) * 100)

        predicted_digits.append(str(digit))
        confidences.append(conf)
        processed_digit_images.append(processed_pil)

    predicted_number_str = "".join(predicted_digits)

    try:
        predicted_number = int(predicted_number_str)
    except ValueError:
        return None, predicted_digits, confidences, processed_digit_images, "The number could not be read."

    if predicted_number < 0 or predicted_number > 999:
        return predicted_number, predicted_digits, confidences, processed_digit_images, \
            "A number was detected, but it is outside the range of 0 to 999."

    return predicted_number, predicted_digits, confidences, processed_digit_images, None


# -------------------------------
# UI
# -------------------------------
st.markdown('<div class="main-title">✍️ Handwritten Number Recognition (0 to 999)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Draw a number from 0 to 999 on the canvas. Keep some space between the digits. Example: 7, 25, 100, 999.</div>',
    unsafe_allow_html=True
)

with st.spinner("The model is loading, please wait..."):
    model = load_digit_model()

brush_size = st.slider("Brush Size", 15, 40, 25)

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=brush_size,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=800,
    height=240,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}"
)

st.markdown(
    '<div class="small-note">Tip: Write the digits separately. For example, draw 1 0 0 or 9 9 9 with a small gap between each digit.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("Predict Number", use_container_width=True)

with col2:
    clear_btn = st.button("Clear Canvas", use_container_width=True)

if clear_btn:
    st.session_state.canvas_key += 1
    st.rerun()

if predict_btn:
    if canvas_result.image_data is None:
        st.warning("The canvas could not be loaded.")
    else:
        image_rgba = canvas_result.image_data

        if np.all(image_rgba[:, :, :3] == 255):
            st.warning("First, draw a number on the canvas.")
        else:
            number, digits, confidences, processed_items, error_msg = predict_number_from_canvas(model, image_rgba)

            if error_msg:
                st.warning(error_msg)

            if digits and processed_items:
                st.subheader("Detected Digits")
                cols = st.columns(len(processed_items))
                for i, item in enumerate(processed_items):
                    with cols[i]:
                        st.image(item, caption=f"Digit {i+1}", width=90)

                st.subheader("Digit-wise Prediction")
                for i, (d, c) in enumerate(zip(digits, confidences), start=1):
                    st.write(f"Digit {i}: **{d}**  | Confidence: **{c:.2f}%**")

            if number is not None:
                if 0 <= number <= 999:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success(f"Final Predicted Number: {number}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Predicted Number: {number} (Range 0 to 999 hona chahiye)")

st.markdown("---")
st.write(
    "This app uses a single-digit CNN model to recognize multi-digit handwritten numbers. "
    "Your original uploaded app also used a base model that classified digits only from 0 to 9. "
    ":contentReference[oaicite:0]{index=0}"
)
