from __future__ import annotations

import json
import hashlib

import streamlit as st
from PIL import Image

from predictor import predict_fashion


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fashion Product Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# Keep styling theme-friendly
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .hero-subtitle {
        opacity: 0.70;
        font-size: 1rem;
        margin-bottom: 0.7rem;
    }

    .disclaimer {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.30);
        background: rgba(128, 128, 128, 0.08);
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }

    .section-heading {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.35rem;
        margin-bottom: 0.6rem;
    }

    .result-label {
        font-size: 0.98rem;
        font-weight: 650;
    }

    .result-rank {
        opacity: 0.55;
        font-size: 0.82rem;
        margin-right: 0.3rem;
    }

    .confidence-text {
        opacity: 0.70;
        font-size: 0.82rem;
    }

    .status-success {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.35);
        margin-bottom: 1rem;
    }

    .status-warning {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.35);
        margin-bottom: 1rem;
    }

    .small-muted {
        opacity: 0.62;
        font-size: 0.82rem;
    }

    .model-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(128, 128, 128, 0.30);
        font-size: 0.78rem;
        margin-right: 0.35rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">👕 Fashion Product Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        CLIP guardrail + class-weighted Vision Transformer for
        fashion attribute prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This model currently supports
        <b>one fashion product per image</b>. For outfit or
        multi-product images, predictions may be mixed across items.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="model-pill">CLIP Guardrail</span>
    <span class="model-pill">Class-weighted ViT</span>
    <span class="model-pill">Top-K Predictions</span>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE UPLOADER
# ============================================================

st.markdown("### Upload image")

uploaded_file = st.file_uploader(
    "Choose a fashion/product image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)


# ============================================================
# PROCESS ONLY WHEN A NEW IMAGE IS UPLOADED
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    if current_hash != st.session_state.file_hash:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.session_state.uploaded_image = image
        st.session_state.file_hash = current_hash

        with st.spinner("Analyzing image with CLIP + ViT..."):

            result = predict_fashion(
                image
            )

        st.session_state.prediction = result


# # ============================================================
# # CONTROLS
# # ============================================================

# if st.session_state.prediction is not None:

#     st.markdown("### Display controls")

#     top_k = st.slider(
#         "Number of predictions per category",
#         min_value=1,
#         max_value=4,
#         value=1,
#         step=1,
#         help="Choose how many top predictions to show for each task.",
#     )

#     show_confidence = st.toggle(
#         "Show confidence percentages",
#         value=True,
#         help="Display the model confidence beside each prediction.",
#     )

#     st.caption(
#         f"Showing Top-{top_k} predictions"
#         + (
#             " with confidence values."
#             if show_confidence
#             else "."
#         )
#     )

#     st.divider()


# ============================================================
# MAIN CONTENT
# ============================================================

if (
    st.session_state.uploaded_image is not None
    and st.session_state.prediction is not None
):

    left_col, right_col = st.columns(
        [1, 1.15],
        gap="large",
    )


    # ========================================================
    # LEFT SIDE — IMAGE
    # ========================================================

    with left_col:

        st.markdown("### Uploaded Image")

        with st.container(border=True):

            st.image(
                st.session_state.uploaded_image,
                use_container_width=True,
            )

        image = st.session_state.uploaded_image

        st.markdown(
            f"""
            <div class="small-muted">
                Resolution: {image.width} × {image.height}px
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # RIGHT SIDE — RESULTS
    # ========================================================

    with right_col:

        result = st.session_state.prediction
                # ----------------------------------------------------
        # DISPLAY CONTROLS
        # ----------------------------------------------------

        

        guardrail = result.get(
            "guardrail",
            {},
        )

        is_clothing = guardrail.get(
            "is_clothing",
            False,
        )

        guardrail_confidence = float(
            guardrail.get(
                "confidence",
                0.0,
            )
        )
        st.markdown("### Display controls")

    # Disable fashion-specific controls when
    # CLIP determines the image is not clothing.
        top_k = st.slider(
            "Top-K predictions",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            disabled=not is_clothing,
            help=(
                "Available only for fashion images."
                if not is_clothing
                else "Choose how many predictions to show."
            )
        )

        show_confidence = st.toggle(
            "Show confidence percentages",
            value=True,
            disabled=not is_clothing,
            help=(
                "Available only for fashion images."
                if not is_clothing
                else "Display prediction confidence."
            )
        )

        if is_clothing:

            st.caption(
                f"Showing Top-{top_k} predictions"
                + (
                    " with confidence values."
                    if show_confidence
                    else ""
                )
            )

        else:

            st.caption(
                "Fashion prediction controls are disabled "
                "because this image was classified as non-fashion."
            )

        st.divider()


        # ----------------------------------------------------
        # CLIP GUARDRAIL
        # ----------------------------------------------------

        if is_clothing:

            st.markdown(
                f"""
                <div class="status-success">
                    ✅ <b>Fashion item detected</b><br>
                    CLIP confidence: {guardrail_confidence * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="status-warning">
                    ⚠️ <b>Non-fashion image detected</b><br>
                    CLIP confidence: {guardrail_confidence * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # NON-CLOTHING RESPONSE
        # ----------------------------------------------------

        if not is_clothing:

            st.warning(
                result.get(
                    "message",
                    "This image does not appear to be "
                    "a fashion product.",
                )
            )

        else:

            predictions = result.get(
                "predictions",
                {},
            )

            display_names = {
                "masterCategory": "Master Category",
                "subCategory": "Sub Category",
                "articleType": "Article Type",
                "baseColour": "Base Colour",
                "usage": "Usage",
            }

            # ------------------------------------------------
            # FIVE TASKS
            # ------------------------------------------------

            for task in [
                "masterCategory",
                "subCategory",
                "articleType",
                "baseColour",
                "usage",
            ]:

                values = predictions.get(
                    task,
                    [],
                )

                if not values:
                    continue

                st.markdown(
                    f"""
                    <div class="section-heading">
                        {display_names[task]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                visible_values = values[
                    :top_k
                ]

                for rank, prediction in enumerate(
                    visible_values,
                    start=1,
                ):

                    label = prediction.get(
                        "label",
                        "Unknown",
                    )

                    confidence = float(
                        prediction.get(
                            "confidence",
                            0.0,
                        )
                    )

                    percent = confidence * 100

                    # -----------------------------
                    # Label + confidence
                    # -----------------------------

                    left, right = st.columns(
                        [4, 1]
                    )

                    with left:

                        st.markdown(
                            f"""
                            <span class="result-rank">
                                #{rank}
                            </span>
                            <span class="result-label">
                                {label}
                            </span>
                            """,
                            unsafe_allow_html=True,
                        )

                    with right:

                        if show_confidence:

                            st.markdown(
                                f"""
                                <div style="text-align:right;">
                                    <span class="confidence-text">
                                        {percent:.2f}%
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # -----------------------------
                    # Confidence bar
                    # -----------------------------

                    if show_confidence:
                        st.progress(
                            confidence
                        )

                st.write("")


# ============================================================
# JSON OUTPUT
# ============================================================

if st.session_state.prediction is not None:

    st.divider()

    st.markdown("### Model output")

    json_string = json.dumps(
        st.session_state.prediction,
        indent=2,
    )

    json_left, json_right = st.columns(
        [1, 1]
    )

    with json_left:

        with st.expander(
            "View raw JSON",
            expanded=False,
        ):

            st.json(
                st.session_state.prediction
            )

    with json_right:

        st.download_button(
            label="⬇️ Download JSON",
            data=json_string,
            file_name="fashion_prediction.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fashion Product Classifier • "
    "CLIP + Class-weighted ViT • "
    "Top-K fashion attribute prediction"
)