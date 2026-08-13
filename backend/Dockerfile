FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    MODEL_MODE=phase6 \
    ARTIFACT_DIR=/artifacts \
    SEGMENTATION_ENABLED=true \
    SEGMENTATION_MODEL_DIR=/artifacts/mask2former_kitti_best

WORKDIR /app
ARG SCIKIT_LEARN_VERSION="1.6.1"
ARG JOBLIB_VERSION="1.5.3"
ARG TORCH_VERSION="2.11.0"
ARG TORCHVISION_VERSION="0.26.0"
ARG TIMM_VERSION="1.0.19"
ARG TRANSFORMERS_VERSION="5.14.0"
ARG INSTALL_ML="true"
COPY pyproject.toml ./
COPY app ./app
COPY artifacts /artifacts
RUN pip install --upgrade pip \
    && pip install . \
    && pip install --force-reinstall "scikit-learn==${SCIKIT_LEARN_VERSION}" "joblib==${JOBLIB_VERSION}" \
    && if [ "$INSTALL_ML" = "true" ]; then \
         pip install --index-url https://download.pytorch.org/whl/cpu \
           "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
         && pip install "timm==${TIMM_VERSION}" "transformers==${TRANSFORMERS_VERSION}" \
         && mkdir -p /opt/huggingface \
         && python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)"; \
       fi \
    && mkdir -p /opt/huggingface \
    && chown -R 10001:10001 /opt/huggingface /artifacts

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
