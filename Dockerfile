FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && python -m nltk.downloader -d /nltk_data punkt stopwords wordnet averaged_perceptron_tagger

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --from=builder /install /usr/local
COPY --from=builder /nltk_data /usr/local/share/nltk_data
COPY . .

RUN chown -R app:app /app
USER app

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
