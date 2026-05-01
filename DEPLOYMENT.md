# Deployment

This project is best deployed as a Streamlit app. The backend is a local ML pipeline, not a separate production API.

## Streamlit Community Cloud

1. Create a new Streamlit app from this repository.
2. Set the app entrypoint:

```text
app/streamlit_app.py
```

3. Use `requirements.txt` for dependencies.
4. Confirm sample documents exist under `data/sample`.
5. Add optional environment variables from `.env.example` if you want non-default clustering settings.

## Docker

```bash
docker build -t document-clustering-topic-modeling .
docker run -p 8501:8501 document-clustering-topic-modeling
```

## Production Notes

- Keep demo datasets small enough for fast startup.
- Do not commit private documents.
- Store generated artifacts outside Git unless they are small examples.
- Re-run `scripts/evaluate.py` after changing preprocessing or clustering settings.
