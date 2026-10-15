# Render deployment notes

Deployment is not automatic. Configure the backend and frontend as separate Render services using the settings below. The Render Free backend is **not recommended and has not been verified to run**: the FastAPI application and Sentence Transformers model used 534.4 MiB resident memory in the local measurement in this repository, which is greater than the 512 MiB Free instance limit. This Windows measurement is a warning, not a substitute for measuring the Linux Render process. Use an instance with at least 2 GiB RAM, then verify its memory from Render metrics before treating it as production-ready.

Render references: [Python version selection](https://render.com/docs/python-version), [web services and port binding](https://render.com/docs/web-services#port-binding), [static sites](https://render.com/docs/static-sites), and [Free instance limitations](https://render.com/docs/free).

## Backend: Render Web Service

Create a Web Service from the repository:

| Setting | Value |
| --- | --- |
| Root Directory | Leave blank (repository root) |
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Instance | At least 2 GiB RAM recommended; do not use the 512 MiB Free instance based on the local measurement above |

Set these service environment variables:

| Key | Value |
| --- | --- |
| `PYTHON_VERSION` | `3.13.3` |
| `FRONTEND_ORIGINS` | `https://<your-frontend-service>.onrender.com` |
| `MAX_FILE_SIZE_MB` | `15` |

Render supplies `PORT` to the web service. Do not set a fixed production port; Uvicorn binds to `0.0.0.0` and uses `$PORT`. The `.python-version` file also pins Python to 3.13.3. This matches the local interpreter used for project tests; the application has not been tested on Render's Linux runtime.

The CORS variable must contain the exact deployed frontend origin, including `https://` and without a path or trailing slash. For multiple trusted frontends, use a comma-separated list of explicit origins. The backend rejects `*`, and CORS credentials are disabled.

## Frontend: Render Static Site

Create a Static Site from the same repository:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |

Set this environment variable before the first build:

| Key | Value |
| --- | --- |
| `VITE_API_BASE_URL` | `https://<your-backend-service>.onrender.com` |

Use the backend's origin only: do not append `/api`. Vite injects `VITE_` variables at build time, so changing this value requires a new static-site build/deploy. After creating both services, set `FRONTEND_ORIGINS` on the backend to the actual static-site origin and redeploy the backend.

## Runtime, model, and upload notes

- The backend imports and loads `sentence-transformers/all-MiniLM-L6-v2` while the FastAPI module loads. The included sample PDF is also indexed at that time. The first start requires outbound access to Hugging Face to download model files; the local model cache measured about 92 MB. A cold start or new instance can therefore take longer than subsequent requests.
- Frontend API calls have a 120-second timeout and display a retry-oriented message if the backend is unavailable or still starting. Render Free web services spin down when idle, so users can experience a cold start. The `/health` check is only available after the module and model finish loading.
- Uploads are limited to 15 MiB by default. `MAX_FILE_SIZE_MB` can change that limit; uploads are read only up to the configured limit plus one byte before rejection. Keep the limit small enough for the selected instance's available memory.
- The requirements file currently leaves package versions unpinned and includes test dependencies. The project was tested in this workspace on Python 3.13.3, but a clean Linux Render build has not been performed. Pin and validate dependency versions in a deployment build before relying on a production release.
- Documents and embeddings live only in process memory. Uploaded documents may disappear after a restart, deploy, instance replacement, or scale-out; no persistent document storage is configured.

## Local development

The local backend command remains:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001
```

Run Vite from `frontend/` with `npm run dev`. When `VITE_API_BASE_URL` is empty or unset, the existing `/api` proxy continues to forward requests to `http://127.0.0.1:8001`. Use `.env.example` and `frontend/.env.example` as variable-name references; Vite and Uvicorn do not load these example files automatically.
