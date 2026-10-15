# Render deployment notes

Deployment is not automatic. Configure the backend and frontend as separate Render services using the settings below. The Render Free backend is **not a reliable target and has not been verified to run**: with CPU-only PyTorch, the backend process used 537.1 MiB RSS after loading the app, sample PDF, and real Sentence Transformers model; after two real QA inferences, peak RSS reached 541.6 MiB. Both exceed Render Free's 512 MiB limit. This was measured on Windows, not Render's Linux process, but it shows the full app does not fit under the stated limit in this environment. Use an instance with at least 2 GiB RAM, then verify its memory from Render metrics before treating it as production-ready.

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
| `FRONTEND_ORIGINS` | `https://docu-mind-ai-phi.vercel.app` |
| `MAX_FILE_SIZE_MB` | `15` |

Render supplies `PORT` to the web service. Do not set a fixed production port; Uvicorn binds to `0.0.0.0` and uses `$PORT`. The `.python-version` file also pins Python to 3.13.3. This matches the local interpreter used for project tests; the application has not been tested on Render's Linux runtime.

The CORS variable must contain the exact deployed frontend origin, including `https://` and without a path or trailing slash. For multiple trusted frontends, use a comma-separated list of explicit origins. The backend rejects `*`, and CORS credentials are disabled. If `FRONTEND_ORIGINS` is unset, the backend's explicit defaults include the deployed Vercel origin and the two local development origins.

## Frontend: Vercel

Import the repository into Vercel and set the project root to `frontend`:

| Setting | Value |
| --- | --- |
| Framework Preset | Vite |
| Root Directory | `frontend` |
| Build Command | `npm run build` (or leave Vercel's detected Vite command) |
| Output Directory | `dist` |

Set this environment variable for Production (and Preview too if preview deployments should access the backend):

| Key | Value |
| --- | --- |
| `VITE_API_BASE_URL` | `https://documind-ai-jx13.onrender.com` |

The checked-in `frontend/.env.production` sets this same public backend origin for Vite production builds. The URL is not a secret; do not put credentials or API keys in any `VITE_` variable. Use the backend's origin only: do not append `/api`. Vite injects `VITE_` variables at build time, so changing the Vercel value requires a new frontend build/deployment. After setting `FRONTEND_ORIGINS` on Render, redeploy the backend too.

## Runtime, model, and upload notes

- `requirements.txt` pins PyTorch to `2.14.1+cpu` from the official PyTorch CPU wheel index, preventing pip from choosing a CUDA-enabled PyTorch distribution. The validated stack is Python 3.13.3, PyTorch 2.14.1+cpu, Sentence Transformers 6.1.0, and Transformers 5.19.0. The CPU build reports `torch.version.cuda is None` and `torch.cuda.is_available() is False`. The exact CPython 3.13 manylinux CPU wheel downloaded successfully for deployment validation (196.3 MB); dependency resolution against `requirements.txt` did not request NVIDIA/CUDA packages.
- The Sentence Transformers model is initialized lazily on first embedding request, exactly once per application process, and explicitly uses CPU. The sample PDF is indexed on the first library/search/question request rather than during module import, so Uvicorn can bind the port and pass `/health` before downloading/loading the model. That first user request can be slow; the model cache measured about 92 MB locally and a new instance needs outbound Hugging Face access. Lazy initialization improves port startup time but does not reduce steady-state model memory.
- Frontend API calls have a 120-second timeout and display a retry-oriented message if the backend is unavailable or still starting. Render Free web services spin down when idle, so users can experience a cold start. The `/health` check does not load the model.
- Uploads are limited to 15 MiB by default. `MAX_FILE_SIZE_MB` can change that limit; uploads are read only up to the configured limit plus one byte before rejection. Keep the limit small enough for the selected instance's available memory.
- Direct dependency versions are pinned in `requirements.txt`, including the CPU-only PyTorch wheel, and the file also retains test dependencies for local use. The project was tested in this workspace on Windows/Python 3.13.3; a clean Linux Render build has not been performed. Pinning these dependencies does not establish Linux deployment compatibility.
- Documents and embeddings live only in process memory. Uploaded documents may disappear after a restart, deploy, instance replacement, or scale-out; no persistent document storage is configured.

## Local development

The local backend command remains:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001
```

Run Vite from `frontend/` with `npm run dev`. When `VITE_API_BASE_URL` is empty or unset, the existing `/api` proxy continues to forward requests to `http://127.0.0.1:8001`. Use `.env.example` and `frontend/.env.example` as variable-name references; Vite and Uvicorn do not load these example files automatically.
