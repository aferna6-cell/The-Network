# Always-on recommendation loop. Build once, run on any persistent host
# (Railway, Fly.io, a small VM). Mount a volume at /app/state so the prediction
# ledger survives restarts.
FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persisted state (ledger, recommendations snapshot). Mount a volume here.
VOLUME ["/app/state"]

# Provide your holdings at runtime, e.g.:
#   docker run -v $PWD/portfolio.json:/app/portfolio.json -v networkstate:/app/state the-network
CMD ["python", "scripts/run_live.py"]
