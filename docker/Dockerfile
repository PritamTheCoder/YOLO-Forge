# docker/Dockerfile
# Multi-stage: Builder -> Slim Runtime
# Runs YOLO-Forge fully with report generation 

##############################
###### BUILD STAGE ###########
##############################
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System packages needed for OpenCV & Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc wget git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Libraries & code
COPY src ./src
COPY configs ./configs
COPY docker/run.sh ./run.sh
RUN chmod +x run.sh


##############################
###### RUNTIME STAGE #########
##############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Inject Poetry-installed site-packages from build stage
COPY --from=build /usr/local/lib/python3.11 /usr/local/lib/python3.11

# Copy application content
COPY --from=build /app/src /app/src
COPY --from=build /app/configs /app/configs
COPY --from=build /app/run.sh /app/run.sh

# Create safe runtime user
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH="/app/src"

ENTRYPOINT ["/app/run.sh"]
CMD ["--help"]
