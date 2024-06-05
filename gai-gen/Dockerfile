# syntax=docker/dockerfile:1.2
FROM kakkoii1337/cu118-cp310-base:1.0 AS base

ARG CATEGORY=ttt
ARG DEBUG=1

# Step 1: Install utils
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && apt install -y git git-lfs ffmpeg python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Create env variables
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
ENV MODEL_PATH="/app/models"
RUN echo '{"app_dir":"/app"}' > /root/.gairc

# Step 3: Create mountpoint for models
WORKDIR /app/models

# Step 4: Install category specific requirements
WORKDIR /app
COPY "requirements_${CATEGORY}.txt" .
RUN pip install pip==23.3.1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_${CATEGORY}.txt

# Step 5: Install category specific apps
RUN if [ "$CATEGORY" == "ttt" ]; then \
        cp ./wheels/exllama-0.0.11-cp310-cp310-linux_x86_64.whl . && \
        pip install exllama-0.0.11-cp310-cp310-linux_x86_64.whl; \
    elif [ "$CATEGORY" == "itt" ]; then \
        cd ./external/LLaVA . && \
        pip install --no-cache-dir -e .; \
    fi

# **OPTIONAL**: Install debugger
RUN --mount=type=cache,target=/var/cache/apt \
    if [ "$DEBUG" = "1" ]; then \
        apt update && apt install -y curl procps netcat && \
        curl -sSL https://aka.ms/getvsdbgsh | bash /dev/stdin -v latest -l /vsdbg; \
    fi

# Step 6: Install Gai
WORKDIR /app
COPY ./requirements_*.txt .
COPY ./setup.py ./gai.json ./
COPY ./gai ./gai
RUN ln -s /app/gai /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Step 7: Start server
ENV CATEGORY=${CATEGORY}
WORKDIR /app/gai/api
ENTRYPOINT ["sh", "-c"]
CMD ["python ${CATEGORY}_api.py"]


