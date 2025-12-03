# syntax=docker/dockerfile:1.4

########################
# Frontend Build Stage #
########################
FROM --platform=$BUILDPLATFORM node:18-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/ /frontend/

RUN if [ ! -d dist ]; then yarn install && yarn build; fi


#####################################
# Select correct Ultralytics base   #
#####################################

# Buildx will set TARGETARCH for us (amd64 / arm64)
ARG TARGETARCH

# AMD64 uses latest-cpu
FROM ultralytics/ultralytics:latest-cpu AS base-amd64

# ARM64 uses latest-arm64
FROM ultralytics/ultralytics:latest-arm64 AS base-arm64

# Dispatch to correct base image based on $TARGETARCH
FROM base-${TARGETARCH} AS runtime

# Re-declare ARG inside this stage so RUN can see it
ARG TARGETARCH

####################################
# Final Runtime Stage
####################################
WORKDIR /docker-app

# Copy backend
COPY . .

# Install requirements:
# - common requirements from requirements_docker.txt on all archs
# - tensorflow-cpu ONLY on amd64 (no wheels for arm64/Py3.12)
RUN pip install --no-cache-dir -r requirements_docker.txt \
    && if [ "$TARGETARCH" = "amd64" ]; then \
         pip install --no-cache-dir tensorflow-cpu; \
       else \
         echo "Skipping tensorflow-cpu on $TARGETARCH"; \
       fi \
    && rm -rf /root/.cache/pip

# Copy frontend built files
COPY --from=frontend-builder /frontend/dist /docker-app/frontend/dist

EXPOSE 8070

CMD ["python", "run.py", "--setup"]
