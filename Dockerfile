# Use a minimal Python base image
FROM python:3.11-alpine as base

# Set working directory
WORKDIR /docker-app

# Copy only requirements first to leverage caching
COPY requirements.txt /docker-app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Separate build stage for frontend using a Node container
FROM node:18-alpine as frontend-builder

WORKDIR /frontend
COPY frontend /frontend

# Install dependencies and build frontend
RUN yarn install && yarn build

# Final image to keep it minimal
FROM python:3.11-alpine

WORKDIR /docker-app

# Copy backend files
COPY --from=base /docker-app /docker-app

# Copy the built frontend
COPY --from=frontend-builder /frontend/dist /docker-app/frontend/dist

# Expose the application port
EXPOSE 8070

# Command to run the application
CMD ["python", "run.py", "--setup"]
