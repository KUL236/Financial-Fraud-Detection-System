FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Pre-create directories
RUN mkdir -p data models reports

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\n\n[theme]\nbase = "dark"\nprimaryColor = "#63b3ed"\nbackgroundColor = "#0f172a"\nsecondaryBackgroundColor = "#1e293b"\ntextColor = "#e2e8f0"' > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
