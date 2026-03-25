FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DEMO_MODE=true \
    OPENAI_MODEL_FILTER=gpt-4o-mini

WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application with proper ownership
COPY --chown=user . $HOME/app

# Create the data directory explicitly and ensure it is writable
RUN mkdir -p $HOME/app/data/demo_guests && chmod -R 777 $HOME/app/data

EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--threads", "4", "--timeout", "120", "app:app"]
