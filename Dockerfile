FROM python:3.11
WORKDIR /app
COPY model_save_detec_tumeur /app/
COPY app.py /app/
COPY requirements.txt /app/
EXPOSE 8000
RUN pip install -r requirements.txt
CMD ["python", "app.py"]