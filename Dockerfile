FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the required files into the container
COPY container/pandey /app/pandey
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE

# Install all the necessary packages from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to run the main function from pipeline.py
CMD ["python", "pandey/task/pipeline.py"]