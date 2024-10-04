
# Base image
FROM python:3.11.4

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the application files
COPY . /app

# Command to run the API using Uvicorn
CMD ["streamlit", "run", "app.py"]
