# Use the Python 3.10 slim-buster image as the base image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# # Copy the model and data directories to the Docker image
# COPY model/ model/
# COPY data/ data/

# Copy the app directory to the working directory
COPY app/ .

# Copy the model and data directories to the Docker image
# COPY ../model/ /app/model/
# COPY ../data/ /app/data/


# Expose port 8501 to allow external connections
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/main.py"]