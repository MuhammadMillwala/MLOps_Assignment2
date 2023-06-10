# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app files into the container
COPY . .
COPY app.py .
COPY btc_model.joblib .

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set the environment variables for the Flask app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app when the container starts
CMD ["flask", "run", "--host=0.0.0.0"]
