# Step 1: Start with an official Python base image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the necessary files into the container
# Copy requirements first to leverage Docker's layer caching
COPY requirement_core.txt .

# Copy the Flask application code and model files
COPY Flask_app/ ./Flask_app
COPY bow_vectorizer.pkl .
COPY decisiontree_model.pkl .

# Install git, which is needed for pip to install from a git repository
RUN apt-get update && apt-get install -y git

# Step 4: Install the Python dependencies
RUN pip install --no-cache-dir -r requirement_core.txt

# Step 5: Expose the port the Flask app runs on
EXPOSE 5000

# Step 6: Define the command to run the application
# Use gunicorn for a production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "Flask_app.app:app"]