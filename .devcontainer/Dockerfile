# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY ./.devcontainer/requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r ./requirements.txt

