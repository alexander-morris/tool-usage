# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .


# Install Git and any other necessary packages
RUN apt-get update && apt-get install -y git sudo

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Set the command to run your application (modify as needed)
CMD ["/bin/bash"]
