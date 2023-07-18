# Use an official Python base image from the Docker Hub
FROM python:3.10

# Install browsers
RUN apt-get update && apt-get install -y \
    chromium-driver firefox-esr \
    ca-certificates \ 
    xvfb

# Install utilities
RUN apt-get install -y curl jq wget git

# Declare working directory
WORKDIR /workspace/answerfromKB

# Copy the current directory contents into the Workspace.
COPY . /workspace/answerfromKB

# Install any necessary packages specified in requirements.txt.
RUN pip install -r requirements.txt
