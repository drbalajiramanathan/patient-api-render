# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 10000 available to the world outside this container
# This should match the port you specified in demo.launch()
EXPOSE 10000

# Define environment variable for the port (Render might use this)
ENV PORT=10000

# Run app.py when the container launches
CMD ["python", "app.py"]