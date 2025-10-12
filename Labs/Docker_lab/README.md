Lab 2 - Docker FastAPI Application

Overview
This lab demonstrates building and running a FastAPI application using Docker containers. The application is a simple Sneaker Collection API that performs CRUD operations.
Application Details
This is a RESTful API for managing a sneaker collection with the following endpoints:

GET /sneakers - Get all sneakers
POST /sneakers - Add a new sneaker
GET /sneakers/{id} - Get a specific sneaker
PUT /sneakers/{id} - Update a sneaker
DELETE /sneakers/{id} - Delete a sneaker

Prerequisites

Docker installed on your machine
Basic understanding of Docker and FastAPI

How to Run
1. Build the Docker Image
bashdocker build -t sneaker-collection-api .
2. Run the Container
bashdocker run -d -p 8080:8080 sneaker-collection-api
3. Access the API

Sample Data
The API comes pre-loaded with three sneakers:

Air Jordan 1 (Nike, Chicago) - $170.00
Yeezy Boost 350 (Adidas, Cream White) - $220.00
Chuck Taylor All Star (Converse, Black) - $55.00
