Cars API Logging Lab 
A Python logging fundamentals lab using FastAPI to demonstrate different log levels for monitoring and debugging applications.

Overview
This lab demonstrates Python's logging module through a RESTful Cars API built with FastAPI. It showcases all major log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL, EXCEPTION) in practical scenarios.


Installation

Clone the repository:

bashgit clone https://github.com/YOUR_USERNAME/MLops_Labs.git
cd MLops_Labs/Logging_lab

Install dependencies:

bashpip install -r requirements.txt

Running the Application

Navigate to the src folder and start the FastAPI server:
bashcd src
python main.py
The API will be available at http://localhost:8080
All logs will be displayed in:

Console output (terminal)
Log file (cars_api.log)

Testing Different Log Levels:

Open a new terminal and run these curl commands to trigger different log levels:
1. DEBUG & INFO - Get all cars
bashcurl http://localhost:8080/cars
Expected Logs: DEBUG (fetching cars), INFO (returning results)
2. WARNING - Add car with duplicate ID
bashcurl -X POST http://localhost:8080/cars \
  -H "Content-Type: application/json" \
  -d '{"id":"1","make":"Ford","model":"Mustang","year":2024,"price":35000.00}'
Expected Logs: WARNING (duplicate ID detected)
3. WARNING - Update with significant price increase
bashcurl -X PUT http://localhost:8080/cars/1 \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Camry","year":2023,"price":70000.00}'
Expected Logs: WARNING (price increased from $28,999 to $70,000)
4. ERROR - Try to get non-existent car
bashcurl http://localhost:8080/cars/999
Expected Logs: ERROR (car not found)
5. CRITICAL - Delete the last remaining car
bashcurl -X DELETE http://localhost:8080/cars/2
curl -X DELETE http://localhost:8080/cars/3
curl -X DELETE http://localhost:8080/cars/1
Expected Logs: CRITICAL (deleting last car in inventory)