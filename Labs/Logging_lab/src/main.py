from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cars_api.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Create a custom logger for the cars API
logger = logging.getLogger("cars_api")

# Car model using Pydantic for request/response validation
class Car(BaseModel):
    id: str
    make: str
    model: str
    year: int
    price: float

# Update model (for PUT requests where ID might not be in body)
class CarUpdate(BaseModel):
    make: str
    model: str
    year: int
    price: float

# Initialize FastAPI app
app = FastAPI(title="Cars API", version="1.0.0")

cars = [
    Car(id="1", make="Toyota", model="Camry", year=2023, price=28999.99),
    Car(id="2", make="Honda", model="Civic", year=2024, price=26500.00),
    Car(id="3", make="Tesla", model="Model 3", year=2024, price=42990.00),
]

logger.info("Cars API initialized with %d cars", len(cars))

# GET /cars - Get all cars
@app.get("/cars", response_model=List[Car])
async def get_cars():
    """Get all cars"""
    logger.debug("Fetching all cars from the inventory")
    logger.info("GET /cars - Returning %d cars", len(cars))
    return cars

# POST /cars - Create a new car
@app.post("/cars", response_model=Car, status_code=201)
async def post_cars(car: Car):
    """Add a new car"""
    logger.debug("Attempting to add new car: %s", car.dict())
    
    # Check for duplicate ID (this will generate a WARNING)
    for existing_car in cars:
        if existing_car.id == car.id:
            logger.warning("Duplicate car ID detected: %s. Overwriting existing car.", car.id)
    
    cars.append(car)
    logger.info("POST /cars - Successfully added car: %s %s %s", car.year, car.make, car.model)
    return car

# GET /cars/{id} - Get car by ID
@app.get("/cars/{id}", response_model=Car)
async def get_car_by_id(id: str):
    """Get a specific car by ID"""
    logger.debug("Searching for car with ID: %s", id)
    
    for car in cars:
        if car.id == id:
            logger.info("GET /cars/%s - Car found: %s %s %s", id, car.year, car.make, car.model)
            return car
    
    logger.error("GET /cars/%s - Car not found", id)
    raise HTTPException(status_code=404, detail="car not found")

# PUT /cars/{id} - Update an existing car
@app.put("/cars/{id}", response_model=Car)
async def update_existing_car(id: str, updated_car: CarUpdate):
    """Update an existing car"""
    logger.debug("Attempting to update car with ID: %s", id)
    
    # Check for suspicious price changes
    for i, car in enumerate(cars):
        if car.id == id:
            old_price = car.price
            new_price = updated_car.price
            
            if new_price > old_price * 2:
                logger.warning("Significant price increase detected for car %s: $%.2f -> $%.2f", 
                             id, old_price, new_price)
            
            # Update the car fields
            cars[i].make = updated_car.make
            cars[i].model = updated_car.model
            cars[i].year = updated_car.year
            cars[i].price = updated_car.price
            
            logger.info("PUT /cars/%s - Successfully updated car: %s %s %s", id, cars[i].year, cars[i].make, cars[i].model)
            return cars[i]
    
    logger.error("PUT /cars/%s - Car not found for update", id)
    raise HTTPException(status_code=404, detail="car not found")

# DELETE /cars/{id} - Delete a car
@app.delete("/cars/{id}")
async def delete_car(id: str):
    """Delete a car by ID"""
    logger.debug("Attempting to delete car with ID: %s", id)
    
    # Check if we're deleting the last car (CRITICAL situation)
    if len(cars) == 1:
        logger.critical("CRITICAL: Attempting to delete the last car in the inventory!")
    
    for i, car in enumerate(cars):
        if car.id == id:
            deleted_car = cars.pop(i)
            logger.info("DELETE /cars/%s - Successfully deleted car: %s %s %s", id, deleted_car.year, deleted_car.make, deleted_car.model)
            return {"message": "car deleted successfully"}
    
    logger.error("DELETE /cars/%s - Car not found for deletion", id)
    raise HTTPException(status_code=404, detail="car not found")

def print_this():
    """Test function that demonstrates exception logging"""
    try:
        if cars:
            logger.debug("print_this() called - accessing first car")
            print(cars[0].id)
            
        # Simulate a potential error scenario
        result = 10 / 0  # This will raise ZeroDivisionError
    except ZeroDivisionError:
        logger.exception("Exception occurred in print_this() function")

# Main entry point
if __name__ == "__main__":
    logger.info("Starting Cars API server on http://0.0.0.0:8080")
    print_this()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")