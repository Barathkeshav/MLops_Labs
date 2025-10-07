from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Sneaker model using Pydantic for request/response validation.
class Sneaker(BaseModel):
    id: str
    name: str
    brand: str
    color: str
    price: float

# Update model (for PUT requests where ID might not be in body)
class SneakerUpdate(BaseModel):
    name: str
    brand: str
    color: str
    price: float

# Initialize FastAPI app
app = FastAPI(title="Sneaker Collection API", version="1.0.0")

sneakers = [
    Sneaker(id="1", name="Air Jordan 1", brand="Nike", color="Chicago", price=170.0),
    Sneaker(id="2", name="Yeezy Boost 350", brand="Adidas", color="Cream White", price=220.0),
    Sneaker(id="3", name="Chuck Taylor All Star", brand="Converse", color="Black", price=55.0),
]

# GET /sneakers - Get all sneakers
@app.get("/sneakers", response_model=List[Sneaker])
async def get_sneakers():
    """Get all sneakers"""
    return sneakers

# POST /sneakers - Create a new sneaker
@app.post("/sneakers", response_model=Sneaker, status_code=201)
async def post_sneaker(sneaker: Sneaker):
    """Add a new sneaker"""
    sneakers.append(sneaker)
    return sneaker

# GET /sneakers/{id} - Get sneaker by ID
@app.get("/sneakers/{id}", response_model=Sneaker)
async def get_sneaker_by_id(id: str):
    """Get a specific sneaker by ID"""
    for sneaker in sneakers:
        if sneaker.id == id:
            return sneaker
    raise HTTPException(status_code=404, detail="sneaker not found")

# PUT /sneakers/{id} - Update an existing sneaker
@app.put("/sneakers/{id}", response_model=Sneaker)
async def update_existing_sneaker(id: str, updated_sneaker: SneakerUpdate):
    """Update an existing sneaker"""
    for i, sneaker in enumerate(sneakers):
        if sneaker.id == id:
            # Update the sneaker fields
            sneakers[i].name = updated_sneaker.name
            sneakers[i].brand = updated_sneaker.brand
            sneakers[i].color = updated_sneaker.color
            sneakers[i].price = updated_sneaker.price
            return sneakers[i]
    
    raise HTTPException(status_code=404, detail="sneaker not found")

# DELETE /sneakers/{id} - Delete a sneaker
@app.delete("/sneakers/{id}")
async def delete_sneaker(id: str):
    """Delete a sneaker by ID"""
    for i, sneaker in enumerate(sneakers):
        if sneaker.id == id:
            sneakers.pop(i)
            return {"message": "sneaker deleted successfully"}
    
    raise HTTPException(status_code=404, detail="sneaker not found")

def print_this():
    if sneakers:
        print(f"First sneaker ID: {sneakers[0].id}")
        print(f"Name: {sneakers[0].name}")

# Main entry point
if __name__ == "__main__":
    print_this()
    uvicorn.run(app, host="0.0.0.0", port=8080)