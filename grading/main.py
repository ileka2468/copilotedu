from fastapi import FastAPI
import os

app = FastAPI(
    title="Teacher Copilot Grading Service",
    description="AI-powered grading service for Teacher Copilot MVP",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Teacher Copilot Grading Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)