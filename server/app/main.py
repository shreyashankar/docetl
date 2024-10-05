from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.app.routes import pipeline

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your Next.js app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)


@app.get("/")
async def root():
    return {"message": "DocETL API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app.main:app", host="0.0.0.0", port=8000, reload=True)
