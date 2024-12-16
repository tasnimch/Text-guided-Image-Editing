import uvicorn
from configuration.config import Config
from source.api.routes import app
from configuration.config import Config
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[Config.FRONT_PATH],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)