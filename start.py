import uvicorn

def start():
    """Lance l'API avec Uvicorn"""
    uvicorn.run("projet_api.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    start()