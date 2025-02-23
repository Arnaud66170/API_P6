import uvicorn

def start():
    """Lance l'API avec Uvicorn"""
    uvicorn.run("API_P6.main:app", host = "0.0.0.0", port = 8000, reload = True)

if __name__ == "__main__":
    start()