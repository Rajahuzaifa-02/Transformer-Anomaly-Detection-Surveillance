from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import tempfile
import traceback
from fastapi.responses import StreamingResponse
from process import process_video_real_time  # Your video processing function

app = FastAPI()  # Initialize the FastAPI app here

# Initialize the OAuth2PasswordBearer for token verification
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# A mock of a valid user for demonstration purposes
users_db = {
    "user1": {"username": "user1", "password": "password123"},
}

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

class Token(BaseModel):
    access_token: str
    token_type: str

# Mock function for password verification (you could use hashing in real-world apps)
def verify_password(plain_password, hashed_password):
    return plain_password == hashed_password

# Mock user authentication
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create the access token (JWT)
    access_token_expires = timedelta(minutes=30)
    access_token = jwt.encode(
        {"sub": form_data.username, "exp": datetime.utcnow() + access_token_expires},
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to verify token and extract user
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_alert(prediction: float, frame_number: int):
    """Generate alerts based on your model's prediction score"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if prediction > 0.8:
        return {
            "score": float(prediction),
            "alert": f"🚨 CRITICAL ANOMALY (Score: {prediction:.2f}) at frame {frame_number}",
            "timestamp": timestamp,
            "severity": "critical"
        }
    elif prediction > 0.5:
        return {
            "score": float(prediction),
            "alert": f"⚠️ WARNING (Score: {prediction:.2f}) at frame {frame_number}",
            "timestamp": timestamp,
            "severity": "warning"
        }
    return {
        "score": float(prediction),
        "alert": None,
        "timestamp": timestamp
    }

# Video prediction endpoint with authentication
@app.post("/predict/")
async def predict_video(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    try:
        # Process video with the current user's authentication context
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        def generate_detection_data():
            frame_number = 0
            for line in process_video_real_time(tmp_path):
                if line.startswith("data:"):
                    # Extract prediction from your process function's output
                    prediction_str = line.split(":")[1].strip()
                    try:
                        prediction = float(prediction_str.split("[")[1].split("]")[0])
                        frame_number += 16  # Assuming 16-frame segments
                        
                        # Generate alert data
                        alert_data = generate_alert(prediction, frame_number)
                        yield f"data: {json.dumps(alert_data)}\n\n"
                    except (IndexError, ValueError):
                        continue


        # Return a streaming response with video processing
        return StreamingResponse(process_video_real_time(tmp_path), media_type="text/event-stream")

    except Exception as e:
        error_message = str(e)
        error_details = traceback.format_exc()
        print(f"Error processing video: {error_message}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error: " + error_message)
