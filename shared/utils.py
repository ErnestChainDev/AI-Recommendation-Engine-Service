from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict, secret: str, algorithm: str, expires_minutes: int) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret, algorithm=algorithm)

def decode_token(token: str, secret: str, algorithm: str) -> dict:
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as e:
        raise ValueError("Invalid token") from e
