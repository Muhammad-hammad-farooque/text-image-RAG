from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from core.security import create_access_token

router = APIRouter()

# Demo user store — replace with a real database in production
_DEMO_USERS: dict[str, str] = {
    "admin": "admin",
    "demo": "demo",
}


@router.post("/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    stored_password = _DEMO_USERS.get(form.username)
    if stored_password is None or stored_password != form.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
    }
