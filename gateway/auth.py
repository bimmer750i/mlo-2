from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, Request, status

from config import settings


def _unauthorized(detail: str = "Invalid or missing token") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_token(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise _unauthorized("Missing Bearer token")

    token = auth[len("Bearer "):].strip()
    if not token:
        raise _unauthorized("Missing Bearer token")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],  # отклоняет alg=none и чужие алгоритмы
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token expired")
    except jwt.InvalidSignatureError:
        raise _unauthorized("Invalid signature")
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid token")

    return payload


require_auth = Depends(verify_token)