from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings


_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Invalid or missing token") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )

def verify_token(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> dict:

    if creds is None or not creds.credentials:
        raise _unauthorized("Missing Bearer token")

    token = creds.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],  # явно: отклоняет alg=none и чужие алгоритмы
            options={"require": ["exp"]},  # токен обязан иметь exp
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token expired")
    except jwt.InvalidSignatureError:
        raise _unauthorized("Invalid signature")
    except jwt.InvalidTokenError:
        # покрывает alg=none, неверный формат, отсутствие exp и т.п.
        raise _unauthorized("Invalid token")

    return payload

require_auth = Depends(verify_token)