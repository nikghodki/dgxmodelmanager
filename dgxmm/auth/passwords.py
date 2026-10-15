from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_ph = PasswordHasher()
MIN_LEN = 12
DUMMY_HASH = _ph.hash("dgxmm-timing-equaliser")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(pw_hash: str, password: str) -> bool:
    try:
        return _ph.verify(pw_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def validate_new_password(password: str) -> None:
    if len(password) < MIN_LEN:
        raise ValueError(f"password must be at least {MIN_LEN} characters")
