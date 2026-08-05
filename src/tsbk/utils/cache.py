import hashlib


def cache_bust_key_material(cache_bust: str | None) -> bytes:
    """Return unambiguous cache-key material for an optional cache-bust value."""
    if not cache_bust:
        return b""
    return b"\x00cache_bust\x00" + cache_bust.encode()


def append_cache_bust(cache_key: str, cache_bust: str | None) -> str:
    """Append an opaque, filesystem-safe cache-bust digest to a cache key."""
    if not cache_bust:
        return cache_key
    digest = hashlib.sha256(cache_bust.encode()).hexdigest()[:16]
    return f"{cache_key}-{digest}"
