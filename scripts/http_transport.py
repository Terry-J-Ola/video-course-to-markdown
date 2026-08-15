from __future__ import annotations

import http.client
import io
import ssl
import urllib.error
import urllib.parse
import urllib.request

import certifi


REDIRECT_STATUSES = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 5


class BufferedResponse:
    def __init__(self, body: bytes, status: int, reason: str, headers):
        self._body = body
        self.status = status
        self.reason = reason
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._body


def verified_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _origin(parts: urllib.parse.SplitResult) -> tuple[str, str, int]:
    return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port or 443


def _without_header(headers: dict[str, str], name: str) -> dict[str, str]:
    lowered = name.lower()
    return {key: value for key, value in headers.items() if key.lower() != lowered}


def open_url(
    request_or_url: urllib.request.Request | str,
    timeout: float | None = None,
) -> BufferedResponse:
    if isinstance(request_or_url, urllib.request.Request):
        current_url = request_or_url.full_url
        method = request_or_url.get_method()
        data = request_or_url.data
        headers = dict(request_or_url.header_items())
    else:
        current_url = str(request_or_url)
        method = "GET"
        data = None
        headers = {}

    for redirect_count in range(MAX_REDIRECTS + 1):
        parts = urllib.parse.urlsplit(current_url)
        if parts.scheme.lower() != "https" or not parts.hostname:
            raise ValueError(f"only HTTPS URLs are supported: {current_url}")
        target = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
        connection = http.client.HTTPSConnection(
            parts.hostname,
            parts.port or 443,
            timeout=timeout,
            context=verified_ssl_context(),
        )
        try:
            connection.request(method, target, body=data, headers=headers)
            response = connection.getresponse()
            body = response.read()
            status = response.status
            reason = response.reason
            response_headers = response.headers
        finally:
            connection.close()

        if status in REDIRECT_STATUSES:
            location = response_headers.get("Location")
            if not location:
                raise urllib.error.HTTPError(
                    current_url,
                    status,
                    "redirect response is missing Location",
                    response_headers,
                    io.BytesIO(body),
                )
            if redirect_count >= MAX_REDIRECTS:
                raise urllib.error.HTTPError(
                    current_url,
                    status,
                    "too many redirects",
                    response_headers,
                    io.BytesIO(body),
                )
            next_url = urllib.parse.urljoin(current_url, location)
            next_parts = urllib.parse.urlsplit(next_url)
            if next_parts.scheme.lower() != "https":
                raise ValueError(f"refusing non-HTTPS redirect: {next_url}")
            if _origin(parts) != _origin(next_parts):
                headers = _without_header(headers, "Authorization")
            if status == 303 or (status in {301, 302} and method.upper() == "POST"):
                method = "GET"
                data = None
                headers = _without_header(headers, "Content-Type")
                headers = _without_header(headers, "Content-Length")
            current_url = next_url
            continue

        if status >= 400:
            raise urllib.error.HTTPError(
                current_url,
                status,
                reason,
                response_headers,
                io.BytesIO(body),
            )
        return BufferedResponse(body, status, reason, response_headers)

    raise RuntimeError("unreachable redirect state")
