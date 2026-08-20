import json
import os
import sys
import urllib.error
from collections.abc import Callable
from typing import Any


AUTHENTICATION_CODES = {
    "invalidapikey",
    "invalidaccesskey",
    "authenticationfailed",
    "unauthorized",
}
PERMISSION_CODES = {
    "accessdenied",
    "forbidden",
    "modelaccessdenied",
    "permissiondenied",
}
RETRYABLE_STATUS_CODES = {408, 425, 429}


def _redact(value: str) -> str:
    redacted = value
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_QWEN_API_KEY",
        "DASHSCOPE_ASR_API_KEY",
    ):
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


class DashScopeAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        service: str,
        category: str,
        retryable: bool,
        http_status: int | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
        retry_after: float | None = None,
        original_type: str | None = None,
    ) -> None:
        super().__init__(_redact(message))
        self.service = service
        self.category = category
        self.retryable = retryable
        self.http_status = http_status
        self.provider_code = provider_code
        self.provider_message = _redact(provider_message or "") or None
        self.retry_after = retry_after
        self.original_type = original_type or type(self).__name__
        self.attempt = 0

    def event(self, attempt: int, max_attempts: int, context: dict | None = None) -> dict:
        return {
            "event": "dashscope_error",
            "service": self.service,
            "category": self.category,
            "http_status": self.http_status,
            "provider_code": self.provider_code,
            "message": str(self),
            "provider_message": self.provider_message,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after,
            "error_type": self.original_type,
            "attempt": attempt,
            "max_attempts": max_attempts,
            **(context or {}),
        }


def _response_details(body: str) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None, body.strip()[:300] or None
    if not isinstance(payload, dict):
        return None, body.strip()[:300] or None
    code = payload.get("code")
    message = payload.get("message")
    return (
        str(code) if code is not None else None,
        str(message)[:300] if message is not None else None,
    )


def _retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return min(60.0, max(0.0, seconds))


def classify_exception(exc: Exception, service: str) -> DashScopeAPIError:
    if isinstance(exc, DashScopeAPIError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        provider_code, provider_message = _response_details(body)
        code_key = (provider_code or "").replace("_", "").replace("-", "").lower()
        if exc.code == 401 or code_key in AUTHENTICATION_CODES:
            category = "authentication"
            retryable = False
            message = "DashScope API Key 无效或已过期"
        elif exc.code == 403 or code_key in PERMISSION_CODES:
            retryable = False
            if service in {"asr-upload", "asr-result"}:
                category = "resource_access"
                message = "DashScope ASR 临时资源访问被拒绝"
            else:
                category = "permission"
                message = "DashScope API Key 没有访问当前模型或资源的权限"
        elif exc.code == 429:
            category = "rate_limit"
            retryable = True
            message = "DashScope 请求受到限流"
        elif exc.code in RETRYABLE_STATUS_CODES:
            category = "timeout"
            retryable = True
            message = "DashScope 请求暂时未能完成"
        elif 500 <= exc.code <= 599:
            category = "service_unavailable"
            retryable = True
            message = "DashScope 服务暂时不可用"
        else:
            category = "request"
            retryable = False
            message = "DashScope 请求被拒绝"
        return DashScopeAPIError(
            f"{message}（HTTP {exc.code}）",
            service=service,
            category=category,
            retryable=retryable,
            http_status=exc.code,
            provider_code=provider_code,
            provider_message=provider_message,
            retry_after=_retry_after(exc.headers),
            original_type=type(exc).__name__,
        )
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError)):
        return DashScopeAPIError(
            "DashScope 网络连接失败",
            service=service,
            category="network",
            retryable=True,
            provider_message=str(exc),
            original_type=type(exc).__name__,
        )
    if isinstance(exc, (json.JSONDecodeError, TypeError, ValueError)):
        return DashScopeAPIError(
            "DashScope 返回了无法解析的响应",
            service=service,
            category="invalid_response",
            retryable=True,
            provider_message=str(exc),
            original_type=type(exc).__name__,
        )
    return DashScopeAPIError(
        "DashScope 调用失败",
        service=service,
        category="unknown",
        retryable=False,
        provider_message=str(exc),
        original_type=type(exc).__name__,
    )


def retry_delay(error: DashScopeAPIError, attempt: int) -> float:
    if error.retry_after is not None:
        return error.retry_after
    return min(30.0, float(2**attempt))


def run_with_retries(
    operation: Callable[[], Any],
    *,
    service: str,
    max_attempts: int = 3,
    sleep: Callable[[float], None],
    context: dict | None = None,
) -> Any:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except (
            DashScopeAPIError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            error = classify_exception(exc, service)
            error.attempt = attempt
            print(
                json.dumps(
                    error.event(attempt, max_attempts, context),
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            if not error.retryable or attempt >= max_attempts:
                raise error from exc
            sleep(retry_delay(error, attempt))
    raise AssertionError("unreachable")
