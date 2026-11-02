"""
describe_exception(e): str(e), with two fixes for real diagnosability gaps
confirmed live (2026-08-19), not hypothetical - three real requests failed
within seconds of each other and every one logged as content-free noise:

1. A bare TimeoutError() and httpx's own timeout exceptions (ReadTimeout,
   ConnectTimeout, ...) all stringify to '' - so `f'{step} failed: {e}'`
   came out as "{step} failed: " with the actual reason silently dropped.
2. An asyncio TaskGroup's own ExceptionGroup/BaseExceptionGroup DOES have a
   non-empty str() ("unhandled errors in a TaskGroup (N sub-exception(s))"),
   which passes right through case 1's check while still hiding the actual
   sub-exception(s) that caused it - looks informative, isn't.
"""


def describe_exception(e: BaseException) -> str:
    # ExceptionGroup/BaseExceptionGroup (Python 3.11+, e.g. from an
    # asyncio.TaskGroup) - unwrap to the real sub-exception(s) instead of
    # its own generic wrapper text. Recurses in case a sub-exception is
    # itself a group, or has its own empty str().
    sub_exceptions = getattr(e, 'exceptions', None)
    if sub_exceptions:
        return '; '.join(describe_exception(sub) for sub in sub_exceptions)

    text = str(e)
    if text:
        return text
    return f"{type(e).__name__} (empty message - likely a timeout)"
