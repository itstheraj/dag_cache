"""Framework integrations. Each adapter is duck-typed and import-guarded:
the real frameworks are optional dependencies, and the adapters work with
any object exposing the expected attributes (which is also how they're
tested -- no API keys required).
"""
