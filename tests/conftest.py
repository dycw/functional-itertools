from __future__ import annotations

from os import getenv

from hypothesis import settings


settings.register_profile("dev", deadline=None, max_examples=10, print_blob=True)
settings.register_profile("default", deadline=None, max_examples=100, print_blob=True)
settings.register_profile("ci", deadline=None, max_examples=1000, print_blob=True)


settings.load_profile(getenv("HYPOTHESIS_PROFILE", "default"))
