from os import getenv

from hypothesis import settings


settings.register_profile("dev", max_examples=10, print_blob=True)
settings.register_profile("default", max_examples=100, print_blob=True)
settings.register_profile("ci", max_examples=1000, print_blob=True)


settings.load_profile(getenv("HYPOTHESIS_PROFILE", "default"))
