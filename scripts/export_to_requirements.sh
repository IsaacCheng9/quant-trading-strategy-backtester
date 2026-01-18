# Only lock the dependencies, not the dev dependencies.
rm uv.lock
uv sync
uv export --format requirements.txt > requirements.txt
# Go back to the full dev environment.
uv sync --dev
