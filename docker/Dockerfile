# syntax=docker/dockerfile:1.7

ARG BILI_SEARCH_BASE_IMAGE=bili-search-base:ubuntu24.04-py3
FROM ${BILI_SEARCH_BASE_IMAGE} AS runtime-base

FROM runtime-base AS deps-builder

COPY pyproject.toml /tmp/pyproject.toml
COPY --from=sedb_src . /tmp/vendor/sedb
COPY --from=tclogger_src . /tmp/vendor/tclogger
COPY --from=webu_src . /tmp/vendor/webu
RUN python3 -c "import pathlib, tomllib; vendor_root = pathlib.Path('/tmp/vendor'); deps = tomllib.loads(pathlib.Path('/tmp/pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']; vendored = {name: (vendor_root / name / 'pyproject.toml').exists() for name in ('tclogger', 'webu')}; filtered = [dep for dep in deps if not ((dep.lower().startswith('tclogger') and vendored['tclogger']) or (dep.lower().startswith('webu') and vendored['webu']))]; pathlib.Path('/tmp/deps.txt').write_text('\n'.join(filtered) + ('\n' if filtered else ''), encoding='utf-8')"
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ -s /tmp/deps.txt ]; then python3 -m pip wheel --wheel-dir /tmp/wheels -r /tmp/deps.txt; else mkdir -p /tmp/wheels; fi

FROM runtime-base AS app-runtime

ENV PYTHONPATH=/app

WORKDIR /app

COPY --from=deps-builder /tmp/wheels /tmp/wheels
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if ls /tmp/wheels/* >/dev/null 2>&1; then python3 -m pip install --break-system-packages --no-cache-dir /tmp/wheels/*; fi \
    && rm -rf /tmp/wheels

COPY --from=sedb_src . /opt/vendor/sedb
COPY --from=tclogger_src . /opt/vendor/tclogger
COPY --from=webu_src . /opt/vendor/webu

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ -f /opt/vendor/tclogger/pyproject.toml ]; then python3 -m pip install --break-system-packages --no-deps -e /opt/vendor/tclogger; fi \
    && if [ -f /opt/vendor/webu/pyproject.toml ]; then python3 -m pip install --break-system-packages --no-deps -e /opt/vendor/webu; fi \
    && if [ -f /opt/vendor/sedb/pyproject.toml ]; then python3 -m pip install --break-system-packages --no-deps -e /opt/vendor/sedb; fi

COPY . /app
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --break-system-packages --no-deps -e /app

CMD ["python3", "-m", "service.container_supervisor"]