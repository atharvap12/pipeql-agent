# Image that runs the LLM-written Pandas code for the ETL agent (see utils/sandbox.py).
# Everything the code may import is installed here at build time, because the
# container runs with no network and cannot install anything later.
FROM python:3.13-slim

RUN pip install --no-cache-dir \
        pandas==3.1.0 \
        numpy==2.5.4 \
        pyarrow==22.0.0

# Unprivileged user. utils/sandbox.py overrides the uid/gid with the host user's
# so the output file it writes is owned by you, but it never runs as root.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /tmp
