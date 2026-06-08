FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY thrombxai ./thrombxai
COPY conf ./conf
RUN pip install --no-cache-dir .

ENTRYPOINT ["thrombxai-weave"]
CMD ["experiment=_smoke"]
