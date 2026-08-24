# Laboratório de serving com vLLM

Em uma máquina Linux com GPU NVIDIA:

```bash
uv sync --extra serving --locked
uv run vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --generation-config vllm \
  --enable-per-request-metrics \
  --enable-server-load-tracking
```

Em outro terminal:

```bash
uv run python modulo-11-inferencia/benchmark_vllm.py \
  --requisicoes 200 --concorrencia 1
uv run python modulo-11-inferencia/benchmark_vllm.py \
  --requisicoes 200 --concorrencia 8
uv run python modulo-11-inferencia/benchmark_vllm.py \
  --requisicoes 200 --concorrencia 32
curl http://127.0.0.1:8000/metrics > metricas-vllm.prom
```

Compare sucesso, latência p50/p95, tokens/s e requisições/s. O endpoint `/metrics`
expõe TTFT, latência entre tokens, uso do KV cache, fila e prefix-cache hits. Não exponha
o servidor diretamente: a chave do vLLM não protege todos os endpoints; use proxy,
TLS, autenticação e rede privada em produção.

O laboratório só muda para **validado** quando os JSONs de carga e o snapshot das
métricas forem preservados com hardware, versão do vLLM, revisão do modelo e custo/hora.
