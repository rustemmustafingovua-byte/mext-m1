# Grafana dashboard for MEXT-M1

MEXT exposes a Prometheus endpoint on the dashboard port:

```bash
curl -s localhost:8765/metrics | head
# HELP mext_pool_resident_bytes Real RSS of the pool (arena pages + mappings)
# TYPE mext_pool_resident_bytes gauge
mext_pool_resident_bytes 134742016
```

## Import (UI)

1. Start MEXT: `mext start --workspace ~/code/project --ram-limit 2G`
2. Make `/metrics` reachable from Grafana. If Grafana runs in Docker on macOS:
   use `http://host.docker.internal:8765/metrics` as the scrape URL.
3. Grafana → **Dashboards → New → Import → Upload JSON** and pick
   `mext-dashboard.json`. Select your Prometheus data source.

## Import (provisioning)

```yaml
# /etc/grafana/provisioning/datasources/mext.yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
```

```yaml
# /etc/grafana/provisioning/dashboards/mext.yaml
apiVersion: 1
providers:
  - name: mext
    folder: MEXT
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

Copy `mext-dashboard.json` into `/var/lib/grafana/dashboards/` and restart
Grafana.

## Scrape config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: mext
    scrape_interval: 2s
    static_configs:
      - targets: ["127.0.0.1:8765"]
```

## Panels

| Panel | Metric | What it answers |
|---|---|---|
| RAM pool physical vs virtual | `mext_pool_resident_bytes`, `mext_pool_logical_bytes`, `mext_pool_budget_bytes` | сколько данных отдаёт пул и сколько это стоит физически |
| Prediction hit rate | `mext_predictor_hit_rate` | точность предсказателя файлов (цель > 60%) |
| Memory pressure | `mext_pressure_info{level}` | зелёная/жёлтая/красная зона macOS |
| SSD → RAM | `rate(mext_pool_ssd_read_bytes_total[1m])`, `rate(mext_neural_streamed_bytes_total[1m])` | скорость подкачки с NVMe |
| Neural hot/cold/resident | `mext_neural_*_bytes` | окно резидентности весов |
| Prefetch accuracy | `mext_neural_prefetch_hit_rate`, `mext_neural_expert_hit_rate` | точность префетча слоёв и экспертов |
| Throughput | `mext_neural_tokens_per_second` | токены/с |
| RAG | `mext_rag_chunks`, `mext_rag_clusters`, `mext_rag_latency_ms`, `mext_rag_prefetch_hit_rate` | состояние индекса и префетча запросов |
| Bytes returned | `mext_pool_released_bytes_total` | сколько реально вернули macOS через `madvise` |
| Latency p95 | `mext_pool_warm_latency_ms_bucket`, `mext_rag_query_latency_ms_bucket` | p95 прогрева файла и RAG-запроса (histogram_quantile) |
| Evictions / RSS | `mext_pool_evictions_total`, `mext_rss_bytes` | давление на бюджет и вес демона |

Полный список метрик: `curl -s localhost:8765/metrics | grep '^# HELP'`.
