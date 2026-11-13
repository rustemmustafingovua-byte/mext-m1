//! Validates the shipped Grafana dashboard so a broken JSON never reaches docs.

use serde_json::Value;

const DASHBOARD: &str = include_str!("../../docs/grafana/mext-dashboard.json");

#[test]
fn dashboard_json_is_valid_and_wired_to_prometheus() {
    let dashboard: Value = serde_json::from_str(DASHBOARD).expect("dashboard JSON parses");

    assert_eq!(dashboard["uid"], "mext-m1");
    assert!(dashboard["title"].as_str().unwrap().contains("MEXT"));

    let panels = dashboard["panels"].as_array().expect("panels array");
    assert!(
        panels.len() >= 10,
        "expected a full dashboard, got {}",
        panels.len()
    );

    let templating = dashboard["templating"]["list"]
        .as_array()
        .expect("templating");
    assert_eq!(templating[0]["name"], "DS_PROMETHEUS");

    for panel in panels {
        let title = panel["title"].as_str().unwrap_or("<untitled>");
        let targets = panel["targets"].as_array().unwrap_or_else(|| {
            panic!("panel `{title}` has no targets");
        });
        assert!(!targets.is_empty(), "panel `{title}` has no queries");
        for target in targets {
            let expr = target["expr"].as_str().unwrap_or_else(|| {
                panic!("panel `{title}` target without expr");
            });
            assert!(expr.contains("mext_"), "panel `{title}` queries `{expr}`");
        }
        assert_eq!(
            panel["datasource"]["type"], "prometheus",
            "panel `{title}` must use the Prometheus data source"
        );
    }

    // The dashboard must cover all three product layers plus macOS health.
    let text = DASHBOARD;
    for needle in [
        "mext_pool_logical_bytes",
        "mext_predictor_hit_rate",
        "mext_pressure_info",
        "mext_neural_expert_hit_rate",
        "mext_rag_prefetch_hit_rate",
    ] {
        assert!(text.contains(needle), "dashboard misses {needle}");
    }
}
