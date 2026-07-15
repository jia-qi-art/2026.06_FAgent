from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from run_quick_benchmark import score_answer, write_results


METRIC_NAMES = [
    "root_cause_accuracy",
    "evidence_coverage",
    "format_compliance",
    "action_coverage",
    "sensor_hallucination_rate",
    "rouge_l_f1",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score an existing quick benchmark run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        row.update(
            score_answer(
                {"input": row["prompt"], "output": row["reference"]},
                row["prediction"],
            )
        )
        grouped.setdefault(row["model"], []).append(row)

    metrics = []
    for label, model_rows in grouped.items():
        total_time = sum(float(row["latency_seconds"]) for row in model_rows)
        total_tokens = sum(int(row["generated_tokens"]) for row in model_rows)
        previous = next(
            item
            for item in json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            if item["model"] == label
        )
        summary = {
            name: sum(float(row[name]) for row in model_rows) / len(model_rows)
            for name in METRIC_NAMES
        }
        summary.update(
            {
                "model": label,
                "model_id": model_rows[0]["model_id"],
                "test_samples": len(model_rows),
                "avg_latency_seconds": total_time / len(model_rows),
                "tokens_per_second": total_tokens / total_time if total_time else 0.0,
                "peak_vram_gb": previous["peak_vram_gb"],
            }
        )
        metrics.append(summary)

    original_environment = (run_dir / "environment.json").read_text(encoding="utf-8")
    write_results(
        run_dir,
        environment["config"],
        split,
        metrics,
        rows,
        environment["train_losses"],
        environment["adapter_path"],
    )
    (run_dir / "environment.json").write_text(original_environment, encoding="utf-8")
    print(f"Re-scored {len(rows)} predictions in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
