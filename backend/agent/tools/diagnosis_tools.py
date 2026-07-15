from __future__ import annotations

from typing import Any

from backend import data_service as svc
from backend.rag import get_knowledge_store


class DiagnosisTools:
    """Domain tools replacing EquiMind's demo tools with Relation-EVGAT evidence."""

    def get_event_summary(self, dataset: str, event_id: int | None) -> dict[str, Any]:
        overview = svc.overview(dataset)
        event = svc.selected_event(dataset, event_id)
        return {
            "dataset": dataset,
            "event": event,
            "current_score": overview["current_score"],
            "threshold": overview["threshold"],
            "alert": overview["alert"],
            "metrics": overview["metrics"],
        }

    def rank_root_causes(self, dataset: str, event_id: int | None) -> dict[str, Any]:
        return svc.root_cause(dataset, event_id)

    def inspect_edge_degradation(self, dataset: str, event_id: int | None) -> dict[str, Any]:
        return svc.relation_graph(dataset, event_id)

    def inspect_sensor_window(self, dataset: str, event_id: int | None) -> dict[str, Any]:
        event = svc.selected_event(dataset, event_id)
        start = int(event.get("raw_start_time", event.get("start", 0)))
        end = int(event.get("raw_end_time", event.get("end", start + 900)))
        return svc.timeseries(dataset, max(0, start - 240), end + 240)

    def retrieve_maintenance_knowledge(self, query: str) -> dict[str, Any]:
        store = get_knowledge_store()
        hits = store.search(query)
        return {"query": query, "hits": hits, "status": store.status}

    def inspect_qwen_evidence(self) -> dict[str, Any]:
        """Expose the latest LoRA adapter as an optional evidence source.

        Loading the local Qwen model is intentionally not performed for every chat
        request; inference can be enabled later by a dedicated worker.
        """
        try:
            from backend.lora_finetune import list_saved_adapters

            adapters = list_saved_adapters()
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc), "source": "lora_qwen"}
        if not adapters:
            return {"status": "unavailable", "reason": "no saved LoRA adapter", "source": "lora_qwen"}
        latest = adapters[0]
        return {
            "status": "available",
            "source": "lora_qwen",
            "adapter": latest,
            "reason": "adapter metadata available; online inference requires the dedicated Qwen worker",
        }

    def generate_report(self, dataset: str, event_id: int | None) -> dict[str, Any]:
        return svc.report(dataset, event_id)

