"""
PaddleOCR 工业文档文字提取模块。
参考论文：DB（Differentiable Binarization）检测 + SVTR_LCNet 识别，适配工业文本场景。
注意：Windows 上需 Tesseract 或修复 PyTorch DLL 后才能使用图片 OCR，DOCX 不受影响。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_OCR_INSTANCE: OCRService | None = None


@dataclass
class OCRResult:
    full_text: str = ""
    lines: list[dict[str, Any]] = field(default_factory=list)
    key_value_pairs: dict[str, str] = field(default_factory=dict)
    tables: list[list[list[str]]] = field(default_factory=list)


class OCRService:
    """PaddleOCR 封装，针对工业文档优化。"""

    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.45,
        det_db_unclip_ratio: float = 1.8,
        rec_batch_num: int = 6,
        show_log: bool = False,
    ):
        try:
            from paddleocr import PaddleOCR as _PaddleOCR
        except ImportError:
            raise ImportError("paddleocr 未安装，请执行: pip install paddlepaddle paddleocr")
        self._ocr = _PaddleOCR(
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            rec_batch_num=rec_batch_num,
            show_log=show_log,
        )

    def ocr(self, img: str | np.ndarray, cls: bool = True) -> list[dict[str, Any]]:
        raw = self._ocr.ocr(img, cls=cls)
        if not raw or not raw[0]:
            return []
        results: list[dict[str, Any]] = []
        for line in raw[0]:
            box, (text, confidence) = line
            results.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "box": [[int(x), int(y)] for x, y in box],
            })
        return results

    def extract_text(self, img: str | np.ndarray, cls: bool = True) -> str:
        items = self.ocr(img, cls=cls)
        return "\n".join(item["text"] for item in items)

    def extract_structured(self, img: str | np.ndarray) -> OCRResult:
        items = self.ocr(img, cls=True)
        full_text = "\n".join(item["text"] for item in items)
        kv_pairs = self._extract_key_values(items) if items else {}
        tables = self._detect_tables(items) if items else []
        return OCRResult(full_text=full_text, lines=items, key_value_pairs=kv_pairs, tables=tables)

    @staticmethod
    def _extract_key_values(items: list[dict[str, Any]]) -> dict[str, str]:
        import re
        kv: dict[str, str] = {}
        kv_patterns = [
            re.compile(r"([一-鿿\w]+)[\s]*[=：:]\s*([一-鿿\w.\-]+)"),
            re.compile(r"([一-鿿\w]+)[\s]+([\d.]+)\s*([一-鿿]+)"),
        ]
        for item in items:
            text = item["text"].strip()
            for pat in kv_patterns:
                m = pat.match(text)
                if m:
                    key = m.group(1).strip()
                    val = " ".join(g for g in m.groups()[1:] if g is not None).strip()
                    kv[key] = val
                    break
        return kv

    @staticmethod
    def _detect_tables(items: list[dict[str, Any]]) -> list[list[list[str]]]:
        if len(items) < 4:
            return []
        y_threshold = 15
        rows: list[list[dict]] = []
        sorted_items = sorted(items, key=lambda it: (it["box"][0][1], it["box"][0][0]))
        current_row: list[dict] = []
        current_y = None
        for item in sorted_items:
            y = item["box"][0][1]
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_row.append(item)
            else:
                if len(current_row) >= 2:
                    rows.append(current_row)
                current_row = [item]
            current_y = y
        if len(current_row) >= 2:
            rows.append(current_row)
        result: list[list[list[str]]] = []
        for row in rows:
            sorted_row = sorted(row, key=lambda it: it["box"][0][0])
            result.append([[cell["text"] for cell in sorted_row]])
        return result


def get_ocr() -> OCRService:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        logger.info("PaddleOCR 初始化中（DB + SVTR_LCNet）...")
        _OCR_INSTANCE = OCRService(lang="ch", use_angle_cls=True, use_gpu=False, show_log=False)
        logger.info("PaddleOCR 初始化完成")
    return _OCR_INSTANCE


def extract_text_from_image(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {"success": False, "text": "", "items": [], "error": f"文件不存在: {image_path}"}
    try:
        ocr = get_ocr()
        items = ocr.ocr(str(path))
        text = "\n".join(item["text"] for item in items)
        return {"success": True, "text": text, "items": items, "error": None}
    except Exception as exc:
        logger.exception("OCR 提取失败")
        return {"success": False, "text": "", "items": [], "error": str(exc)}


def extract_text_from_bytes(image_bytes: bytes) -> dict[str, Any]:
    import tempfile
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(image_bytes)
        tmp.close()
        return extract_text_from_image(tmp.name)
    except Exception as exc:
        logger.exception("OCR 字节流提取失败")
        return {"success": False, "text": "", "items": [], "error": str(exc)}
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)
