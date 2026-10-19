"""
🐉 Dragon Fruit Stem Disease — Multi-Agent System
=================================================
Run with:   streamlit run app.py

Agents (coordinated by an Orchestrator through a shared Blackboard):

  1. ImageQualityAgent   - checks blur / exposure / resolution / plant content
  2. ClassifierAgent     - ResNet50 classification; optional Test-Time Augmentation
  3. SeverityAgent       - HSV lesion segmentation -> % area -> severity band
  4. TreatmentAgent      - pesticide / dose / schedule lookup + field practices
  5. SafetyReviewAgent   - critic: checks consistency, gates risky advice
  6. AdvisorAgent        - writes the final report (Gemini LLM if API key is set,
                           otherwise a deterministic template) and answers follow-ups

Agentic behaviours: planning, shared memory, self-reflection (re-runs the
classifier with TTA when it is unsure), a critic/gating step, tool use and a
full execution trace.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# 0. DOMAIN KNOWLEDGE (carried over from the original notebook)
# ──────────────────────────────────────────────────────────────────────────
CLASSES = ["Anthracnose", "Soft_Rot", "Gray_Blight", "Brown_Stem_Spot", "Stem_Canker"]
IMG_SIZE = (224, 224)

# Confidence thresholds used by the agents
CONF_REFLECT = 75.0   # below this the Orchestrator asks for a second opinion (TTA)
CONF_OK = 70.0        # >= this: treatment "recommended"
CONF_MIN = 50.0       # <  this: treatment "on hold"

DISEASE_INFO = {
    "Anthracnose": "Fungal disease caused by Colletotrichum species -- dark sunken lesions on stems/fruits.",
    "Soft_Rot": "Bacterial/fungal infection -- soft, mushy, water-soaked areas, often foul-smelling.",
    "Gray_Blight": "Fungal disease -- grayish lesions with yellow halos, spreads fast in humidity.",
    "Brown_Stem_Spot": "Fungal disease -- circular brown spots on stems, worsens in wet seasons.",
    "Stem_Canker": "Caused by Neoscytalidium dimidiatum -- sunken cankers that can girdle/kill cladodes.",
}

PESTICIDE_DB = {
    "Anthracnose": {
        "Mild": {"product": "Neem Oil (Azadirachtin 1500 ppm)", "dose": "5 ml/L water (0.5%)", "dose_pct_of_label": 25, "frequency": "Every 7 days, 2-3 sprays", "type": "Organic"},
        "Moderate": {"product": "Copper Oxychloride 50% WP", "dose": "2 g/L water (0.2%)", "dose_pct_of_label": 60, "frequency": "Every 7-10 days, 3 sprays", "type": "Chemical (contact)"},
        "Severe": {"product": "Mancozeb 75% WP", "dose": "2.5 g/L water (0.25%)", "dose_pct_of_label": 100, "frequency": "Every 5-7 days until controlled", "type": "Chemical (protective)"},
    },
    "Soft_Rot": {
        "Mild": {"product": "Bordeaux Mixture 1%", "dose": "10 g/L water (1%)", "dose_pct_of_label": 30, "frequency": "Every 10 days", "type": "Mineral"},
        "Moderate": {"product": "Metalaxyl-M 8% + Mancozeb 64% WP", "dose": "2 g/L water (0.2%)", "dose_pct_of_label": 65, "frequency": "Every 7 days, 2-3 sprays", "type": "Chemical (systemic)"},
        "Severe": {"product": "Copper Hydroxide 77% WP", "dose": "3 g/L water (0.3%)", "dose_pct_of_label": 100, "frequency": "Every 5 days + soil drench + fix drainage", "type": "Chemical (systemic)"},
    },
    "Gray_Blight": {
        "Mild": {"product": "Mancozeb 75% WP", "dose": "2 g/L water (0.2%)", "dose_pct_of_label": 40, "frequency": "Weekly", "type": "Chemical (protective)"},
        "Moderate": {"product": "Carbendazim 50% WP", "dose": "1 g/L water (0.1%)", "dose_pct_of_label": 65, "frequency": "Every 7 days", "type": "Chemical (systemic)"},
        "Severe": {"product": "Carbendazim 12% + Mancozeb 63% WP", "dose": "2 g/L water (0.2%)", "dose_pct_of_label": 100, "frequency": "Every 5 days", "type": "Chemical (systemic+contact)"},
    },
    "Brown_Stem_Spot": {
        "Mild": {"product": "Copper Oxychloride 50% WP", "dose": "3 g/L water (0.3%)", "dose_pct_of_label": 35, "frequency": "Every 10 days", "type": "Chemical (contact)"},
        "Moderate": {"product": "Chlorothalonil 75% WP", "dose": "2 ml/L water (0.2%)", "dose_pct_of_label": 60, "frequency": "Every 7 days", "type": "Chemical (protective)"},
        "Severe": {"product": "Chlorothalonil 75% WP", "dose": "2.5 ml/L water (0.25%)", "dose_pct_of_label": 100, "frequency": "Every 5-7 days + check for pest vectors", "type": "Chemical (protective)"},
    },
    "Stem_Canker": {
        "Mild": {"product": "Bordeaux Paste", "dose": "Apply as paste on canker (10% w/w)", "dose_pct_of_label": 30, "frequency": "Single application, monitor weekly", "type": "Mineral"},
        "Moderate": {"product": "Thiophanate-Methyl 70% WP", "dose": "1 g/L water (0.1%)", "dose_pct_of_label": 65, "frequency": "Every 7 days", "type": "Chemical (systemic)"},
        "Severe": {"product": "Thiophanate-Methyl 70% WP", "dose": "1.5 g/L water (0.15%)", "dose_pct_of_label": 100, "frequency": "Every 5 days + sterilize tools with 70% alcohol", "type": "Chemical (systemic)"},
    },
}

# General, non-chemical good practice (generic guidance, not a prescription)
FIELD_PRACTICES = {
    "Anthracnose": ["Remove and destroy badly infected cladode parts.", "Avoid overhead irrigation; keep foliage dry."],
    "Soft_Rot": ["Improve drainage around the base.", "Avoid wounding the stems; cut away mushy tissue with a clean tool."],
    "Gray_Blight": ["Improve airflow by pruning crowded stems.", "Reduce humidity around the canopy."],
    "Brown_Stem_Spot": ["Remove heavily spotted tissue.", "Avoid overhead irrigation, especially in wet seasons."],
    "Stem_Canker": ["Prune cankered tissue back to healthy stem.", "Sterilize pruning tools (70% alcohol) between plants."],
}

BAND_ICON = {"Healthy": "🟢", "Mild": "🔵", "Moderate": "🟡", "Severe": "🔴"}
BAND_COLOR = {"Healthy": "green", "Mild": "blue", "Moderate": "orange", "Severe": "red"}


# ──────────────────────────────────────────────────────────────────────────
# 1. SHARED TOOLS (pure functions the agents call)
# ──────────────────────────────────────────────────────────────────────────
def preprocess(rgb: np.ndarray, key: str) -> np.ndarray:
    """Re-implementation of the Keras preprocess_input functions used in training."""
    x = rgb.astype("float32")
    if key == "mobilenet":          # [-1, 1]
        return x / 127.5 - 1.0
    if key == "resnet":             # caffe style: RGB->BGR, subtract ImageNet mean
        x = x[..., ::-1]
        return x - np.array([103.939, 116.779, 123.68], dtype="float32")
    return x                        # efficientnet: rescaling is inside the model


def tta_variants(rgb: np.ndarray) -> List[np.ndarray]:
    """Original + horizontal flip + 90% centre zoom (same as the notebook)."""
    h, w = rgb.shape[:2]
    flipped = cv2.flip(rgb, 1)
    ch, cw = int(h * 0.9), int(w * 0.9)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    zoomed = cv2.resize(rgb[y0:y0 + ch, x0:x0 + cw], (w, h))
    return [rgb, flipped, zoomed]


def estimate_severity(rgb: np.ndarray):
    """HSV lesion segmentation (same colour ranges as the notebook)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lesion = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([30, 255, 200]))
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    plant = cv2.bitwise_or(green, lesion)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion = cv2.morphologyEx(cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    plant = cv2.morphologyEx(plant, cv2.MORPH_OPEN, k)
    l_px, p_px = cv2.countNonZero(lesion), cv2.countNonZero(plant)
    pct = round(l_px / p_px * 100, 2) if p_px > 0 else 0.0
    return pct, lesion


def severity_band(pct: float) -> str:
    # NOTE: the notebook's version had gaps (e.g. 0.5% and 15.5% fell through to
    # "Severe"). These thresholds are contiguous.
    if pct < 1:
        return "Healthy"
    if pct <= 15:
        return "Mild"
    if pct <= 40:
        return "Moderate"
    return "Severe"


def lesion_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    m = mask > 0
    out[m] = (0.45 * out[m] + 0.55 * red[m]).astype("uint8")
    return out


# ──────────────────────────────────────────────────────────────────────────
# 2. MODEL REGISTRY (finds + lazily loads the trained .keras files)
# ──────────────────────────────────────────────────────────────────────────
MODEL_NAME = "ResNet50"
BEST_MODEL_FILENAME = "dragon_fruit_best_model (1).keras"


@st.cache_resource(show_spinner="Loading model…")
def load_keras_model(path: str, mtime: float):
    import tensorflow as tf  # imported lazily so the UI starts fast
    return tf.keras.models.load_model(path, compile=False)


class ModelRegistry:
    def __init__(self, models_dir: str):
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir))
        self.paths: Dict[str, str] = {}
        self.metrics: Optional[dict] = None
        self._discover()

    def _discover(self):
        candidates = [
            os.path.join(self.models_dir, "dragon_fruit_resnet50_model.keras"),
            *sorted(glob.glob(os.path.join(self.models_dir, "*resnet50*.keras"))),
            *sorted(glob.glob(os.path.join(self.models_dir, "*resnet50*.h5"))),
            os.path.join(self.models_dir, BEST_MODEL_FILENAME),
            os.path.join(os.path.dirname(__file__), BEST_MODEL_FILENAME),
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path:
            self.paths[MODEL_NAME] = path
        mj = os.path.join(self.models_dir, "model_comparison.json")
        if os.path.exists(mj):
            try:
                with open(mj) as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None

    @property
    def available(self) -> List[str]:
        return list(self.paths)

    def get(self, name: str):
        p = self.paths[name]
        return load_keras_model(p, os.path.getmtime(p))


# ──────────────────────────────────────────────────────────────────────────
# 3. BLACKBOARD (shared memory) + AGENT BASE CLASS
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TraceEvent:
    agent: str
    icon: str
    status: str          # ok | warn | error | info
    message: str
    seconds: float = 0.0


class Blackboard:
    """Shared memory every agent reads from / writes to, plus the execution trace."""

    def __init__(self, image_rgb: np.ndarray):
        self.d: Dict[str, Any] = {"image": image_rgb, "flags": []}
        self.trace: List[TraceEvent] = []
        self.listener: Optional[Callable[[TraceEvent], None]] = None
        self.fatal: Optional[str] = None

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def get(self, k, default=None):
        return self.d.get(k, default)

    def log(self, agent: str, icon: str, status: str, message: str, seconds: float = 0.0):
        ev = TraceEvent(agent, icon, status, message, seconds)
        self.trace.append(ev)
        if self.listener:
            self.listener(ev)

    def flag(self, level: str, text: str):
        self.d["flags"].append({"level": level, "text": text})

    def case_summary(self) -> dict:
        """JSON-safe snapshot (no arrays) – used for the LLM prompt."""
        g = self.get
        return {
            "disease": g("disease"), "confidence_pct": g("confidence"),
            "model_votes": g("votes"), "agreement": g("agreement"),
            "used_tta": g("used_tta"), "severity_pct": g("severity_pct"),
            "severity_band": g("band"), "treatment_status": g("treatment_status"),
            "treatment": g("treatment"), "field_practices": g("practices"),
            "image_quality": g("quality"), "review_flags": g("flags"),
            "disease_info": DISEASE_INFO.get(g("disease") or "", ""),
        }


class Agent:
    name = "Agent"
    icon = "🤖"
    role = ""

    def run(self, bb: Blackboard) -> str:  # returns a one-line summary
        raise NotImplementedError

    def __call__(self, bb: Blackboard) -> bool:
        t0 = time.perf_counter()
        try:
            msg = self.run(bb)
            bb.log(self.name, self.icon, "ok", msg, time.perf_counter() - t0)
            return True
        except Exception as e:  # noqa: BLE001
            bb.log(self.name, self.icon, "error", f"{type(e).__name__}: {e}", time.perf_counter() - t0)
            return False


# ──────────────────────────────────────────────────────────────────────────
# 4. THE AGENTS
# ──────────────────────────────────────────────────────────────────────────
class ImageQualityAgent(Agent):
    name, icon = "Image Quality Agent", "🔍"
    role = "Checks the photo is usable before spending compute on it."

    def run(self, bb):
        rgb = bb["image"]
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (min(w, 512), int(h * min(w, 512) / w)))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(gray.mean())
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        plant = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])),
            cv2.inRange(hsv, np.array([10, 40, 40]), np.array([30, 255, 200])))
        plant_ratio = float(cv2.countNonZero(plant)) / plant.size

        issues = []
        if min(h, w) < 150:
            issues.append(f"low resolution ({w}x{h})")
        if blur < 60:
            issues.append("image looks blurry")
        if bright < 50:
            issues.append("image is too dark")
        elif bright > 210:
            issues.append("image is over-exposed")
        if plant_ratio < 0.05:
            issues.append("very little green/brown plant tissue visible - is this a dragon fruit stem?")

        bb["quality"] = {"sharpness": round(blur, 1), "brightness": round(bright, 1),
                         "plant_ratio_pct": round(plant_ratio * 100, 1), "issues": issues}
        for i in issues:
            bb.flag("warn", f"Photo quality: {i}.")
        return "Photo OK." if not issues else f"{len(issues)} issue(s): " + "; ".join(issues)


class ClassifierAgent(Agent):
    name, icon = "Classifier Agent", "🧠"
    role = "Classifies the image with ResNet50."

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def run(self, bb):
        if not self.registry.available:
            raise RuntimeError(f"No trained models found in '{self.registry.models_dir}'.")
        use_tta = bool(bb.get("tta", False))
        resized = cv2.resize(bb["image"], IMG_SIZE)
        variants = tta_variants(resized) if use_tta else [resized]

        model = self.registry.get(MODEL_NAME)
        batch = np.stack([preprocess(v, "resnet") for v in variants])
        probs = model.predict(batch, verbose=0).mean(axis=0)
        idx = int(np.argmax(probs))
        votes = [{"model": MODEL_NAME, "prediction": CLASSES[idx],
                  "confidence": round(float(probs[idx]) * 100, 1), "agrees": True}]
        bb["probs"] = {CLASSES[i]: float(probs[i]) * 100 for i in range(len(CLASSES))}
        bb["class_idx"] = idx
        bb["disease"] = CLASSES[idx]
        bb["confidence"] = round(float(probs[idx]) * 100, 1)
        bb["votes"] = votes
        bb["agreement"] = sum(v["agrees"] for v in votes) / len(votes)
        bb["used_tta"] = use_tta
        return f"{bb['disease']} @ {bb['confidence']}% | ResNet50 | TTA={'on' if use_tta else 'off'}"


class SeverityAgent(Agent):
    name, icon = "Severity Agent", "📏"
    role = "Measures how much of the plant is affected."

    def run(self, bb):
        resized = cv2.resize(bb["image"], IMG_SIZE)
        pct, mask = estimate_severity(resized)
        bb["resized"] = resized
        bb["severity_pct"] = pct
        bb["lesion_mask"] = mask
        bb["band"] = severity_band(pct)
        return f"{pct}% of plant area affected -> {bb['band']}"


class TreatmentAgent(Agent):
    name, icon = "Treatment Agent", "💊"
    role = "Looks up product, dose and schedule for this disease + severity."

    def run(self, bb):
        disease, band = bb["disease"], bb["band"]
        bb["practices"] = FIELD_PRACTICES.get(disease, [])
        if band == "Healthy":
            bb["treatment"] = None
            return "No lesions detected -> no pesticide needed."
        rec = PESTICIDE_DB.get(disease, {}).get(band)
        bb["treatment"] = rec
        return f"{rec['product']} - {rec['dose']}" if rec else "No database entry; refer to agronomist."


class SafetyReviewAgent(Agent):
    name, icon = "Safety Review Agent", "🛡️"
    role = "Critic: cross-checks the other agents and gates risky advice."

    def run(self, bb):
        conf, agree = bb["confidence"], bb["agreement"]
        pct, band, disease = bb["severity_pct"], bb["band"], bb["disease"]
        n0 = len(bb["flags"])

        # Decide how far the treatment advice can be trusted
        if band == "Healthy":
            status = "none"
        elif conf < CONF_MIN:
            status = "on_hold"
        elif conf < CONF_OK or agree < 1.0:
            status = "provisional"
        else:
            status = "recommended"
        bb["treatment_status"] = status

        if conf < CONF_MIN:
            bb.flag("critical", f"Diagnosis is uncertain ({conf}% confidence). Retake a sharper, well-lit close-up "
                                "and get it confirmed before spraying anything.")
        elif conf < CONF_OK:
            bb.flag("warn", f"Moderate confidence ({conf}%). Treat the recommendation as provisional.")
        if agree < 1.0:
            names = ", ".join(v["model"] for v in bb["votes"] if not v["agrees"])
            bb.flag("warn", f"Specialist models disagree ({names} voted differently).")
        if band == "Healthy" and conf >= 80:
            bb.flag("warn", f"Classifier says {disease} ({conf}%) but colour analysis found almost no lesion area. "
                            "Inspect the plant by eye - early or off-colour lesions can be missed.")
        elif band == "Healthy":
            bb.flag("info", "No lesions detected and classifier is unsure - the plant is likely healthy.")
        if band != "Healthy" and conf < 80 and pct < 3:
            bb.flag("info", "Very small lesion area with a low-confidence diagnosis - monitor and re-photograph in a few days.")
        if band == "Severe":
            bb.flag("critical", "Severe infection - consult a qualified agronomist / extension officer promptly.")
        rec = bb.get("treatment")
        if rec and rec["type"].lower().startswith("chemical"):
            bb.flag("info", "Chemical product: wear gloves, mask and eye protection, follow the label (PHI/re-entry "
                            "intervals) and rotate active ingredients between sprays.")
        if band != "Healthy":
            bb.flag("info", "Severity is a colour-based estimate (brown/yellow-orange tissue). Soil, dry bark or background "
                            "can inflate it.")

        added = len(bb["flags"]) - n0
        return f"Treatment status: {status} | {added} note(s) added"


class AdvisorAgent(Agent):
    name, icon = "Advisor Agent", "📝"
    role = "Writes the farmer-friendly report and answers follow-up questions."

    SYSTEM = (
        "You are the Advisor agent in a dragon fruit (pitahaya) stem-disease decision-support system. "
        "Other agents already diagnosed the plant; you only EXPLAIN their findings. "
        "Rules: use ONLY the products, doses and schedules given in the case data - never invent or change a dose; "
        "respect treatment_status (if 'on_hold' do not tell the user to spray; if 'provisional' say it is provisional); "
        "mention the important review_flags; use plain, practical language; be concise."
    )

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key, self.model = api_key.strip(), model.strip()

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def _call_llm(self, system: str, messages: list, max_tokens: int = 900) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        content = resp.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return content.strip()

    @staticmethod
    def _api_error_message(error: Exception) -> str:
        status = getattr(error, "status_code", None)
        if status == 401:
            return "Google rejected the API key (401). Check that it is valid and from Google AI Studio."
        if status == 403:
            return "Google denied this request (403). Check API access and the key's project permissions."
        if status == 404:
            return ("Gemini could not find or access this model (404). In the sidebar, try "
                    "`gemini-3.8-flash` and check Google's model list: "
                    "https://ai.google.dev/gemini-api/docs/models.")
        if status == 429:
            return "Gemini API quota or rate limit reached (429). Check your free-tier limits and try again later."
        if status in (500, 502, 503, 504):
            return f"Gemini is temporarily unavailable (HTTP {status}). Please wait a little and try again."
        if status:
            return f"Google Gemini API request failed (HTTP {status}): {type(error).__name__}."
        return f"Gemini unavailable ({type(error).__name__})."

    def template_report(self, bb: Blackboard) -> str:
        d, c, band, pct = bb["disease"], bb["confidence"], bb["band"], bb["severity_pct"]
        status, rec = bb["treatment_status"], bb.get("treatment")
        out = []
        if band == "Healthy":
            out.append(f"**Result:** The plant looks **healthy** ({pct}% lesion area). Closest class the model saw: "
                       f"{d.replace('_', ' ')} at {c}% confidence, which is not enough to act on.")
            out.append("**What to do:** Keep routine monitoring every 2 weeks; no pesticide needed.")
        else:
            out.append(f"**Diagnosis:** {d.replace('_', ' ')} ({c}% confidence). {DISEASE_INFO[d]}")
            out.append(f"**Severity:** {band} - about {pct}% of the plant area is affected.")
            if status == "on_hold":
                out.append("**Treatment:** On hold. The diagnosis is too uncertain - retake the photo and confirm "
                           "before applying any chemical.")
            elif rec:
                prefix = "Provisional treatment" if status == "provisional" else "Recommended treatment"
                out.append(f"**{prefix}:** {rec['product']} at {rec['dose']}. "
                           f"Type: {rec['type']}. Schedule: {rec['frequency']}. This is {rec['dose_pct_of_label']}% of the full label rate.")
            if bb.get("practices"):
                out.append("**Field practices:** " + " ".join(bb["practices"]))
        return "\n\n".join(out)

    def run(self, bb):
        if self.llm_enabled:
            try:
                txt = self._call_llm(
                    self.SYSTEM,
                    [{"role": "user", "content": "Write the final report for this case (max ~180 words, markdown, "
                                                 "with short bold labels).\n\nCASE DATA:\n"
                                                 + json.dumps(bb.case_summary(), indent=2, default=str)}])
                if txt:
                    bb["report"], bb["report_source"] = txt, "LLM"
                    return f"Report written by LLM ({self.model})."
            except Exception as e:  # noqa: BLE001
                message = self._api_error_message(e) + " Using template report."
                bb.log(self.name, self.icon, "warn", message)
        bb["report"], bb["report_source"] = self.template_report(bb), "template"
        return "Report generated from template."

    def chat(self, bb: Blackboard, history: List[dict], question: str) -> str:
        if not self.llm_enabled:
            return "Add a Gemini API key in the sidebar to chat with the Advisor agent."
        system = (self.SYSTEM + " You may also answer general dragon-fruit agronomy follow-ups, but for anything "
                  "outside the case data, say it is general guidance and recommend confirming with a local extension officer. "
                  "Never give a dose that is not in the case data.\n\nCASE DATA:\n"
                  + json.dumps(bb.case_summary(), indent=2, default=str))
        msgs = [{"role": m["role"], "content": m["content"]} for m in history] + [{"role": "user", "content": question}]
        try:
            return self._call_llm(system, msgs, max_tokens=700)
        except Exception as e:  # noqa: BLE001
            if getattr(e, "status_code", None) in (401, 403, 404, 429, 500, 502, 503, 504):
                return self._api_error_message(e) + " Your question was not answered."
            raise


# ──────────────────────────────────────────────────────────────────────────
# 5. ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────
class Orchestrator:
    """Plans the run, delegates to agents, reflects on results and re-routes."""

    def __init__(self, registry: ModelRegistry, advisor: AdvisorAgent, tta_mode: str = "auto"):
        self.quality = ImageQualityAgent()
        self.classifier = ClassifierAgent(registry)
        self.severity = SeverityAgent()
        self.treatment = TreatmentAgent()
        self.review = SafetyReviewAgent()
        self.advisor = advisor
        self.tta_mode = tta_mode  # auto | always | never

    def run(self, image_rgb: np.ndarray, listener: Optional[Callable[[TraceEvent], None]] = None) -> Blackboard:
        bb = Blackboard(image_rgb)
        bb.listener = listener
        O = "Orchestrator"
        bb.log(O, "🎛️", "info", f"Plan: quality -> classify (TTA={self.tta_mode}) -> severity -> treatment -> review -> report")

        self.quality(bb)

        bb["tta"] = self.tta_mode == "always"
        if not self.classifier(bb):
            bb.fatal = bb.trace[-1].message
            return bb

        # Reflection: if unsure, ask the classifier for a second opinion with TTA
        if self.tta_mode == "auto" and (bb["confidence"] < CONF_REFLECT or bb["agreement"] < 1.0):
            before = (bb["disease"], bb["confidence"])
            bb.log(O, "🎛️", "info", f"Reflect: confidence {bb['confidence']}% / agreement {bb['agreement']:.0%} is weak "
                                    "-> re-running classifier with Test-Time Augmentation")
            bb["tta"] = True
            if self.classifier(bb):
                changed = "changed the diagnosis" if bb["disease"] != before[0] else "kept the same diagnosis"
                bb.log(O, "🎛️", "info", f"Second opinion {changed}: {before[0]} {before[1]}% -> {bb['disease']} {bb['confidence']}%")

        self.severity(bb)
        self.treatment(bb)
        self.review(bb)
        self.advisor(bb)
        bb.log(O, "🎛️", "ok", "Pipeline complete.")
        return bb


# ──────────────────────────────────────────────────────────────────────────
# 6. STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────
PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#2a2f3a", fontcolor="white", color="#8e24aa", fontname="Helvetica"];
  edge [color="#999999"];
  IMG [label="📷 Image", fillcolor="#37474f"];
  ORC [label="🎛️ Orchestrator", fillcolor="#8e24aa"];
  Q [label="🔍 Image Quality"];
  C [label="🧠 Classifier\\n(ResNet50)"];
  S [label="📏 Severity (HSV)"];
  T [label="💊 Treatment"];
  R [label="🛡️ Safety Review"];
  A [label="📝 Advisor (LLM)"];
  IMG -> ORC -> Q -> C -> S -> T -> R -> A;
  C -> C [label="low conf → TTA", fontcolor="#cccccc", fontsize=10];
  R -> ORC [style=dashed, constraint=false, label="flags", fontcolor="#cccccc", fontsize=10];
}
"""

CSS = """
<style>
.main-title{font-size:2.3rem;font-weight:800;background:linear-gradient(90deg,#e53935,#8e24aa);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:.2rem}
.subtitle{text-align:center;color:#888;margin-bottom:1.2rem}
.info-box{background:#161b22;border-radius:8px;padding:.8rem 1rem;font-size:.92rem;color:#ccc;
 border-left:3px solid #8e24aa;margin:.5rem 0}
</style>
"""

TRACE_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "💭"}


def render_flags(flags):
    for f in flags:
        {"critical": st.error, "warn": st.warning, "info": st.info}[f["level"]](f["text"])


def render_diagnosis(bb: Blackboard):
    disease, conf = bb["disease"], bb["confidence"]
    band, pct, status = bb["band"], bb["severity_pct"], bb["treatment_status"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🦠 Disease", disease.replace("_", " "))
    m2.metric("📊 Confidence", f"{conf:.1f}%")
    m3.metric("📏 Lesion area", f"{pct}%")
    m4.metric("⚠️ Severity", f"{BAND_ICON[band]} {band}")

    if bb["flags"]:
        st.markdown("#### 🛡️ Safety review")
        render_flags(sorted(bb["flags"], key=lambda f: {"critical": 0, "warn": 1, "info": 2}[f["level"]]))

    st.markdown("#### 📝 Advisor report")
    st.markdown(bb["report"])
    st.caption(f"Report source: {bb['report_source']}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📷 Input**")
        st.image(Image.fromarray(bb["image"]), width=300)
    with c2:
        st.markdown("**🧪 Lesion overlay**")
        st.image(lesion_overlay(bb["resized"], bb["lesion_mask"]), width=300)
    with c3:
        import matplotlib.pyplot as plt
        st.markdown("**📊 Class probabilities**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        names = [c.replace("_", " ") for c in CLASSES]
        vals = [bb["probs"][c] for c in CLASSES]
        ax.barh(names, vals, color=["#e53935" if i == bb["class_idx"] else "#546e7a" for i in range(len(CLASSES))])
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### 🧠 ResNet50 prediction")
    st.dataframe([{"Model": v["model"], "Prediction": v["prediction"].replace("_", " "),
                   "Confidence (%)": v["confidence"]}
                  for v in bb["votes"]], hide_index=True)

    st.divider()
    st.markdown("### 💊 Pesticide & dosage")
    st.markdown(f'<div class="info-box">🔬 <b>{disease.replace("_", " ")}:</b> {DISEASE_INFO[disease]}</div>',
                unsafe_allow_html=True)
    rec = bb.get("treatment")
    if status == "none":
        st.success("✅ Plant appears healthy. Continue routine monitoring every 2 weeks. No pesticide needed.")
    elif rec is None:
        st.info("No database entry - consult an agronomist for a tailored recommendation.")
    else:
        def show_rec():
            a, b = st.columns([2, 1])
            with a:
                st.markdown(f"**Product:** {rec['product']}")
                st.markdown(f"**Dose:** {rec['dose']}")
                st.markdown(f"**Frequency:** {rec['frequency']}")
                st.markdown(f"**Type:** {rec['type']}")
            with b:
                st.progress(rec["dose_pct_of_label"] / 100)
                st.markdown(f"**{rec['dose_pct_of_label']}%** of full label rate")

        if status == "on_hold":
            st.error("⛔ Treatment ON HOLD - diagnosis too uncertain. Retake the photo first.")
            with st.expander("Show the tentative recommendation anyway"):
                show_rec()
        else:
            if status == "provisional":
                st.warning("🟡 PROVISIONAL recommendation - confirm the diagnosis before spraying.")
            show_rec()
        if bb.get("practices"):
            st.markdown("**Field practices:** " + " ".join(bb["practices"]))

    st.caption("⚕️ Decision-support tool. Always confirm dosage with the product label and a qualified agronomist.")


def render_trace(bb: Blackboard):
    st.graphviz_chart(PIPELINE_DOT)
    st.markdown("#### Execution trace")
    for ev in bb.trace:
        secs = f"  _({ev.seconds:.2f}s)_" if ev.seconds else ""
        st.markdown(f"{TRACE_ICON.get(ev.status, '•')} **{ev.icon} {ev.agent}**{secs}  \n&nbsp;&nbsp;&nbsp;&nbsp;{ev.message}")
    total = sum(e.seconds for e in bb.trace)
    st.caption(f"Total agent compute time: {total:.2f}s")


def render_chat(bb: Blackboard, advisor: AdvisorAgent):
    if not advisor.llm_enabled:
        st.info("💬 Enable the LLM in the sidebar (Gemini API key) to ask the Advisor agent follow-up questions "
                "about this case. The rest of the system works without it.")
        return
    st.caption("Ask the Advisor about this diagnosis - e.g. *Why not Anthracnose?*, *How do I mix the spray?*, *When should I re-check?*")
    hist = st.session_state.setdefault("chat", [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask the Advisor agent…")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Advisor is thinking…"):
                try:
                    ans = advisor.chat(bb, hist, q)
                except Exception as e:  # noqa: BLE001
                    ans = f"⚠️ LLM error: {type(e).__name__}: {e}"
            st.markdown(ans)
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


def render_models(registry: ModelRegistry):
    st.markdown(f"**Models folder:** `{registry.models_dir}`")
    row = {
        "Model": MODEL_NAME,
        "File found": os.path.basename(registry.paths[MODEL_NAME]) if MODEL_NAME in registry.paths else "❌ missing",
    }
    m = (registry.metrics or {}).get(MODEL_NAME)
    if m:
        row.update({"Val Acc (%)": m["val_accuracy"], "Macro F1 (%)": m["macro_f1"], "Weighted F1 (%)": m["weighted_f1"]})
    rows = [row]
    st.dataframe(rows, hide_index=True)


def main():
    st.set_page_config(page_title="Dragon Fruit Multi-Agent Advisor", page_icon="🐉", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🐉 Multi-Agent Advisor")
        models_dir = st.text_input("Models folder", os.environ.get("DF_MODELS_DIR", "./models"),
                                   help="Folder containing your ResNet50 .keras model (or the best-model file).")
        tta_mode = st.selectbox("Test-Time Augmentation", ["auto", "always", "never"],
                                help="auto = the Orchestrator re-runs the classifier with TTA only when it is unsure.")
        st.divider()
        st.markdown("**🤖 Gemini Advisor (optional)**")
        st.markdown("[Get a Gemini API key from Google AI Studio](https://aistudio.google.com/apikey)")
        api_key = st.text_input("Gemini API key", os.environ.get("GEMINI_API_KEY", ""), type="password",
                                help="Google AI Studio offers free-tier API access for eligible models, subject to quotas. Leave empty to use the built-in template.")
        llm_model = st.text_input("Gemini model", os.environ.get("DF_LLM_MODEL", "gemini-3.8-flash"),
                                  help="Use a model available to your API key, for example gemini-3.8-flash.")
        st.divider()
        st.markdown("**Agents**")
        for a in (ImageQualityAgent, ClassifierAgent, SeverityAgent, TreatmentAgent, SafetyReviewAgent, AdvisorAgent):
            st.markdown(f"{a.icon} **{a.name}**  \n<small>{a.role}</small>", unsafe_allow_html=True)
        st.divider()
        st.caption("⚕️ For severe cases always consult a qualified agronomist.")

    st.markdown('<p class="main-title">🐉 Dragon Fruit Disease — Multi-Agent System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Six cooperating agents diagnose, grade severity, check safety and advise on treatment</p>',
                unsafe_allow_html=True)

    registry = ModelRegistry(models_dir)
    advisor = AdvisorAgent(api_key, llm_model)

    if not registry.available:
        st.error(f"No trained models found in `{registry.models_dir}`.")
        st.markdown("Copy `dragon_fruit_resnet50_model.keras` into the models folder, or place "
                    f"`{BEST_MODEL_FILENAME}` next to `app.py` or in the models folder.")
        st.stop()

    up = st.file_uploader("📁 Upload a dragon fruit stem image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    run = st.button("🚀 Run multi-agent analysis", type="primary", disabled=up is None)

    if run and up is not None:
        img = np.array(Image.open(up).convert("RGB"))
        orch = Orchestrator(registry, advisor, tta_mode)
        with st.status("Agents at work…", expanded=True) as status:
            def live(ev: TraceEvent):
                st.write(f"{TRACE_ICON.get(ev.status, '•')} {ev.icon} **{ev.agent}** - {ev.message}")
            bb = orch.run(img, listener=live)
            if bb.fatal:
                status.update(label="Pipeline failed", state="error")
            else:
                status.update(label="All agents finished", state="complete", expanded=False)
        st.session_state["case"] = bb
        st.session_state["chat"] = []

    bb: Optional[Blackboard] = st.session_state.get("case")
    tab_d, tab_t, tab_c, tab_m = st.tabs(["🩺 Diagnosis", "🧭 Agent trace", "💬 Ask the Advisor", "📊 Models"])
    with tab_d:
        if bb is None:
            st.info("👆 Upload an image and press **Run multi-agent analysis**.")
            st.graphviz_chart(PIPELINE_DOT)
        elif bb.fatal:
            st.error(bb.fatal)
        else:
            render_diagnosis(bb)
    with tab_t:
        if bb is None:
            st.info("Run an analysis to see what each agent did.")
        else:
            render_trace(bb)
    with tab_c:
        if bb is None or bb.fatal:
            st.info("Run an analysis first.")
        else:
            render_chat(bb, advisor)
    with tab_m:
        render_models(registry)


if __name__ == "__main__":
    main()
