# 🐉 Dragon Fruit Disease — Multi-Agent System

## Run it (VS Code terminal)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10 – 3.12 (TensorFlow does not support newer versions yet).

## Models
Create a `models/` folder next to `app.py` and copy in your ResNet50 model:
```
models/
  dragon_fruit_resnet50_model.keras
```
The existing `dragon_fruit_best_model (1).keras` can also be used directly from the
project folder (next to `app.py`) or copied into `models/`; it is loaded as a
ResNet50-preprocessed classifier when a specifically named ResNet50 model is not
found. The Classifier Agent only runs ResNet50; other model files are ignored.
Different folder? Change it in the sidebar or set `DF_MODELS_DIR`.

## Optional: LLM Advisor
Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey), then paste it
into the **Gemini API key** field in the sidebar. The default model is `gemini-3.8-flash`; change
the **Gemini model** field if you need another model available to your key. The Advisor uses
Google's OpenAI-compatible Gemini API. Free-tier access is subject to Google's model availability,
quotas, and terms; check the [pricing and free-tier details](https://ai.google.dev/gemini-api/docs/pricing).
See Google's [available model IDs](https://ai.google.dev/gemini-api/docs/models) if a model is not
available to your API key.
Keep your key private and do not commit it to source control. Without a key, image analysis still
works and the Advisor uses its built-in template report.

## Agents
| Agent | Job |
|---|---|
| 🎛️ Orchestrator | Plans the run, reflects, re-routes (re-runs classifier with TTA when unsure) |
| 🔍 Image Quality | Blur / exposure / resolution / "is this even a plant?" |
| 🧠 Classifier | ResNet50, optional TTA |
| 📏 Severity | HSV lesion segmentation → % area → Mild / Moderate / Severe |
| 💊 Treatment | Product, dose, schedule from your pesticide DB + field practices |
| 🛡️ Safety Review | Critic – flags disagreement/low confidence, puts treatment on hold if unsure |
| 📝 Advisor | Final report + chat (Gemini API, optional) |
