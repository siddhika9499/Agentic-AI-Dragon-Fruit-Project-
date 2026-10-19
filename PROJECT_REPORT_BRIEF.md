# Dragon Fruit Disease Multi-Agent Advisor — Report Brief

Use this brief as context when asking an AI assistant to draft a project report. It describes the current application based on its source code and README. It does not include private API keys or claim unverified model results.

## Copy-and-paste prompt

> Write a clear, well-structured academic project report about the **Dragon Fruit Disease Multi-Agent Advisor** using only the information in the project brief below. Do not invent dataset size, data sources, train/test split, model accuracy, experimental results, citations, deployment results, or user-study outcomes. Where information is missing, explicitly say it was not provided and mark it as something the project author should add. Explain that this is a decision-support prototype, not a validated diagnostic or pesticide-prescription system. Include an abstract, introduction, problem statement, objectives, system design and workflow, implementation, testing/results (only verifiable facts; clearly mark missing evaluation data), limitations, ethical/safety considerations, future work, conclusion, and a short list of information still needed from the author. Keep the writing accessible and avoid overstating what the AI or agents do.

## Project summary

- **Project name:** Dragon Fruit Disease Multi-Agent Advisor.
- **Purpose:** A Streamlit prototype that accepts an image of a dragon fruit stem, predicts one of five disease classes with a locally stored CNN, estimates lesion area using image processing, looks up a treatment suggestion, runs safety checks, and presents a farmer-oriented report. An optional Gemini-powered Advisor can write the report and answer follow-up questions.
- **Application type:** Local Python web application.
- **Main entry point:** `app.py`.
- **Interface:** Streamlit, with Diagnosis, Agent trace, Ask the Advisor, and Models tabs.
- **Image input:** JPG, JPEG, or PNG; converted to RGB and resized to 224 × 224 for model classification.
- **Model selection:** The classifier runs **one model only**, named ResNet50 in the application. It first looks for `dragon_fruit_resnet50_model.keras` or another filename containing `resnet50` in the models folder. If none is found, it falls back to `dragon_fruit_best_model (1).keras` in that folder or next to `app.py`. The fallback is processed using ResNet50/Caffe-style preprocessing. Other model files are ignored.
- **Model present in this project folder at the time this brief was prepared:** `dragon_fruit_best_model (1).keras`. The code treats it as a ResNet50-preprocessed classifier, but this brief does not independently verify the saved model's internal architecture or training history.
- **Disease classes:** Anthracnose, Soft Rot, Gray Blight, Brown Stem Spot, and Stem Canker.
- **Prediction confidence:** The classifier produces class probabilities and reports the top class confidence. No measured accuracy or validation metric is documented in the project files.

## Agent workflow

The agents are Python classes coordinated by an Orchestrator. They exchange information through a shared Blackboard, which also records an execution trace.

1. **Image Quality Agent:** Measures image sharpness, brightness, and an approximate green/brown plant-pixel ratio; raises warnings for image quality concerns.
2. **Classifier Agent:** Runs the single ResNet50 model and records the top class and confidence.
3. **Orchestrator reflection / TTA:** Test-Time Augmentation can be set to `auto`, `always`, or `never`. In `auto`, if confidence is below 75%, the classifier is rerun using the original image, a horizontal flip, and a 90% center crop resized to the input size. This is a second inference with the same model, not a second model or independent expert.
4. **Severity Agent:** Uses HSV colour thresholds and morphological operations to estimate the percentage of detected brown/yellow-orange lesion pixels relative to detected green-or-lesion plant pixels. It assigns severity bands:
   - Less than 1%: Healthy
   - 1% through 15%: Mild
   - Above 15% through 40%: Moderate
   - Above 40%: Severe
5. **Treatment Agent:** Looks up a product, dose, and schedule from a static in-code table keyed by predicted class and severity. It also adds general field practices.
6. **Safety Review Agent:** Uses confidence thresholds and severity to set treatment status. Below 50% confidence, treatment is put on hold; below 70%, it is provisional. It adds warnings about uncertainty, severe infection, chemical handling, and the limitations of colour-based severity estimation.
7. **Advisor Agent:** If configured, sends case information to Gemini for a concise report and follow-up chat. Otherwise, a deterministic template report is used. Prompt instructions tell the language model not to change treatment doses and to respect on-hold/provisional statuses.

## Technical implementation

- **Language:** Python.
- **Web UI:** Streamlit.
- **Deep learning:** TensorFlow/Keras; `.keras` model loaded lazily and cached.
- **Image processing:** OpenCV and NumPy.
- **Image reading:** Pillow.
- **Charts/data display:** Matplotlib and pandas.
- **Optional LLM client:** OpenAI Python SDK used with Google's OpenAI-compatible Gemini endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`).
- **Default Gemini model setting:** `gemini-3.8-flash`. Model availability depends on Google's API and the user's project/key.
- **API key:** Entered in the password-masked sidebar field or provided with the `GEMINI_API_KEY` environment variable. Never include a real key in a report, screenshot, source file, or shared prompt.
- **Model folder:** Defaults to `./models`; can be changed in the sidebar or with `DF_MODELS_DIR`.
- **LLM model override:** `DF_LLM_MODEL` environment variable or the sidebar's Gemini model field.
- **Dependencies:** Streamlit, TensorFlow, OpenCV headless, NumPy, Pillow, Matplotlib, pandas, and the OpenAI Python SDK.
- **Run locally on Windows:** Activate `.venv` with `.venv\Scripts\activate`, install dependencies using `pip install -r requirements.txt`, then run `streamlit run app.py`. The app has been run locally at `http://localhost:8501`.

## Gemini API behaviour

The Gemini integration is optional and only used by the Advisor for generated reports and chat. Image classification, severity estimation, treatment lookup, and template reports are local application logic. If the key is absent, or Gemini is unavailable, the app falls back to a template report for the case. Chat requires the Gemini API. Free-tier availability, quota, and data handling are subject to Google's current policies and account configuration.

## Known limitations and safety considerations

- The dataset, number of images, labels, collection method, train/validation/test splits, training procedure, and independent evaluation results are not documented in the project files. Add these details only if verified from the original training work.
- No model accuracy, precision, recall, F1 score, confusion matrix, or comparison results are documented here. Do not invent or infer them from the presence of a trained model file.
- Image colour thresholds are heuristic and can be affected by lighting, camera settings, soil, dry tissue, shadows, and background objects. The calculated lesion percentage is an estimate, not a clinical/agronomic measurement.
- A single image classifier may misidentify a disease. Its output should be independently checked by a qualified agronomist or plant-disease specialist.
- The pesticide table is static code, not a live or regionally verified pesticide database. Product registration, label rates, crop approval, local regulations, personal protective equipment, pre-harvest intervals, and re-entry intervals vary by jurisdiction and must be checked against the current product label and local expert advice. Do not treat application output as a prescription.
- The optional language model can be unavailable due to API-key, quota, network, or model-service issues. It can also generate inaccurate text; its response must not override the safety review or authoritative labels.
- The app is a prototype/decision-support aid and should not be described as a validated commercial or clinical system.

## Testing and verification known from the development session

- Python syntax compilation passed for `app.py`.
- A classifier smoke test with a mocked model verified that only the ResNet50 registry entry is requested.
- A Gemini adapter smoke test with a mocked SDK verified the configured endpoint, model parameter, system/user messages, and error handling. This did not make a live Gemini API call.
- The local Streamlit server returned HTTP 200 during verification.
- These checks are software smoke tests, not scientific validation of disease predictions or treatments.

## Details the project author should add before final submission

- Author name, institution, course/program, supervisor, and submission date.
- The real-world problem context and intended users/location.
- Dataset source, collection period, class counts, image examples/criteria, permissions, and labelling procedure.
- Model architecture confirmation, training notebook/code, pretrained weights, augmentation, optimizer, hyperparameters, epochs, and hardware.
- Train/validation/test methodology and numerical evaluation results.
- Baselines or other models actually evaluated (if any). The current application only runs one ResNet50 model.
- Screenshots of the actual UI and outputs, with API keys and personal information hidden.
- Test cases, user feedback, deployment environment, and known reproducible results.
- Local agricultural sources and regulations used to validate each pesticide product and dose.
- References in the format required by the institution.
