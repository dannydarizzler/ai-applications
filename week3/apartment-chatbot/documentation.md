# Documentation
## Apartment Rent Chatbot (Regression Model + LLM Workflow)

Use this file to document what you built, tested, and learned in this exercise.

Do not rename this file to `README.md`, because `README.md` is needed by Hugging Face Spaces.

This file is part of the submission. Complete it after you have tested and deployed your app.

---

## 1. Project Summary

**Short description of your app:**  
This app is a conversational chatbot that predicts apartment rents in the Canton of Zurich. The user types a natural language request (in German or English), and the app uses Claude (claude-opus-4-7) to extract the relevant parameters, runs a pre-trained Random Forest regression model for the price prediction, and returns a friendly explanation in natural language. The LLM acts both as a parameter extractor (NLU) and as a natural language generator (NLG) for the final explanation.

---

## 2. Files Used

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio chatbot application |
| `apartment_model.pkl` | Saved Random Forest regression model (trained in Week 3) |
| `bfs_municipality_and_tax_data.csv` | Municipality features used for prediction |
| `requirements.txt` | Python dependencies |
| `documentation.md` | Written documentation for the submission |

---

## 3. Numeric Prediction Part

### 3.1 Reused Model

**Which saved model did you use?**  
`apartment_model.pkl` (Random Forest Regressor trained in Week 3)

**What does the model predict?**  
The model predicts the monthly rent in CHF for an apartment in the Canton of Zurich based on apartment characteristics and municipality statistics.

**Which input features are used for prediction?**

1. `rooms` — number of rooms (e.g. 3.5)
2. `area` — living area in m²
3. `pop` — population of the municipality
4. `pop_dens` — population density
5. `frg_pct` — percentage of foreign residents
6. `emp` — number of employees in the municipality
7. `tax_income` — average taxable income in the municipality
8. `area_per_room` — derived feature: area divided by number of rooms

### 3.2 Prediction Logic

The user's municipality name is looked up in `bfs_municipality_and_tax_data.csv`. The municipality's statistical features (population, density, tax income, etc.) are combined with the user's input (rooms, area) and a derived feature (`area_per_room = area / rooms`). This feature vector is passed to the Random Forest model via `model.predict()`. The result is capped at 0 CHF minimum and returned as a JSON object.

---

## 4. LLM Extraction Part

### 4.1 Goal

The LLM extracts three parameters from the user's free-text input: `rooms` (number of rooms), `area` (living area in m²), and `municipality` (name of the municipality in the Canton of Zurich).

### 4.2 Prompt Design

A system prompt instructs Claude to act as a Swiss real estate assistant. It defines:
- The role: extract parameters from natural language, call the prediction tool
- Fallback behavior: make reasonable assumptions if parameters are missing (e.g. 80m² for a 3-room flat)
- Output format: always show extracted JSON, predicted rent, and an explanation
- Language rule: respond in the same language the user writes in

Claude uses **tool use** (function calling): a tool `predict_apartment_rent` with three required parameters is defined. Claude decides when to call it and with which values. The prompt caching feature (`cache_control: ephemeral`) is applied to the system prompt to reduce API costs across turns.

### 4.3 Expected Output Format

```json
{"rooms": 3.5, "area": 85, "municipality": "Winterthur"}
```

The tool returns:
```json
{
  "municipality": "Winterthur",
  "rooms": 3.5,
  "area_m2": 85,
  "predicted_monthly_rent_chf": 2140
}
```

### 4.4 Validation

The extracted values are validated implicitly: if `rooms <= 0` or `area <= 0`, the prediction would be nonsensical (caught by the model). If the municipality is not found in the lookup table, fuzzy matching is applied (substring match). If no match is found, an error JSON is returned and Claude explains the problem to the user.

---

## 5. LLM Explanation Part

### 5.1 Goal

After the prediction, Claude formulates a short, friendly explanation of the result. Importantly, Claude does **not** calculate or invent a new price — it only explains the prediction value returned by the regression model.

### 5.2 Prompt Design

The system prompt instructs Claude to always structure its response in three parts:
1. Extracted parameters as a JSON code block (visible to the user)
2. The predicted rent in CHF
3. A 2-3 sentence explanation that mentions the prediction is an estimate and real prices may vary

Claude receives the tool result (predicted rent) and uses it directly in the response. The LLM adds context about uncertainty (model limitations, location factors not captured).

### 5.3 Expected Output Format

Example response from Claude:

```
**Extracted parameters (JSON):**
```json
{"rooms": 3.5, "area": 85, "municipality": "Winterthur"}
```

**Predicted rent:** CHF 2'140 / month

**Explanation:** Für eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur schätzt das Modell rund 2'140 CHF pro Monat. Zu beachten ist, dass der tatsächliche Mietpreis je nach Zustand, Stockwerk und Mikrolage abweichen kann. Das Modell basiert auf statistischen Gemeindedaten und liefert eine fundierte Schätzung.
```

---

## 6. End-to-End Pipeline

1. User enters an apartment request in German or English (e.g. "Wie viel kostet eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur?")
2. Claude extracts `rooms`, `area`, and `municipality` and calls the `predict_apartment_rent` tool
3. Python validates the municipality name (fuzzy match if needed) and builds the feature vector
4. The Random Forest model predicts the monthly rent in CHF
5. The tool result (JSON with prediction) is returned to Claude
6. Claude generates a structured response: extracted JSON + predicted rent + explanation
7. The Gradio chatbot displays the full response to the user

---

## 7. Test Cases

| Test Input | Extracted Output Correct? | Prediction Returned? | Explanation Returned? | Notes |
|------------|---------------------------|----------------------|-----------------------|-------|
| `Wie viel kostet eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur?` | Yes | Yes | Yes | All parameters present, clean extraction |
| `Ich suche eine 4-Zimmer-Wohnung in Zürich, ca. 100 m².` | Yes | Yes | Yes | Claude assumed typical area |
| `Was kostet eine kleine 2-Zimmer-Wohnung in Dietikon?` | Yes | Yes | Yes | Claude assumed ~55m² for 2 rooms |
| `Estimate rent for a 5-room apartment in Küsnacht (ZH), 150m².` | Yes | Yes | Yes | English input, responded in English |
| `Wohnung in Bern` | Partial | No | Yes | Bern is not in Canton of Zurich, Claude explained this |

---

## 8. Errors and Problems

**Problem:** HuggingFace Space showed 401/404 errors on startup  
**Cause:** The model `Danydarizzler/pokemon-vit` (from another project) didn't exist yet  
**Fix:** Trained and uploaded the model via Python script

**Problem:** `tokenizer` argument in `Trainer` caused `TypeError`  
**Cause:** Newer version of `transformers` renamed the parameter  
**Fix:** Changed `tokenizer=processor` to `processing_class=processor`

**Problem:** `accelerate` library missing  
**Cause:** Not installed in conda base environment  
**Fix:** `pip install "accelerate>=1.1.0"`

**Problem:** HF token had Read-only permissions  
**Cause:** Created the wrong token type  
**Fix:** Created a new token with Write permissions

---

## 9. Deployment Notes

### 9.1 Files included

- `app.py`
- `requirements.txt`
- `apartment_model.pkl`
- `bfs_municipality_and_tax_data.csv`
- `documentation.md`

### 9.2 Secrets / Environment Variables

- `ANTHROPIC_API_KEY` — required for Claude API access

### 9.3 Deployment Result

The Space runs successfully. The chatbot accepts German and English inputs, extracts parameters, runs the regression model, and returns structured responses with visible JSON, prediction, and explanation.

### 9.4 Screenshots

![Example 1](screenshot1.png)
*German query: "Wie viel kostet eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur?" — Claude extracts parameters, shows JSON, predicts ~2140 CHF, and explains the result in German.*

![Example 2](screenshot2.png)
*German query: "Ich suche eine 4-Zimmer-Wohnung in Zürich mit 100 m²." — Claude extracts parameters, shows JSON, predicts ~2800 CHF, and provides a friendly explanation.*

---

## 10. Reflection

The combination of regression model and LLM works well: the regression model provides reliable, data-driven predictions while the LLM handles the messy real-world input (different phrasings, missing values, different languages). The system is fragile when users name municipalities outside the Canton of Zurich or use very unusual phrasing that causes wrong parameter extraction. German input is important because Swiss users naturally write in German and the chatbot should feel natural and local. The model is missing important factors like apartment condition, floor level, proximity to public transport, and year of construction — all of which significantly affect rent in Switzerland. In a next version, I would add validation that shows the user the extracted parameters before running the prediction, so they can correct mistakes.

---

## 11. Responsible Use Note

This app provides rent estimates based on a statistical model trained on historical data from the Canton of Zurich — it is not a guarantee of actual market prices. The LLM may occasionally extract parameters incorrectly from ambiguous inputs, leading to wrong predictions. The model uses aggregated municipality statistics and does not account for individual apartment characteristics like condition, renovation status, or exact street-level location. Users should treat predictions as a rough guide and always verify with current listings on platforms like Homegate or ImmoScout24.
