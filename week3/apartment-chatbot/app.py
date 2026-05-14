import os
import json
import pickle
import pandas as pd
import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load model and municipality data
# ---------------------------------------------------------------------------

with open("apartment_model.pkl", "rb") as f:
    model = pickle.load(f)

df_bfs = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs["tax_income"] = df_bfs["tax_income"].str.replace("'", "").astype(float)
municipalities = sorted(df_bfs["bfs_name"].dropna().unique().tolist())
bfs_lookup = df_bfs.set_index("bfs_name").to_dict("index")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE"))

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict_rent(rooms: float, area: float, municipality: str) -> str:
    if municipality not in bfs_lookup:
        matches = [m for m in municipalities if municipality.lower() in m.lower()]
        if matches:
            municipality = matches[0]
        else:
            return json.dumps({
                "error": f"Municipality '{municipality}' not found in Canton of Zurich.",
                "suggestion": "Try one of: Zürich, Winterthur, Dietikon, Uster, Dübendorf, Küsnacht (ZH)"
            })

    row = bfs_lookup[municipality]
    area_per_room = area / rooms

    features = pd.DataFrame([{
        "rooms":         rooms,
        "area":          area,
        "pop":           row["pop"],
        "pop_dens":      row["pop_dens"],
        "frg_pct":       row["frg_pct"],
        "emp":           row["emp"],
        "tax_income":    row["tax_income"],
        "area_per_room": area_per_room,
    }])

    predicted_price = max(0, model.predict(features)[0])
    return json.dumps({
        "municipality": municipality,
        "rooms": rooms,
        "area_m2": area,
        "predicted_monthly_rent_chf": round(predicted_price)
    })

# ---------------------------------------------------------------------------
# OpenAI tool definition
# ---------------------------------------------------------------------------

PREDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_apartment_rent",
        "description": (
            "Predicts the monthly rent (in CHF) for an apartment in the Canton of Zurich "
            "based on number of rooms, living area in m², and municipality name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rooms": {
                    "type": "number",
                    "description": "Number of rooms (e.g. 2.5, 3.0, 4.5). Half-rooms are common in Switzerland."
                },
                "area": {
                    "type": "number",
                    "description": "Living area in square meters (e.g. 60, 80, 120)."
                },
                "municipality": {
                    "type": "string",
                    "description": "Municipality name in the Canton of Zurich (e.g. 'Zürich', 'Winterthur', 'Dietikon')."
                }
            },
            "required": ["rooms", "area", "municipality"]
        }
    }
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly Swiss real estate assistant that predicts apartment rents in the Canton of Zurich.

When a user asks about apartment prices or rents, extract rooms, area (m²), and municipality from their message, then call the predict_apartment_rent tool.

Rules:
- If parameters are missing, make reasonable assumptions (e.g. 80m² for a 3-room flat) and tell the user.
- The model only covers the Canton of Zurich. Common municipalities: Zürich, Winterthur, Dietikon, Uster, Dübendorf, Küsnacht (ZH), Zollikon, Thalwil, Regensdorf, Schlieren, Adliswil.
- Respond in the same language the user writes in (German or English).
- Half-rooms are standard in Switzerland (e.g. 3.5 Zimmer = 3 rooms + 1 bathroom counted as half).

Always structure your response in this exact format:
1. **Extracted parameters (JSON):** show the extracted values as a JSON code block
2. **Predicted rent:** state the CHF amount clearly
3. **Explanation:** 2-3 sentences explaining the prediction in a friendly tone, mentioning that it is an estimate and real prices may vary.

Keep responses friendly and helpful."""

# ---------------------------------------------------------------------------
# Chat function with OpenAI tool use loop
# ---------------------------------------------------------------------------

def chat(message: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for entry in history:
        if isinstance(entry, dict):
            messages.append({"role": entry["role"], "content": entry["content"]})
        else:
            human, assistant = entry
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    # Agentic loop: run until model stops calling tools
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[PREDICT_TOOL],
            tool_choice="auto",
        )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = predict_rent(
                    rooms=args["rooms"],
                    area=args["area"],
                    municipality=args["municipality"]
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Zurich Rent Chatbot") as demo:
    gr.Markdown("""
    # Zurich Apartment Rent Chatbot
    Frag mich nach Mietpreisen im **Kanton Zürich** — auf Deutsch oder Englisch.

    **Beispiele:**
    - *"Wie viel kostet eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur?"*
    - *"Ich suche eine 4-Zimmer-Wohnung in Zürich mit 100 m²."*
    """)

    gr.ChatInterface(
        fn=chat,
        examples=[
            "Wie viel kostet eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur?",
            "Ich suche eine 4-Zimmer-Wohnung in Zürich mit 100 m².",
            "Was kostet eine 2-Zimmer-Wohnung in Dietikon, ca. 55 m²?",
            "Wie teuer ist eine 5-Zimmer-Wohnung in Küsnacht (ZH) mit 140 m²?",
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
