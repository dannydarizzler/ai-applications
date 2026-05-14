import gradio as gr
import numpy as np
import pandas as pd
import pickle

# ---------------------------------------------------------------------------
# Load model and municipality data
# ---------------------------------------------------------------------------
with open("apartment_model.pkl", "rb") as f:
    model = pickle.load(f)

df_bfs = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs["tax_income"] = df_bfs["tax_income"].str.replace("'", "").astype(float)

# Build dropdown choices (sorted alphabetically)
municipalities = sorted(df_bfs["bfs_name"].dropna().unique().tolist())
bfs_lookup = df_bfs.set_index("bfs_name").to_dict("index")

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_rent(rooms, area, municipality):
    if rooms <= 0 or area <= 0:
        return "Please enter valid values for rooms and area."

    row = bfs_lookup[municipality]

    # New feature: average room size in m² (Iteration 2 feature)
    area_per_room = area / rooms

    features = pd.DataFrame([{
        "rooms":        rooms,
        "area":         area,
        "pop":          row["pop"],
        "pop_dens":     row["pop_dens"],
        "frg_pct":      row["frg_pct"],
        "emp":          row["emp"],
        "tax_income":   row["tax_income"],
        "area_per_room": area_per_room,
    }])

    predicted_price = model.predict(features)[0]
    predicted_price = max(0, predicted_price)

    return f"CHF {predicted_price:,.0f} / month"

# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=predict_rent,
    inputs=[
        gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=3.5,
                  label="Number of Rooms"),
        gr.Slider(minimum=20, maximum=300, step=5, value=80,
                  label="Living Area (m²)"),
        gr.Dropdown(choices=municipalities, value="Zürich",
                    label="Municipality"),
    ],
    outputs=gr.Textbox(label="Predicted Monthly Rent"),
    title="Zurich Apartment Rent Predictor",
    description=(
        "Predicts the monthly rent for apartments in the Canton of Zurich. "
        "Select the number of rooms, living area, and municipality, then click Submit."
    ),
    examples=[
        [3.5, 80,  "Zürich"],
        [4.5, 110, "Winterthur"],
        [2.5, 55,  "Dietikon"],
        [5.5, 150, "Küsnacht (ZH)"],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
