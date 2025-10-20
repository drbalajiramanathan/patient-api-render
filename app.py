import gradio as gr
import os
import json
from huggingface_hub import InferenceClient

# --- Configuration ---
# Reads the HF_TOKEN secret from Render's environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Global variable for the client, for delayed loading ---
client = None

def get_inference_client():
    """Initializes the InferenceClient only when it's first needed."""
    global client
    if client is None:
        print("Initializing InferenceClient for the first time...")
        if not HF_TOKEN:
            print("ERROR: HF_TOKEN secret not found!")
            raise gr.Error("HF_TOKEN secret is not configured on the server.")
        try:
            client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
            print("InferenceClient initialized successfully.")
        except Exception as e:
            print(f"FATAL: Error initializing InferenceClient: {e}")
            raise gr.Error(f"Could not initialize AI model client: {e}")
    return client

def build_json_prompt(diagnosis, age, comorbidities):
    """Creates the prompt to generate the JSON data."""
    comorbidity_list = ", ".join(comorbidities) if comorbidities else "None"
    return f"""
    You are a clinical data simulator. Your task is to generate a realistic 7-day hospital
    trajectory for a patient.

    Patient Profile:
    - Diagnosis: {diagnosis}
    - Age: {age}
    - Comorbidities: {comorbidity_list}

    Generate daily values for: HR (Heart Rate), BP_Sys (Systolic BP), Temp_C (Temperature in Celsius),
    WBC_Count (White Blood Cell Count), and a 1-sentence clinical note.

    Respond *only* with a valid JSON object in the following exact format.
    Do not include any other text, explanation, or markdown formatting.

    {{"trajectory": [
        {{"day": 1, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}},
        {{"day": 2, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}},
        {{"day": 3, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}},
        {{"day": 4, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}},
        {{"day": 5, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}},
        {{"day": 6, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}},
        {{"day": 7, "hr": 0, "bp_sys": 0, "temp_c": 0.0, "wbc_count": 0.0, "note": "string"}}
    ]}}
    """

def generate_trajectory_and_summary(diagnosis, age, comorbidities):
    """Generates and returns both the JSON data and a detailed layman's summary."""
    raw_json_response = ""
    try:
        inference_client = get_inference_client()

        # --- Step 1: Generate the trajectory JSON ---
        json_prompt = build_json_prompt(diagnosis, age, comorbidities)
        json_response_obj = inference_client.chat_completion(
            messages=[{"role": "user", "content": json_prompt}],
            max_tokens=1500, temperature=0.7
        )
        raw_json_response = json_response_obj.choices[0].message.content
        json_output = json.loads(raw_json_response)

        # --- Step 2: Generate the new, detailed layman's summary ---
        # This prompt is updated to ask for the day-by-day list and the narrative.
        summary_prompt = f"""
        Analyze the following clinical JSON data. Your task is to create a summary for a non-medical person.
        The summary must have two parts:
        1. A day-by-day list of the key metrics (HR, BP, Temp, WBC).
        2. A concluding narrative paragraph that describes the patient's overall progress.

        Example format:
        Day 1: HR 95, BP 140, Temp 38.8°C, WBC 15.2
        Day 2: HR 92, BP 135, Temp 38.1°C, WBC 13.5
        ...
        Day 7: HR 78, BP 120, Temp 36.7°C, WBC 8.1
        
        The synthetic patient showed significant improvement over the course of treatment, with fever subsiding, white blood cell count decreasing, and vital signs stabilizing, ultimately leading to a successful discharge on the seventh day.

        Now, generate a similar summary for this data:
        Data: {json.dumps(json_output)}
        """
        summary_response_obj = inference_client.chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500, temperature=0.6
        )
        summary_text = summary_response_obj.choices[0].message.content.strip()

        # Gradio returns multiple outputs as a tuple
        return json_output, summary_text

    except Exception as e:
        print(f"Error in generation logic: {e}")
        error_message = f"An error occurred: {e}. Raw AI response was: {raw_json_response}"
        # Raise a Gradio-specific error to display it nicely in the UI
        raise gr.Error(error_message)

# --- Gradio UI Layout ---
# Removed the theme to go back to the default orange style.
with gr.Blocks(css=".gradio-container {max-width: 800px !important; margin: auto !important;}") as demo:
    gr.Markdown("# Interactive Synthetic Patient Generator")
    gr.Markdown("Select patient parameters and generate a simulated 7-day hospital trajectory using a Large Language Model.")

    with gr.Row():
        diag_input = gr.Dropdown(label="Admission Diagnosis", choices=["Pneumonia", "Heart Failure Exacerbation", "Post-Op Hip Replacement", "Sepsis"], value="Pneumonia")
        age_input = gr.Slider(label="Patient Age", minimum=18, maximum=100, value=65, step=1)

    comorbid_input = gr.CheckboxGroup(label="Comorbidities", choices=["Diabetes", "Hypertension", "COPD", "Smoker"])
    
    submit_btn = gr.Button("Generate Trajectory", variant="primary")

    with gr.Accordion("How It Works (Technical Details)", open=False):
        gr.Markdown(
            """
            This tool leverages a Large Language Model (LLM), `meta-llama/Meta-Llama-3-8B-Instruct`, accessed via an API. When you click "Generate," the user inputs are sent to the backend.
            
            1.  **JSON Generation:** The backend uses **structured prompt engineering** to create a highly specific prompt, constraining the LLM to act as a clinical simulator and return only a valid JSON object representing the 7-day trajectory.
            2.  **Summary Generation:** Upon receiving the valid JSON, the backend makes a second API call. It feeds the generated data back to the LLM with a new set of instructions: to produce a detailed day-by-day list of vitals and a concluding narrative paragraph.
            3.  **Display:** The backend returns both the structured JSON and the detailed summary to be displayed in the respective output boxes below.
            """
        )

    gr.Markdown("## Generated Trajectory")
    
    # Increased the height of the summary textbox
    summary_output_ui = gr.Textbox(
        label="Layman's Summary", 
        interactive=False, 
        placeholder="A detailed summary of the patient's progress will appear here...",
        lines=10, 
        max_lines=15
    )
    json_output_ui = gr.JSON(label="Raw JSON Data")

    submit_btn.click(
        fn=generate_trajectory_and_summary,
        inputs=[diag_input, age_input, comorbid_input],
        outputs=[json_output_ui, summary_output_ui]
    )

# --- Launch Gradio Server for Render ---
server_port = int(os.environ.get('PORT', 10000))
print(f"Attempting to launch Gradio on 0.0.0.0:{server_port}")

demo.launch(server_name="0.0.0.0", server_port=server_port)

