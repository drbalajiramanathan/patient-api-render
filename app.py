import gradio as gr
import os
import json
from huggingface_hub import InferenceClient

# --- Configuration ---
# Reads the HF_TOKEN secret from Render's environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Global variable for the client, initialized as None for delayed loading ---
client = None

def get_inference_client():
    """Initializes the InferenceClient only when it's first needed."""
    global client
    if client is None:
        print("Initializing InferenceClient for the first time...")
        if not HF_TOKEN:
            print("ERROR: HF_TOKEN secret not found!")
            raise ValueError("HF_TOKEN environment variable not set.")
        try:
            client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
            print("InferenceClient initialized successfully.")
        except Exception as e:
            print(f"FATAL: Error initializing InferenceClient: {e}")
            client = None # Ensure client stays None on failure
            raise # Re-raise the exception to stop the process
    return client

def build_json_prompt(diagnosis, age, comorbidities):
    """Creates the detailed instruction prompt to generate the JSON data."""
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

def generate_trajectory_logic_api(diagnosis, age, comorbidities):
    """Function called by the API. Generates and returns both JSON and a summary."""
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

        # --- Step 2: Generate the layman's summary ---
        summary_prompt = f"""
        Summarize the key trend in the following clinical data in one simple, easy-to-understand sentence for a non-medical person.
        Start the sentence with "The synthetic patient...".
        
        Data: {json.dumps(json_output)}
        """
        summary_response_obj = inference_client.chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=100, temperature=0.5
        )
        summary_text = summary_response_obj.choices[0].message.content.strip()

        # Gradio returns multiple outputs as a tuple
        return json_output, summary_text

    except Exception as e:
        print(f"Error in API logic: {e}")
        error_json = {"error": str(e), "raw_response_for_debugging": raw_json_response}
        error_summary = "An error occurred while generating the trajectory."
        return error_json, error_summary

# --- Gradio UI Setup (for testing on Render) ---
with gr.Blocks() as demo:
    gr.Markdown("# Synthetic Patient Trajectory API 🏥")
    gr.Markdown("Test UI for the generation endpoint.")

    with gr.Row():
        diag_input = gr.Dropdown(label="Admission Diagnosis", choices=["Pneumonia", "Heart Failure Exacerbation", "Post-Op Hip Replacement", "Sepsis"])
        age_input = gr.Slider(label="Patient Age", minimum=18, maximum=100, value=65)

    comorbid_input = gr.CheckboxGroup(label="Comorbidities", choices=["Diabetes", "Hypertension", "COPD", "Smoker"])
    submit_btn = gr.Button("Generate Test Data")
    
    # Define the two outputs for the UI
    json_output_ui = gr.JSON(label="Generated JSON Output")
    summary_output_ui = gr.Textbox(label="Layman's Summary Output", interactive=False)

    submit_btn.click(
        fn=generate_trajectory_logic_api,
        inputs=[diag_input, age_input, comorbid_input],
        outputs=[json_output_ui, summary_output_ui], # Gradio maps the function's tuple output here
        api_name="predict"
    )

# --- Launch Gradio Server for Render ---
# Reads the PORT environment variable set by Render
server_port = int(os.environ.get('PORT', 10000))
print(f"Attempting to launch Gradio on 0.0.0.0:{server_port}")

# Launch specifying host 0.0.0.0 and the port Render provides
demo.launch(server_name="0.0.0.0", server_port=server_port)
