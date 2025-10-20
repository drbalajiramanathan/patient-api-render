import gradio as gr
import os
import json
from huggingface_hub import InferenceClient

# --- Configuration ---
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Global variable for the client, initialize as None ---
client = None

def get_inference_client():
    """Initializes the InferenceClient only when first needed."""
    global client
    if client is None:
        print("Initializing InferenceClient...") # Log attempt
        if not HF_TOKEN:
            print("HF_TOKEN secret not found!")
            # Raise error immediately if token is missing
            raise ValueError("HF_TOKEN environment variable not set.") 

        try:
            client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
            print("InferenceClient initialized successfully.")
        except Exception as e:
            print(f"Error initializing InferenceClient: {e}")
            client = None # Ensure client stays None on failure
            raise # Re-raise the exception to signal failure
    # If client was already initialized or just initialized, return it
    if client is None:
         # This should ideally not be reached if initialization failed and raised error
         raise ValueError("Failed to get Inference Client instance.")
    return client

# --- Core Generation Logic & Build Prompt ---
def build_prompt(diagnosis, age, comorbidities):
    comorbidity_list = ", ".join(comorbidities) if comorbidities else "None"
    # Using the same detailed prompt as before...
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
    Do not include any other text, explanation, or markdown formatting like ```json.

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
    """Function called by the API - includes client initialization."""
    raw_response_text = ""
    try:
        inference_client = get_inference_client() # Get/Initialize client

        prompt = build_prompt(diagnosis, age, comorbidities)

        response = inference_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        raw_response_text = response.choices[0].message.content
        json_output = json.loads(raw_response_text)
        return json_output

    except Exception as e:
        print(f"Error in generate_trajectory_logic_api: {e}") # Log error server-side
        # Return an error structure suitable for Gradio's JSON output
        # Ensure the keys are strings
        return {"error": str(e), "raw_response_for_debugging": raw_response_text}


# --- Gradio UI Setup ---
with gr.Blocks() as demo:
    gr.Markdown("# Synthetic Patient Trajectory API 🏥")
    gr.Markdown("Test UI & API endpoint.")

    with gr.Row():
        diag_input = gr.Dropdown(label="Admission Diagnosis", choices=["Pneumonia", "Heart Failure Exacerbation", "Post-Op Hip Replacement", "Sepsis"])
        age_input = gr.Slider(label="Patient Age", minimum=18, maximum=100, value=65)

    comorbid_input = gr.CheckboxGroup(label="Comorbidities", choices=["Diabetes", "Hypertension", "COPD", "Smoker"])
    submit_btn = gr.Button("Generate Test Data")
    json_output_ui = gr.JSON(label="Generated JSON Output")

    # Connect the button to the API logic function
    submit_btn.click(
        fn=generate_trajectory_logic_api, # This function now handles client init
        inputs=[diag_input, age_input, comorbid_input],
        outputs=json_output_ui,
        api_name="predict" # Keep the API name for the endpoint path
    )

# --- Launch Gradio Server for Render ---
# Read the port assigned by Render (default 10000 if not set)
server_port = int(os.environ.get('PORT', 10000))
print(f"Attempting to launch Gradio on 0.0.0.0:{server_port}")

# Launch specifying host 0.0.0.0 and the port Render provides
demo.launch(server_name="0.0.0.0", server_port=server_port)
print("demo.launch() called.")