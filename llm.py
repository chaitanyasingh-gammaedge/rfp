import requests
import json
import time

class LLM:
    def __init__(self, api_key: str, model_name="gemini-1-advanced", api_url="https://gemini.googleapis.com/v1/generate"):
        """
        Initializes the LLM model for interaction with Gemini API.
        Args:
        - api_key: Your Google Cloud API Key to access Gemini models.
        - model_name: The Gemini model you wish to use (default: "gemini-1-advanced").
        - api_url: The URL for the Gemini text generation endpoint (replace with the actual endpoint if different).
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url

        print(f"[LLM] Using model: {model_name} via API...")

    def generate(self, prompt: str, max_length=512, temperature=0.9, top_p=0.95, num_beams=5):
        """
        Generates a response using the Gemini API based on the provided prompt.
        Args:
        - prompt: The input text to generate a response from.
        - max_length: Maximum number of tokens to generate (default: 512).
        - temperature: Controls randomness in the generation (default: 0.9).
        - top_p: Top-p sampling (default: 0.95).
        - num_beams: Beam search parameter (default: 5).
        """
        prompt = prompt.strip()
        if len(prompt) < 5:
            return "[Error: Empty or invalid prompt.]"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "num_beams": num_beams
            }

            # Debug: Log the payload to be sent to Gemini API
            print(f"Sending payload to Gemini API: {json.dumps(payload, indent=2)}")

            # Send POST request to Gemini API with retry mechanism
            retries = 3
            for _ in range(retries):
                response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    break
                else:
                    print(f"Error during API call, retrying... Status Code: {response.status_code}")
                    time.sleep(5)  # Retry after 5 seconds
            else:
                return f"[Error: {response.status_code} - {response.text}]"

            # Log the response for debugging
            print(f"Response received: {response.json()}")

            # Extract response text
            generated_text = response.json().get("generated_text", "").strip()

            # Check if the output is reasonable, else return an error message
            if not generated_text or generated_text.lower() in ["nn", "n", ""]:
                return "[LLM returned empty or meaningless output — try a clearer or longer prompt.]"

            return generated_text

        except Exception as e:
            return f"[Error during generation: {e}]"
