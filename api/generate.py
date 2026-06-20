from http.server import BaseHTTPRequestHandler
import json
from gradio_client import Client

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Read the incoming request data from frontend
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        prompt = data.get("prompt", "Hello!!")
        
        try:
            # 2. Initialize the Gradio client mapping to FLUX.1-dev
            client = Client("black-forest-labs/FLUX.1-dev")
            
            # 3. Trigger the '/infer' endpoint based on your API documentation
            result = client.predict(
                prompt=prompt,
                input_images=[],           # Empty list based on default/optional parameters
                seed=0,
                randomize_seed=True,
                width=1024,
                height=1024,
                num_inference_steps=28,
                guidance_scale=3.5,
                prompt_upsampling=True,
                api_name="/infer"          # Matches the active endpoint in your image
            )
            
            # The API returns a list/tuple where index 0 contains image file data
            # e.g., {'path': '...', 'url': 'https://...'}
            image_info = result[0]
            image_url = image_info.get("url") if isinstance(image_info, dict) else None

            # 4. Respond back to the frontend
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {"success": True, "image_url": image_url}
            self.wfile.write(json.dumps(response).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))