from http.server import BaseHTTPRequestHandler
import json
import os
import tempfile
import base64
from gradio_client import Client, handle_file

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "status": "online", 
            "message": "The Flux 2 API endpoint is configured! Send a POST request with parameters."
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        prompt = data.get("prompt", "Hello!!")
        base64_image = data.get("image")
        filename = data.get("filename", "input.png")
        
        input_images_payload = []
        temp_file_path = None

        try:
            if base64_image:
                image_data = base64.b64decode(base64_image)
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, filename)
                with open(temp_file_path, "wb") as f:
                    f.write(image_data)
                
                input_images_payload = [{"image": handle_file(temp_file_path), "caption": None}]

            hf_token = os.environ.get("HF_TOKEN")

            # 👇 FIXED: Changed argument from hf_token to token
            client = Client("black-forest-labs/FLUX.2-dev", token=hf_token)
            
            result = client.predict(
                prompt=prompt,
                input_images=input_images_payload,
                seed=0,
                randomize_seed=True,
                width=1024,
                height=1024,
                num_inference_steps=30, 
                guidance_scale=4,      
                prompt_upsampling=True,
                api_name="/infer"          
            )
            
            image_info = result[0]
            image_url = image_info.get("url") if isinstance(image_info, dict) else None

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
            
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass