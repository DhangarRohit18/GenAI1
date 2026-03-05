import requests

OLLAMA_BASE_URL = 'http://localhost:11434'

def _ollama_is_running():
    try:
        r = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def _ollama_model_available(model_name):
    try:
        r = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=3)
        if r.status_code == 200:
            models = [m['name'].split(':')[0] for m in r.json().get('models', [])]
            return model_name.split(':')[0] in models
        return False
    except Exception:
        return False

class LocalLLMScratch:
    def __init__(self, model_id='mistral'):
        self.model_id = model_id
        self.backend = 'ollama' if _ollama_is_running() else 'transformers'
        self._hf_model = None
        self._hf_tokenizer = None
        self._device = 'cpu'

    @property
    def backend_info(self):
        if self.backend == 'ollama':
            return f'Ollama ({self.model_id})'
        return 'Local TinyLlama-1.1B (transformers fallback)'

    def set_model(self, model_id):
        self.model_id = model_id

    def _load_hf_model(self):
        if self._hf_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            fallback = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._hf_tokenizer = AutoTokenizer.from_pretrained(fallback)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                fallback,
                torch_dtype=torch.float16 if self._device == 'cuda' else torch.float32,
                device_map='auto'
            )

    def generate(self, prompt, max_new_tokens=512):
        if self.backend == 'ollama':
            return self._generate_ollama(prompt, max_new_tokens)
        return self._generate_transformers(prompt, max_new_tokens)

    def _generate_ollama(self, prompt, max_new_tokens=512):
        try:
            resp = requests.post(
                f'{OLLAMA_BASE_URL}/api/generate',
                json={
                    'model': self.model_id,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': max_new_tokens,
                    }
                },
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get('response', '').strip()
            elif resp.status_code == 404:
                return (
                    f'Model {self.model_id!r} not found in Ollama. '
                    f'Run:  ollama pull {self.model_id}'
                )
            return f'[Ollama error {resp.status_code}]: {resp.text[:300]}'
        except requests.exceptions.ConnectionError:
            self.backend = 'transformers'
            return self._generate_transformers(prompt, max_new_tokens)
        except Exception as e:
            return f'[Ollama error]: {e}'

    def _generate_transformers(self, prompt, max_new_tokens=256):
        import torch
        self._load_hf_model()
        inputs = self._hf_tokenizer(prompt, return_tensors='pt').to(self._device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._hf_tokenizer.eos_token_id
            )
        new_tokens = outputs[0][input_len:]
        return self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def get_summary(self, full_text):
        truncated = full_text[:3000]
        if self.backend == 'ollama':
            prompt = (
                'Summarize these study notes into a concise bullet-point study guide. '
                'Include key topics, definitions, and important concepts.\n\n'
                'Notes:\n' + truncated + '\n\nSummary:'
            )
        else:
            prompt = (
                'You are a study assistant. Summarize the notes concisely.\n'
                'Notes: ' + truncated + '\n\nSummary:'
            )
        return self.generate(prompt, max_new_tokens=512)
