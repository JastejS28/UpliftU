import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch.amp import autocast
import logging
from typing import List, Dict, Optional
from functools import lru_cache
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthChatbot:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.max_tokens = 256  # Set consistent max token limit
        self.setup_model()
        
    def setup_model(self):
        try:
            # Optimize tokenizer settings with 256 token limit
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                truncation_side="left",
                model_max_length=self.max_tokens,  # Set to 256
                use_fast=True
            )
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            
            self.model.config.use_cache = True
            if hasattr(self.model, 'config'):
                self.model.config.pretraining_tp = 1
            
            if torch.cuda.is_available():
                self.model = torch.compile(self.model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @lru_cache(maxsize=128)
    def format_prompt(self, instruction: str, input_text: str) -> str:
        # Truncate instruction and input if needed to fit within token limit
        max_instruction_length = 50  # Limit instruction length
        max_input_length = 150  # Limit input length to leave room for response
        
        instruction = instruction[:max_instruction_length]
        input_text = input_text[:max_input_length]
        
        return f"""### Instruction:
You are a supportive mental health assistant who understands Indian culture, family dynamics, and societal pressures. {instruction}

### Input:
{input_text}

### Response:"""

    def generate_response(
        self,
        user_input: str,
        instruction: str = "Provide culturally sensitive mental health support."  # Shortened instruction
    ) -> str:
        try:
            formatted_prompt = self.format_prompt(instruction, user_input)
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens,  # Use class max_tokens
                padding=True,
                return_attention_mask=True
            ).to(self.model.device)
            
            # Calculate remaining tokens for generation
            input_length = inputs.input_ids.shape[1]
            max_new_tokens = min(self.max_tokens - input_length, 128)  # Ensure we don't exceed 256 total
            
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "length_penalty": 0.6,
                "use_cache": True,
                "return_dict_in_generate": False,
            }
            
            with autocast('cuda'):
                with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    generated_ids = self.model.generate(
                        **inputs,
                        **generation_config
                    )
            
            response = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            if "### Response:" in response:
                response = response.split("### Response:")[1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try rephrasing your message."

class ChatInterface:
    def __init__(self, chatbot: MentalHealthChatbot):
        self.chatbot = chatbot
        self.crisis_resources = {
            "AASRA 24/7 Helpline": "Call: 91-9820466726",
            "iCall Psychosocial Helpline": "Call: 022-25521111",
            "Vandrevala Foundation": "Call: 1860-2662-345",
            "NIMHANS": "Call: 080-46110007",
            "Sneha India": "Call: 044-24640050"
        }  # Shortened resource list for more concise responses
    
    @lru_cache(maxsize=32)
    def get_crisis_response(self) -> str:
        return "Here are some mental health helplines in India:\n\n" + \
               "\n".join([f"• {k}: {v}" for k, v in self.crisis_resources.items()])
    
    def chat_with_bot(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> tuple:
        if not history:
            history = [
                {"role": "assistant", "content": "Namaste! How can I assist you today?"}  # Shortened initial message
            ]
        
        if user_input.strip().lower() in ['help', 'crisis', 'emergency', 'helpline']:
            response = self.get_crisis_response()
        else:
            response = self.chatbot.generate_response(user_input)
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        return history, history

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="purple",
            ),
            analytics_enabled=False
        ) as demo:
            gr.Markdown("""
                <div style='text-align: center; padding: 1rem;'>
                    <h1>Mental Health Support Assistant</h1>
                    <p>Type 'help' for Indian mental health helplines</p>
                </div>
            """)
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                container=True,
                bubble_full_width=False,
                show_label=False,
                type="messages"
            )
            
            state = gr.State()
            
            with gr.Row():
                with gr.Column(scale=9):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Share your thoughts here...",
                        container=False
                    )
                with gr.Column(scale=1, min_width=80):
                    submit_btn = gr.Button("Send", variant="primary")
            
            gr.Examples(
                examples=[
                    "I'm feeling pressured about my marriage",
                    "My parents don't understand my career choices",
                    "help"
                ],
                inputs=txt
            )
            
            txt.submit(
                self.chat_with_bot,
                inputs=[txt, state],
                outputs=[chatbot, state]
            ).then(lambda: "", None, txt)
            
            submit_btn.click(
                self.chat_with_bot,
                inputs=[txt, state],
                outputs=[chatbot, state]
            ).then(lambda: "", None, txt)
            
        return demo

def main():
    model_name = "Aditya0619/Medical_Mistral"
    chatbot = MentalHealthChatbot(model_name)
    interface = ChatInterface(chatbot)
    demo = interface.create_interface()
    
    demo.queue(concurrency_count=3)
    
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        show_error=True,
        server_port=7860,
        height=800,
    )

if __name__ == "__main__":
    main()