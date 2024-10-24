import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch.amp import autocast
import logging
from typing import List, Dict, Optional
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthChatbot:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.setup_model()
        
    def setup_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                truncation_side="left",
                model_max_length=512
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
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def format_prompt(self, instruction: str, input_text: str) -> str:
        """Format the prompt in Alpaca style with Indian context"""
        return f"""### Instruction:
You are a supportive mental health assistant who understands Indian culture, family dynamics, and societal pressures. {instruction}

### Input:
{input_text}

### Response:"""

    def generate_response(
        self,
        user_input: str,
        instruction: str = "Provide culturally sensitive mental health support while maintaining empathy and understanding of Indian values and family dynamics. Consider social, cultural, and familial contexts in your responses."
    ) -> str:
        """Generate a response using the model with Indian context"""
        try:
            formatted_prompt = self.format_prompt(instruction, user_input)
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.model.device)
            
            with autocast('cuda'):
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_beams=2,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=0.6
                    )
            
            response = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
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
            "iCall Psychosocial Helpline": "Call: 022-25521111 (Monday to Saturday, 8:00 AM to 10:00 PM)",
            "Vandrevala Foundation": "Call: 1860-2662-345 or 1800-2333-330 (24/7)",
            "NIMHANS": "Call: 080-46110007 (24/7)",
            "Sneha India": "Call: 044-24640050 (24/7)",
            "Parivarthan": "Call: +91-7676602602 (Monday to Friday, 4:00 PM to 10:00 PM)",
            "Mann Talks": "Call: +91-8686139139"
        }
        
    def chat_with_bot(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> tuple:
        """Handle chat interaction with Indian context"""
        if not history:
            history = [
                {"role": "assistant", "content": "Namaste! I'm here to support you. I understand the unique challenges we face in Indian society and families. How can I assist you today?"}
            ]
        
        if user_input.strip().lower() in ['help', 'crisis', 'emergency', 'helpline']:
            response = "Here are some mental health helplines in India that you can reach out to:\n\n" + \
                      "\n".join([f"• {k}: {v}" for k, v in self.crisis_resources.items()]) + \
                      "\n\nThese helplines maintain confidentiality and provide support in multiple Indian languages."
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
                    <h1 style='color: #4A90E2;'>Mental Health Support Assistant</h1>
                    <h2 style='color: #666;'>Powered by UpliftU</h2>
                    <p style='color: #666; margin-bottom: 1rem;'>
                        You can speak to me openly about your concerns. Understand that i am a AI chatbot so my responses can be wrong, I am still learning
                    </p>
                    <p style='color: #888; font-size: 0.9rem;'>
                        Type 'help' for Indian mental health helplines and resources
                    </p>
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
                    submit_btn = gr.Button(
                        "Send",
                        variant="primary"
                    )
            
            gr.Examples(
                examples=[
                    "I'm feeling pressured about my marriage",
                    "My parents don't understand my career choices",
                    "I'm stressed about competitive exams",
                    "I'm worried about what society will think",
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
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        show_error=True,
        server_port=7860,
        height=800,
    )
if __name__ == "__main__":
    main()
