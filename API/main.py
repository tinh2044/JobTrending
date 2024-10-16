from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
# model = AutoModelForSeq2SeqLM.from_pretrained("tinh2312/MBart-salary-pred")
# Create a FastAPI instance
app = FastAPI()


# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    prompt: str


# Define the POST endpoint
@app.post("/echo")
async def echo_text(request: TextRequest):
    info = request.prompt
    input_ids = tokenizer.encode(info, return_tensors="pt").to("cpu")
    # #
    # output_ids = model.generate(input_ids, max_length=32, num_beams=6, early_stopping=True)
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"output": info}


# Run the application (if needed)
if __name__ == "__main__":

    uvicorn.run(app, host='127.0.0.1', port=5000, reload=True)
