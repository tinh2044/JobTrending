from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import pipeline, AutoTokenizer

# from optimum.onnxruntime import  ORTModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
# model = ORTModelForSeq2SeqLM.from_pretrained("tinh2312/MBart-salary-pred")
# Create a FastAPI instance
app = FastAPI()
model_id = "tinh2312/Bart-salary-pred-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_id)

generator = pipeline(task="text2text-generation", model=ort_model, tokenizer=tokenizer)


# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    prompt: str


# Define the POST endpoint
@app.post("/pred_salary")
async def echo_text(request: TextRequest):
    info = request.prompt
    output = generator(info)[0]["generated_text"]
    return {"output": output}


# Run the application (if needed)
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000, reload=True)
