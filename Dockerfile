FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

#install dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch transformers fastapi uvicorn accelerate

#load model
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
                AutoTokenizer.from_pretrained('bigscience/bloomz-1b1'); \
                AutoModelForCausalLM.from_pretrained('bigscience/bloomz-1b1')"

COPY ./app /app
WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]