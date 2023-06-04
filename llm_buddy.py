import os

import streamlit as st 

from getpass import getpass
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

from flanpipeline import TextGenerationFlanPipeline
from transformers import T5ForConditionalGeneration, AutoTokenizer

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

pipe = TextGenerationFlanPipeline(model, tokenizer, framework='pt', task='text-generation')
hf = HuggingFacePipeline(pipeline=pipe)

template = """Question: {question}

            Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=hf)

st.title('Slow and Steady')

prompt = st.text_input('Enter your prompt here!')

if prompt: 
    
    response = llm_chain.run(prompt)
    st.write(response)
