from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# grade prompt template
from main import ss

prompt_template  = PromptTemplate(
    template="""You are a expert in translating text from Sanskrit to English. Translate the following Sanskrit sentence into English: {sentence} to English using the context {context}. 
    The given text is in IAST format.""",
    input_variables=["sentence", "context"]
)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY']= "xxxxxxxxxxx"
llm_AIo = ChatOpenAI(model="gpt-4o", temperature=0)

chain = prompt_template | llm_AIo | StrOutputParser()
sentence = "dhrtarastra uvaca dharmaksetre kuruksetre samaveta yuyutsavah mamakah pandavas caiva kim akurvata Samjaya "
context = ss(sentence)
response = chain.invoke({"sentence":sentence,"context":context})
print(response)