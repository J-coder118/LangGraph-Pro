import os
from langchain.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_agent_executor
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
import operator
from langchain.prompts import PromptTemplate
from var import LANGGRAPH

langchain_api_key = ''
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"] ="multi-agent"

api_key = ""

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

code_pmt = """
You are professional coder.
This is the langGraph code sample for generating a chart of average temperature in alaska over the past decade.
code sample: {code}
from this code sample, give me only perfect and exact langGraph code to excute for implementing this task without any bug.
task: {task}"""


prompt_code = PromptTemplate(template=code_pmt, input_variables=["code", "task"])
llm_chain_code = LLMChain(prompt=prompt_code, llm=llm)


requirements = "create a langgraph that uses github to make a pr"

input_dict = {"code": LANGGRAPH, "task": requirements}
result = llm_chain_code.invoke(input_dict)
# print("reslut", result)
print("result----",  result["text"])