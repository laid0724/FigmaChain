import os
import webbrowser
import argparse
from langchain.document_loaders.figma import FigmaFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FigmaFileLoader with environment variables
figma_loader = FigmaFileLoader(
    os.environ.get("ACCESS_TOKEN"),
    os.environ.get("NODE_IDS"),
    os.environ.get("FILE_KEY"),
)

# Create an index and retriever for Figma documents
index = VectorstoreIndexCreator().from_loaders([figma_loader])
figma_doc_retreiver = index.vectorstore.as_retriever()

# Define system and human prompt templates
user_prompt = input("Enter your prompt: ")

code_sample = """
ADD CODE HERE
"""

system_prompt_template = """Act as senior web developer who is an expert in the Angular framework.
Use the provided design context to create idiomatic HTML/SCSS/TypeScript code based on the user request.
Everything must be output in one file.
Write code that matches the Figma file nodes and metadata as exactly as you can, with the request's specifications in mind.
Figma file nodes and metadata: {context}"""

human_prompt_template = 'Code this request: "{user_prompt}" using the following code sample as its basis "{code_sample}": . Ensure that the code is mobile responsive and follows modern design principles.'

# Initialize the ChatOpenAI model
gpt_model = ChatOpenAI(
    temperature=0.03, model_name="gpt-3.5-turbo", request_timeout=120
)


def generate_code(input_text, code_sample):
    # Get relevant nodes from the Figma document retriever
    relevant_nodes = figma_doc_retreiver.get_relevant_documents(input_text)

    # Create system and human message prompts
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_prompt_template
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_prompt_template
    )

    # Create the chat prompt using the system and human message prompts
    conversation = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(conversation)
    response = gpt_model(
        chat_prompt.format_prompt(
            context=relevant_nodes, user_prompt=input_text, code_sample=code_sample
        ).to_messages()
    )

    return response.content


# Generate the code using the input prompt and code sample
response = generate_code(user_prompt, code_sample)

# Define the output file name
file_name = "output.txt"

# Save the generated code to the output file
with open(file_name, "w") as file:
    file.write(response)
    print(f"Output file saved as {file_name}")
