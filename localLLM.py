from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os

load_dotenv()

llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS



class QASystem:
    """custom QA system"""

    def __init__(self):

        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever()
        callbacks = [StreamingStdOutCallbackHandler()]
        # Prepare the LLM chain
        match model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=True)
            case "GPT4All":
                llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=True)
            case _default:
                print(f"Model {model_type} not supported!")
                exit;


        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    def prompt_once(self, query: str):
        """run a prompt"""
        # Get the answer from the chain
        res = self.qa(query)
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

        return res

def main():
    # Interactive questions and answers
    qa_system = QASystem()
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa_system.prompt_once(query)


if __name__ == "__main__":
    main()
