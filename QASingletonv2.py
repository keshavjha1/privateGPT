from dotenv import load_dotenv
import argparse
import os
import qdrant_client

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma, Qdrant
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from CustomChains import StuffQA
import qdrant_client

from load_env import get_embedding_model
from utils import prompt_HTML, print_HTML

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
chain_type = os.environ.get('CHAIN_TYPE')

model_max_tokens = int(os.environ.get("MODEL_MAX_TOKENS"))
model_temp = float(os.environ.get("MODEL_TEMP", "0.8"))
model_stop = os.environ.get("MODEL_STOP", "")
n_retrieve_documents = os.environ.get('N_RETRIEVE_DOCUMENTS')
n_forward_documents = os.environ.get('N_FORWARD_DOCUMENTS')
from constants import CHROMA_SETTINGS


class QASingleton:
    """Singleton class that provides qa as one property."""

    def __init__(self):
        """Property that returns the qa object."""
        args = parse_arguments()

        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
       # db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
       # retriever = db.as_retriever()

        self.qdrant_client = qdrant_client.QdrantClient(path=persist_directory, prefer_grpc=True)
        self.qdrant_langchain = Qdrant(client=self.qdrant_client, collection_name="test", embeddings=embeddings)


        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [StreamingStdOutCallbackHandler()]
        # Prepare the LLM
        print(model_path)
        llm = None
        match model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=model_path,
                               n_ctx=model_n_ctx,
                               temperature=model_temp,
                               callbacks=callbacks,
                               verbose=True,
                               n_threads=6,
                               n_batch=1000,
                               max_tokens=model_max_tokens,
                               echo=True
                               )
                object.__setattr__(llm, "get_num_tokens",
                                   lambda text: len(llm.client.tokenize(b" " + text.encode("utf-8"))))
            case "GPT4All":
                llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            case _default:
                print(f"Model {model_type} not supported!")
                exit;

        retriever = self.qdrant_langchain.as_retriever(search_type="mmr")
        if chain_type == "betterstuff":
            self.qa = StuffQA(retriever=retriever, llm=llm)
        else:
            self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                             return_source_documents=not args.hide_source)

        self.qa.retriever.search_kwargs = {**self.qa.retriever.search_kwargs, "k": n_forward_documents,
                                           "fetch_k": n_retrieve_documents}

        def prompt_once(self, query: str) -> tuple[str, str]:
            """run a prompt"""
            # Get the answer from the chain
            res = self.qa(query)
            answer, docs = res["result"], res["source_documents"]

            return answer

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

def main() -> None:
    session = PromptSession(auto_suggest=AutoSuggestFromHistory())
    qa_system = QASingleton(get_embedding_model()[0], persist_directory, model_path, model_n_ctx, model_temp, model_stop, use_mlock, n_gpu_layers)
    while True:
        query = prompt_HTML(session, "\n<b>Enter a query</b>: ").strip()
        if query == "exit":
            break
        elif not query:  # check if query empty
            print_HTML("<r>Empty query, skipping</r>")
            continue
        qa_system.prompt_once(query)


if __name__ == "__main__":
    main()
