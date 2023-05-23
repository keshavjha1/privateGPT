from flask import Flask, request
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse

from QASingleton import QASingleton

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

app = Flask(__name__)
qa2 = QASingleton().qa
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>privateGPT</title>
    </head>
    <body>
        <h1>privateGPT</h1>
        <p>Ask questions to your documents without an internet connection, using the power of LLMs.</p>
        <form action="/query" method="post">
            <input type="text" name="query" placeholder="Enter a query">
            <input type="submit" value="Ask">
        </form>
    </body>
    </html>
    """

@app.route("/query", methods=["POST"])
def query():
    query = request.form["query"]
    if query == "exit":
        return "Goodbye!"
    args = parse_arguments()
    res = qa2(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)


    return answer


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    app.run(debug=True)