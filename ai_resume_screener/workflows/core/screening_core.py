from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
import qdrant_client
import os
import pickle
import os.path
import openai
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class ScreeningCore:
    def __init__(self, candidate_doc: str):
        # Using OpenAI GPT-4 for LLM
        llm = OpenAI(model="gpt-4", request_timeout=300)

        # Use OpenAI embeddings
        embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002")
        
        text_parser = SentenceSplitter(chunk_size=128, chunk_overlap=100)
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        self.candidate_summary = candidate_doc

        # Update settings with OpenAI models
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.transformations = [text_parser]
        Settings.callback_manager = callback_manager

        reader = SimpleDirectoryReader(input_files=[f"ai_resume_screener/data/{candidate_doc}"])
        self.docs = reader.load_data(show_progress=True)
        self.base_nodes = None
        self.node_mappings = None
        self.retriever = None
        self._pre_process()

    def _pre_process(self):
        node_parser = UnstructuredElementNodeParser()
        pickle_file = f"./{self.candidate_summary.rstrip('.pdf')}.pkl"
        if not os.path.exists(pickle_file):
            raw_nodes = node_parser.get_nodes_from_documents(self.docs)
            pickle.dump(raw_nodes, open(pickle_file, "wb"))
        else:
            raw_nodes= pickle.load(open(pickle_file, "rb"))

        self.base_nodes, self.node_mappings = node_parser.get_base_nodes_and_mappings(
            raw_nodes
        )
        self._index_in_vector_store()

    def _index_in_vector_store(self):
        # Create a local Qdrant vector store
       # client = qdrant_client.QdrantClient(url=os.getenv('QDRANT_URL'), port=6333, grpc_port=6333, api_key=os.getenv('QDRANT_API_KEY'))
        
                # Recreate the collection with the correct vector size
     #   client.create_collection(
     #       collection_name=f"{self.candidate_summary.strip('.pdf')}",
     #       vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
     #   )
        vector_store = QdrantVectorStore(
            url=os.getenv('QDRANT_URL'),  
            api_key=os.getenv('QDRANT_API_KEY'), 
            collection_name=f"{self.candidate_summary.strip('.pdf')}"
        )
        
        # Construct top-level vector index and query engine
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        vector_index = VectorStoreIndex(
            nodes=self.base_nodes,
            storage_context=storage_context,
            transformations=Settings.transformations,
            embed_model=Settings.embed_model
        )
        
        self.retriever = vector_index.as_retriever(similarity_top_k=5)

    def retriever_query_engine(self):
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": self.retriever},
            node_dict=self.node_mappings,
            verbose=True,
        )
        query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
        return query_engine