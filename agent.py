from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from tools import download_pdf, fetch_from_arxiv

class Agent:
    def __init__(self, index, llm_model):
        self.index = index
        self.llm_model = llm_model
        self.build_query_engine()
        self.build_rag_tool()
        self.build_pdf_downloader_tool()
        self.build_fetch_arxiv_tool()
        self.build_agent()
    
    def build_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            llm_model=self.llm_model, 
            similarity_top_k=5
            )
    def build_rag_tool(self):
        self.rag_tool = QueryEngineTool.from_defaults(
            self.query_engine,
            name="research_paper_query_engine_tool",
            description="A RAG engine with recent research papers."
            )
        
    def build_pdf_downloader_tool(self):
        self.pdf_downloader_tool = FunctionTool.from_defaults(
            download_pdf,
            name="pdf_downloader_tool",
            description="A tool to download PDFs from given URLs.",
        )
    
    def build_fetch_arxiv_tool(self):
        self.fetch_arxiv_tool = FunctionTool.from_defaults(
            fetch_from_arxiv,
            name="fetch_arxiv_tool",
            description="A tool to fetch papers from ArXiv given their IDs.",
        )
    
    def build_agent(self):
        self.agent = ReActAgent.from_tools(
            [self.rag_tool, self.pdf_downloader_tool, self.fetch_arxiv_tool],
            llm_model=self.llm_model,
            max_iterations=12,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def chat(self, user_input: str):
        return self.agent.chat(user_input)
        