from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List
from tools import download_pdf, fetch_from_arxiv, format_arxiv_results, logger


class Agent:
    def __init__(self, index, llm_model):
        self.index = index
        self.llm_model = llm_model
        self.tools: List = []  # Explicit type hint for Python 3.12+
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Build all required tools in optimal order"""
        self.build_query_engine()
        self.build_rag_tool()
        self.build_pdf_downloader_tool()
        self.build_fetch_arxiv_tool()
        self.build_agent()

    def build_query_engine(self) -> None:
        """Create the core query engine with enhanced settings"""
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_model,  # Changed from llm_model to llm (LlamaIndex 0.10+)
            similarity_top_k=5,
            streaming=True,  # Enable for long responses
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )

    def build_rag_tool(self) -> None:
        """Enhanced RAG tool with metadata filtering"""
        self.rag_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="research_paper_engine",
            description=(
                "Access to the latest AI research papers. "
                "Use for questions about machine learning, LLMs, "
                "and technical concepts from academic papers."
            ),
            return_direct=False,  # Let agent decide when to use
        )
        self.tools.append(self.rag_tool)

    def build_pdf_downloader_tool(self) -> None:
        """Robust PDF downloader with auto-filename support"""
        self.pdf_tool = FunctionTool.from_defaults(
            fn=download_pdf,
            name="pdf_downloader",
            description=(
                "Downloads PDFs from arXiv URLs. "
                "Input MUST be a JSON string with keys:\n"
                "- 'pdf_url': Direct PDF URL (e.g., 'https://arxiv.org/pdf/1234.5678v1.pdf')\n"
                "- 'output_file_name': Optional filename (default: auto-generated from arXiv ID)\n\n"
                "Example inputs:\n"
                '{"pdf_url": "https://arxiv.org/pdf/1234.5678v1.pdf"}\n'
                '{"pdf_url": "https://arxiv.org/pdf/1234.5678v1.pdf", "output_file_name": "my_paper.pdf"}'
            ),
        )
        self.tools.append(self.pdf_tool)

    def build_fetch_arxiv_tool(self) -> None:
        """Enhanced ArXiv fetcher with full parameter support"""
        self.arxiv_tool = FunctionTool.from_defaults(
            fn=fetch_from_arxiv,
            name="arxiv_fetcher",
            description=(
                "Fetches papers from arXiv. Input format:\n"
                "{\n"
                '  "query": "search terms",\n'
                '  "max_results": 10, (1-1000)\n'
                '  "days_old": 30 (optional)\n'
                "}\n"
                "Returns complete paper metadata including:\n"
                "- title, summary, authors\n"
                "- pdf_url, arxiv_url\n"
                "- journal_ref, doi, categories"
            ),
        )
        self.tools.append(self.arxiv_tool)

    def build_agent(self) -> None:
        """Configure the ReAct agent with robust settings"""
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm_model,
            max_iterations=8,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            tool_retry_settings={
                "max_retries": 2,  # Prevent infinite retries
                "retry_delay": 1.0,
            },
            system_prompt=(
                "You are an expert research assistant. When fetching papers:\n"
                "1. Always format results clearly\n"
                "2. Show title, authors, and summary\n"
                "3. Include arXiv URL\n"
                "4. Stop after presenting results - don't keep iterating\n"
                "5. If no papers found, say so immediately"
            ),
        )

    def chat(self, user_input: str) -> str:
        try:
            # First check if this is a paper request
            if any(
                keyword in user_input.lower()
                for keyword in ["papers about", "find papers", "research on"]
            ):
                # Directly call arXiv fetcher with formatted query
                papers = fetch_from_arxiv(
                    query=user_input.replace("papers about", "").strip(), max_results=5
                )
                return format_arxiv_results(papers, user_input)

            # Otherwise use normal agent flow
            response = self.agent.chat(user_input)
            return str(response.response if hasattr(response, "response") else response)

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return "I encountered an error processing your request. Please try again."
