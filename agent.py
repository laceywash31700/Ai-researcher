from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from typing import List
from tools import (
    download_pdf,
    fetch_from_arxiv,
    clean_text,
    extract_pdf_text,
    analyze_pdf_structure,
)


class Agent:
    def __init__(self,index_manager, llm_model):
        self.index_manager = index_manager
        self.index= self.index_manager.retrieve_index()
        self.llm_model = llm_model
        self.tools: List = []
        self.build_rag_tool()
        self._initialize_tools()
        self.build_agent()

    def _initialize_tools(self) -> None:
        """Build all required tools in optimal order"""
        self.build_pdf_downloader_tool()
        self.build_fetch_arxiv_tool()
        self.build_clean_text_tool()
        self.build_extract_pdf_text_tool()
        self.build_analyze_pdf_structure_tool()
        
        
        
    def build_rag_tool(self) -> None:
        """Enhanced RAG tool with metadata filtering"""
        self.rag_tool = QueryEngineTool.from_defaults(
            query_engine=self.index_manager.retrieve_index().as_query_engine(),
            name="research_paper_query_engine_tool",
            description="Use for questions about academic papers."
            )
        self.tools.append(self.rag_tool)
        
    def build_fetch_arxiv_tool(self) -> None:
        """Enhanced ArXiv fetcher with full parameter support"""
        def fetch_and_update_papers(query: str, max_results: int):
            papers = fetch_from_arxiv(query, max_results)
            self.index_manager.upsert_to_index(papers)
            return papers
            
        self.fetch_arxiv_tool = FunctionTool.from_defaults(
            fn=fetch_and_update_papers,
            name="arxiv_fetcher",
            description=(
                "Fetches papers from arXiv and updates Pinecone Vectors. Input format:\n"
                "{\n"
                '  "query": "search terms",\n'
                '  "max_results": 10, (1-1000)\n'
                "}\n"
                "Returns complete paper metadata including:\n"
                "- title, summary, authors\n"
                "- pdf_url, arxiv_url\n"
                "- journal_ref, doi, categories"
            )
        )
        self.tools.append(self.fetch_arxiv_tool)

    def build_pdf_downloader_tool(self):
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
            )
        )
        self.tools.append(self.pdf_tool)
        
    def build_extract_pdf_text_tool(self) -> None:
        """Extracts text from PDFs"""
        self.extract_pdf_text_tool = FunctionTool.from_defaults(
            fn=extract_pdf_text,
            name="pdf_text_extractor",
            description=(
            "Extracts raw text from PDF files. "
            "Input MUST be a JSON string with 'pdf_path' key. "
            "Example: {'pdf_path': '/path/to/file.pdf'}"
        )
        )
        self.tools.append(self.extract_pdf_text_tool)
    
    def build_analyze_pdf_structure_tool(self) -> None:
        """Analyzes PDF structure for identifying key sections"""
        self.analyze_pdf_structure_tool = FunctionTool.from_defaults(
            fn=analyze_pdf_structure,
            name="pdf_structure_analyzer",
            description="Analyzes PDF structure for identifying key sections."
        )
        self.tools.append(self.analyze_pdf_structure_tool)

  
        
    def build_clean_text_tool(self):
        """Text cleaning tool for removing noise and formatting"""
        self.clean_text_tool = FunctionTool.from_defaults(
            fn=clean_text,
            name="text_cleaner",
            description="Removes noise and formatting from text."
        )
        self.tools.append(self.clean_text_tool)

    def build_agent(self) -> None:
        """Configure the ReAct agent with robust settings"""
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm_model,
            max_iterations=8,
            verbose=True,
            tool_retry_settings={
                "max_retries": 2, 
                "retry_delay": 1.0,
            },
            system_prompt=(
                "You are an expert research assistant with these strict rules:\n"
        
                "1. When asked to fetch or find papers ALWAYS use the research_paper_query_engine_tool first if, you can't find any in there use the Fetch_from_arxiv_tool next\n"
                "2. For text extraction, ALWAYS use the pdf_text_extractor tool\n"
                "3. When presenting papers, include title, authors, summary and arXiv URL\n"
                "4. If a tool is available for a task, you MUST use it\n"
                "5. After using a tool, present the results immediately\n\n"
                "6. when asked to analyze a paper, ALWAYS use the pdf_structure_analyzer tool after you have downloaded the pdf and use any tool that can help you analyze that PDF\n"
                "Tool capabilities:\n"
                "- research_paper_query_engine_tool: Searches for research papers in index\n"
                "- text_cleaner: Removes noise and formatting from text\n"
                "- pdf_downloader: Downloads PDFs from URLs\n"
                "- pdf_text_extractor: Extracts raw text from PDFs\n"
                "- pdf_structure_analyzer: Identifies sections in PDFs\n"
                "- arxiv_fetcher: Finds research papers and adds them to index\n"
            ),
        )

    def chat(self, message: str) -> str:
        return self.agent.chat(message)
        

    def _format_response(self, response) -> str:
        """Format different response types consistently"""
        if isinstance(response, str):
            return response
            
        if hasattr(response, "response"):
            result = str(response.response)
            if hasattr(response, "sources"):
                sources = "\n".join(f"- {s}" for s in response.sources)
                result += f"\n\nSources:\n{sources}"
            return result
            
        return str(response)  
    
      
    def _format_error(self, error: Exception, message: str) -> str:
        """User-friendly error messages"""
        error_msg = str(error).lower()
         
        if "pdf" in error_msg or "download" in error_msg:
            return (
                "I encountered a PDF processing error. This could be because:\n"
                "1. The PDF is image-based/scanned\n"
                "2. The download was incomplete\n"
                "3. The file is corrupted\n\n"
                "Try checking the PDF status or downloading it again."
            )
            
        if "arxiv" in error_msg or "paper" in error_msg:
            return (
                "Research paper access failed. Possible reasons:\n"
                "1. arXiv service may be down\n"
                "2. The paper ID might be incorrect\n"
                "3. Network connectivity issues\n\n"
                "Please try again or verify the paper details."
            )
            
        return (
            "I encountered an error processing your request. "
            "The technical details have been logged. "
            "Please try rephrasing your question or try again later."
        )
        
