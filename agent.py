from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List
import re
from tools import (
    download_pdf,
    fetch_from_arxiv,
    format_arxiv_results,
    clean_text,
    extract_pdf_text,
    analyze_pdf_structure,
    logger,
)


class Agent:
    def __init__(self, index, llm_model):
        self.index = index
        self.llm_model = llm_model
        self.tools: List = [] 
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Build all required tools in optimal order"""
        self.build_query_engine()
        self.build_rag_tool()
        self.build_pdf_downloader_tool()
        self.build_fetch_arxiv_tool()
        self.build_clean_text_tool()
        self.build_extract_pdf_text_tool()
        self.build_analyze_pdf_structure_tool()
        self.build_agent()

    def build_query_engine(self) -> None:
        """Create the core query engine with enhanced settings"""
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_model,
            similarity_top_k=5,
            streaming=True,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )
        
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

    def build_rag_tool(self) -> None:
        """Enhanced RAG tool with metadata filtering"""
        self.rag_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="research_paper_query_engine_tool",
            description="Use for questions about academic papers."
            )
        self.tools.append(self.rag_tool)

   
        
    def build_clean_text_tool(self):
        """Text cleaning tool for removing noise and formatting"""
        self.clean_text_tool = FunctionTool.from_defaults(
            fn=clean_text,
            name="text_cleaner",
            description="Removes noise and formatting from text."
        )
        self.tools.append(self.clean_text_tool)

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
                "}\n"
                "Returns complete paper metadata including:\n"
                "- title, summary, authors\n"
                "- pdf_url, arxiv_url\n"
                "- journal_ref, doi, categories"
            )
        )
        self.tools.append(self.arxiv_tool)

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
                "max_retries": 2, 
                "retry_delay": 1.0,
            },
            system_prompt=(
                "You are an expert research assistant with these strict rules:\n"
                "1. When asked to analyze a paper, ALWAYS use the pdf_structure_analyzer tool first\n"
                "2. For text extraction, ALWAYS use the pdf_text_extractor tool\n"
                "3. When presenting papers, include title, authors, summary and arXiv URL\n"
                "4. If a tool is available for a task, you MUST use it\n"
                "5. After using a tool, present the results immediately\n\n"
                "Tool capabilities:\n"
                "- pdf_downloader: Downloads PDFs from URLs\n"
                "- pdf_text_extractor: Extracts raw text from PDFs\n"
                "- pdf_structure_analyzer: Identifies sections in PDFs\n"
                "- arxiv_fetcher: Finds research papers\n"
            ),
        )

    def chat(self, message: str) -> str:
        try:
            # Handle direct paper searches
            if any(keyword in message.lower() for keyword in ["papers about", "find papers", "research on"]):
                papers = fetch_from_arxiv(
                    query=message.replace("papers about", "").strip(), 
                    max_results=5
                )
                return format_arxiv_results(papers, message)
            
            # Handle PDF analysis requests
            if "analyze the paper" in message.lower() or "analyze it" in message.lower():
                # Default path - you might want to make this configurable
                pdf_path = "C:\\Users\\Owner\\Desktop\\Ai-researcher\\storage\\papers\\Quantum_Computing_Paper_1.pdf"
                
                # First extract text
                extracted_text = extract_pdf_text(pdf_path)
                
                # Then analyze structure
                analysis = analyze_pdf_structure(pdf_path)
                
                return f"Text extracted successfully. Analysis results:\n{analysis}"
                
            # For all other queries, use the agent
            response = self.agent.chat(message)
            
            # Handle the response format consistently
            if hasattr(response, 'response'):
                return str(response.response)
            if hasattr(response, 'sources'):
                return f"{response}\n\nSources: {response.sources}"
            return str(response)

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return self._format_error(e, message)
        
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
        
        
    def _handle_pdf_status_query(self, query: str) -> str:
        """Handle PDF status inquiries"""
        try:
            arxiv_id = None
            if "arxiv:" in query:
                arxiv_id = query.split("arxiv:")[1].strip().split()[0]
            else:             
                match = re.search(r'\d{4}\.\d{4,5}', query)
                if match:
                    arxiv_id = match.group(0)
            
            if not arxiv_id:
                return "Please specify an arXiv ID (e.g., 'status of arXiv:1234.5678')"
                
            status = self.pdf_status_tool.fn(arxiv_id)
            if not status["exists"]:
                return f"PDF for arXiv:{arxiv_id} not downloaded yet. Use the pdf_downloader tool first."
                
            if status["error"]:
                return f"PDF error: {status['error']}"
                
            return (
                f"PDF Status for arXiv:{arxiv_id}:\n"
                f"- Pages: {status.get('pages', 'unknown')}\n"
                f"- Readable: {'Yes' if status['readable'] else 'No (may be scanned)'}\n"
                f"- Size: {status['size']:,} bytes"
            )
        except Exception as e:
            return f"Failed to check PDF status: {str(e)}"
        
    def _handle_paper_query(self, query: str) -> str:
        """Special handling for paper search requests"""
        # Extract clean search terms
        search_terms = query.replace("papers about", "").replace("find papers on", "").strip()
        
        try:
            papers = fetch_from_arxiv(query=search_terms, max_results=5)
            if not papers:
                return "No papers found matching your query. Please try different search terms."
                
            # Format with download status
            return format_arxiv_results(
                papers, 
                search_terms,
                pdf_dir=self.pdf_dir
            )
        except Exception as e:
            return f"Failed to search arXiv: {str(e)}. Please try again later."
   
