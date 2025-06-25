import requests
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

class TreatmentType(Enum):
    OVERRULED = "overruled"
    REVERSED = "reversed"
    VACATED = "vacated"
    SUPERSEDED = "superseded"
    UNCONSTITUTIONAL = "unconstitutional"
    CRITICIZED = "criticized"
    DISTINGUISHED = "distinguished"
    LIMITED = "limited"
    NONE = "none"

class CaseTreatment(BaseModel):
    """Structured output for case treatment analysis"""
    case_name: str = Field(description="Name of the referenced legal case")
    case_citation: str = Field(description="Legal citation as it appears in the opinion")
    negative_treatment: bool = Field(description="Whether negative treatment is present")
    treatment_types: List[TreatmentType] = Field(description="Specific types of negative treatment found")
    relevant_excerpts: List[str] = Field(description="Sentences mentioning this case")
    confidence_score: float = Field(description="Confidence in the analysis (0-1)")
    explanation: str = Field(description="Detailed explanation of the treatment determination")

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    case_treatments: List[CaseTreatment]
    total_cases_found: int
    analysis_summary: str
    processing_time: float

class AnalysisState(TypedDict):
    """State for the LangGraph workflow"""
    opinion_text: str
    slug_id: str
    extracted_cases: List[Dict]
    case_treatments: List[CaseTreatment]
    confidence_scores: List[float]
    needs_human_review: bool
    error_messages: List[str]

class CaseExtractionAgent:
    """Specialized agent for extracting case citations from legal opinions"""
    
    def __init__(self, llm):
        self.tools = [
            Tool(
                name="extract_citations",
                description="Extract legal case citations using pattern matching",
                func=self._extract_citations_pattern
            ),
            Tool(
                name="validate_citation_format",
                description="Validate if a citation follows proper legal format",
                func=self._validate_citation_format
            )
        ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal citation expert specializing in extracting case references 
            from legal opinions. Your task is to:
            
            1. Identify ALL case citations in the text
            2. Extract the full case name and citation
            3. Find all sentences that mention each case
            4. Distinguish between different citation formats (parallel citations, short forms, etc.)
            
            Use your tools to systematically extract and validate citations.
            
            Return a JSON array with this structure:
            [
                {
                    "case_name": "Smith v. Jones",
                    "citation": "123 F.3d 456 (5th Cir. 2000)",
                    "mentions": ["First sentence mentioning Smith.", "Second sentence about Smith v. Jones."],
                    "confidence": 0.95
                }
            ]
            """),
            ("human", "Extract all case citations from this legal opinion:\n\n{opinion_text}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            return_intermediate_steps=True
        )
    
    def extract_cases(self, opinion_text: str) -> List[Dict]:
        """Extract all case citations from the opinion"""
        try:
            result = self.agent_executor.invoke({"opinion_text": opinion_text})
            return json.loads(result["output"])
        except Exception as e:
            print(f"Error in case extraction: {e}")
            return []
    
    def _extract_citations_pattern(self, text: str) -> str:
        """Tool to extract citations using regex patterns"""
        import re
        
        # Common legal citation patterns
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+v\.\s+(\w+(?:\s+\w+)*),?\s+(\d+\s+[A-Za-z\.]+\s+\d+)',  # Basic v. pattern
            r'(\w+(?:\s+\w+)*)\s+v\.\s+(\w+(?:\s+\w+)*),?\s+(\d+\s+[A-Za-z\.]+\d*\s+\d+)',  # With court
            r'In re\s+(\w+(?:\s+\w+)*),?\s+(\d+\s+[A-Za-z\.]+\s+\d+)',  # In re cases
        ]
        
        citations_found = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citations_found.append(match.group(0))
        
        return f"Found {len(citations_found)} potential citations: {citations_found[:10]}"  # Limit output
    
    def _validate_citation_format(self, citation: str) -> str:
        """Tool to validate citation format"""
        import re
        
        # Check for common citation elements
        has_volume = bool(re.search(r'\d+', citation))
        has_reporter = bool(re.search(r'[A-Za-z\.]+', citation))
        has_page = bool(re.search(r'\d+', citation))
        has_vs = 'v.' in citation.lower() or 'vs.' in citation.lower()
        
        confidence = sum([has_volume, has_reporter, has_page, has_vs]) / 4
        
        return f"Citation validation - Volume: {has_volume}, Reporter: {has_reporter}, Page: {has_page}, V.: {has_vs}, Confidence: {confidence:.2f}"

class TreatmentAnalysisAgent:
    """Specialized agent for analyzing how cases are treated in the opinion"""
    
    def __init__(self, llm):
        self.tools = [
            Tool(
                name="analyze_negative_language",
                description="Analyze text for negative treatment language patterns",
                func=self._analyze_negative_language
            ),
            Tool(
                name="classify_treatment_type",
                description="Classify the specific type of negative treatment",
                func=self._classify_treatment_type
            ),
            Tool(
                name="extract_relevant_context",
                description="Extract contextual sentences around case mentions",
                func=self._extract_relevant_context
            )
        ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal analysis expert specializing in case treatment analysis.
            Your expertise is in determining how legal opinions treat previously decided cases.
            
            For each case provided, analyze:
            1. Whether the opinion gives negative treatment to the case
            2. The specific type of negative treatment (overruled, criticized, etc.)
            3. The strength and clarity of the treatment
            4. Any nuances or complexities in the treatment
            
            Negative treatment includes:
            - STRONG: overruled, reversed, vacated, superseded, unconstitutional
            - MODERATE: criticized, distinguished, limited in scope
            
            Use your tools to systematically analyze the treatment.
            
            Return analysis in this JSON format:
            {
                "negative_treatment": true/false,
                "treatment_types": ["overruled", "criticized"],
                "confidence": 0.85,
                "explanation": "Detailed explanation...",
                "supporting_excerpts": ["Quote 1", "Quote 2"]
            }
            """),
            ("human", """Analyze the treatment of this case:
            Case: {case_name}
            Citation: {case_citation}
            Context sentences: {mentions}
            
            Determine if negative treatment is present and explain your reasoning."""),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def analyze_treatment(self, case_name: str, citation: str, mentions: List[str]) -> Dict:
        """Analyze how a specific case is treated"""
        try:
            result = self.agent_executor.invoke({
                "case_name": case_name,
                "case_citation": citation,
                "mentions": " | ".join(mentions)
            })
            return json.loads(result["output"])
        except Exception as e:
            print(f"Error analyzing treatment for {case_name}: {e}")
            return {
                "negative_treatment": False,
                "treatment_types": [],
                "confidence": 0.0,
                "explanation": f"Analysis failed: {str(e)}",
                "supporting_excerpts": []
            }
    
    def _analyze_negative_language(self, text: str) -> str:
        """Tool to detect negative treatment language"""
        negative_terms = {
            "strong": ["overrul", "revers", "vacat", "supersed", "unconstitutional", "invalid"],
            "moderate": ["critic", "distinguish", "limit", "narrow", "restrict", "question"],
            "neutral": ["follow", "reaffirm", "confirm", "consistent", "accord"]
        }
        
        text_lower = text.lower()
        strong_count = sum(1 for term in negative_terms["strong"] if term in text_lower)
        moderate_count = sum(1 for term in negative_terms["moderate"] if term in text_lower)
        neutral_count = sum(1 for term in negative_terms["neutral"] if term in text_lower)
        
        return f"Language analysis - Strong negative: {strong_count}, Moderate negative: {moderate_count}, Neutral: {neutral_count}"
    
    def _classify_treatment_type(self, text: str) -> str:
        """Tool to classify specific treatment types"""
        classifications = []
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["overrul", "overrid"]):
            classifications.append("overruled")
        if any(term in text_lower for term in ["revers", "vacat"]):
            classifications.append("reversed")
        if "distinguish" in text_lower:
            classifications.append("distinguished")
        if any(term in text_lower for term in ["critic", "question"]):
            classifications.append("criticized")
        
        return f"Treatment types identified: {classifications}"
    
    def _extract_relevant_context(self, text: str) -> str:
        """Tool to extract contextual information around case mentions"""
        sentences = text.split('.')
        relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return f"Found {len(relevant_sentences)} contextual sentences for analysis"

class NegativeTreatmentWorkflow:
    """LangGraph workflow orchestrating the entire analysis"""
    
    def __init__(self, llm):
        self.llm = llm
        self.case_extraction_agent = CaseExtractionAgent(llm)
        self.treatment_analysis_agent = TreatmentAnalysisAgent(llm)
        
        # Create workflow
        self.workflow = StateGraph(AnalysisState)
        
        # Add nodes
        self.workflow.add_node("extract_cases", self.extract_cases_node)
        self.workflow.add_node("analyze_treatments", self.analyze_treatments_node)
        self.workflow.add_node("quality_check", self.quality_check_node)
        self.workflow.add_node("finalize_results", self.finalize_results_node)
        
        # Define edges
        self.workflow.set_entry_point("extract_cases")
        self.workflow.add_edge("extract_cases", "analyze_treatments")
        self.workflow.add_edge("analyze_treatments", "quality_check")
        self.workflow.add_conditional_edges(
            "quality_check",
            self.should_review,
            {
                "human_review": "finalize_results",
                "complete": "finalize_results"
            }
        )
        self.workflow.add_edge("finalize_results", END)
        
        self.app = self.workflow.compile()
    
    def extract_cases_node(self, state: AnalysisState) -> AnalysisState:
        """Extract all case citations from the opinion"""
        print("Extracting case citations...")
        
        try:
            extracted_cases = self.case_extraction_agent.extract_cases(state["opinion_text"])
            state["extracted_cases"] = extracted_cases
            print(f"Found {len(extracted_cases)} cases")
        except Exception as e:
            state["error_messages"].append(f"Case extraction failed: {str(e)}")
            state["extracted_cases"] = []
        
        return state
    
    def analyze_treatments_node(self, state: AnalysisState) -> AnalysisState:
        """Analyze treatment for each extracted case"""
        print("Analyzing case treatments...")
        
        case_treatments = []
        confidence_scores = []
        
        for case_data in state["extracted_cases"]:
            try:
                treatment_analysis = self.treatment_analysis_agent.analyze_treatment(
                    case_data["case_name"],
                    case_data["citation"],
                    case_data["mentions"]
                )
                
                case_treatment = CaseTreatment(
                    case_name=case_data["case_name"],
                    case_citation=case_data["citation"],
                    negative_treatment=treatment_analysis["negative_treatment"],
                    treatment_types=[TreatmentType(t) for t in treatment_analysis["treatment_types"]],
                    relevant_excerpts=case_data["mentions"],
                    confidence_score=treatment_analysis["confidence"],
                    explanation=treatment_analysis["explanation"]
                )
                
                case_treatments.append(case_treatment)
                confidence_scores.append(treatment_analysis["confidence"])
                
            except Exception as e:
                state["error_messages"].append(f"Treatment analysis failed for {case_data.get('case_name', 'unknown')}: {str(e)}")
        
        state["case_treatments"] = case_treatments
        state["confidence_scores"] = confidence_scores
        
        return state
    
    def quality_check_node(self, state: AnalysisState) -> AnalysisState:
        """Check analysis quality and flag for human review if needed"""
        print("Performing quality check...")
        
        avg_confidence = sum(state["confidence_scores"]) / len(state["confidence_scores"]) if state["confidence_scores"] else 0
        has_errors = len(state["error_messages"]) > 0
        complex_cases = sum(1 for ct in state["case_treatments"] if len(ct.treatment_types) > 2)
        
        # Flag for human review if confidence is low or there are complex cases
        state["needs_human_review"] = (
            avg_confidence < 0.7 or 
            has_errors or 
            complex_cases > len(state["case_treatments"]) * 0.3
        )
        
        print(f"Quality check: Avg confidence {avg_confidence:.2f}, Errors: {has_errors}, Complex cases: {complex_cases}")
        
        return state
    
    def should_review(self, state: AnalysisState) -> str:
        """Decide if human review is needed"""
        if state["needs_human_review"]:
            print("Flagging for human review due to quality concerns")
            return "human_review"
        else:
            print("Analysis quality sufficient, proceeding automatically")
            return "complete"
    
    def finalize_results_node(self, state: AnalysisState) -> AnalysisState:
        """Finalize and format results"""
        print("Finalizing results...")
        
        # Generate summary
        total_cases = len(state["case_treatments"])
        negative_cases = sum(1 for ct in state["case_treatments"] if ct.negative_treatment)
        
        summary = f"Analysis complete: {total_cases} cases found, {negative_cases} with negative treatment"
        if state["needs_human_review"]:
            summary += " (flagged for human review)"
        
        print(summary)
        return state

class CasetextAPI:
    """Enhanced API client with error handling and caching"""
    
    @staticmethod
    def fetch_opinion(slug: str) -> Optional[str]:
        """Fetch opinion text from Casetext API with error handling"""
        try:
            url = f'https://casetext.com/api/search-api/doc/{slug}/html'
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            opinion_text = soup.get_text()
            
            if len(opinion_text.strip()) < 100:
                raise ValueError("Opinion text too short - possible API error")
            
            return opinion_text
            
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing opinion: {e}")
            return None

def extract_negative_treatments_agentic(slug: str) -> Optional[AnalysisResult]:
    """Main function using agentic approach"""
    import time
    start_time = time.time()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,  # Low temperature for consistent legal analysis
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Fetch opinion text
    print(f"Fetching opinion for slug: {slug}")
    opinion_text = CasetextAPI.fetch_opinion(slug)
    
    if not opinion_text:
        print("Failed to fetch opinion text")
        return None
    
    print(f"Opinion fetched: {len(opinion_text)} characters")
    
    # Initialize workflow
    workflow = NegativeTreatmentWorkflow(llm)
    
    # Create initial state
    initial_state = AnalysisState(
        opinion_text=opinion_text,
        slug_id=slug,
        extracted_cases=[],
        case_treatments=[],
        confidence_scores=[],
        needs_human_review=False,
        error_messages=[]
    )
    
    # Run analysis workflow
    print("Starting agentic analysis workflow...")
    try:
        final_state = workflow.app.invoke(initial_state)
        
        processing_time = time.time() - start_time
        
        # Create structured result
        result = AnalysisResult(
            case_treatments=final_state["case_treatments"],
            total_cases_found=len(final_state["case_treatments"]),
            analysis_summary=f"Analyzed {len(final_state['case_treatments'])} cases in {processing_time:.2f}s",
            processing_time=processing_time
        )
        
        return result
        
    except Exception as e:
        print(f"Workflow execution failed: {e}")
        return None

def print_results(result: AnalysisResult):
    """Pretty print analysis results"""
    print("\n" + "="*80)
    print("NEGATIVE TREATMENT ANALYSIS RESULTS")
    print("="*80)
    print(f"Summary: {result.analysis_summary}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Total cases analyzed: {result.total_cases_found}")
    
    if result.case_treatments:
        print("\nCASE TREATMENTS:")
        print("-" * 50)
        
        for i, treatment in enumerate(result.case_treatments, 1):
            print(f"\n{i}. {treatment.case_name}")
            print(f"   Citation: {treatment.case_citation}")
            print(f"   Negative Treatment: {'YES' if treatment.negative_treatment else 'NO'}")
            print(f"   Confidence: {treatment.confidence_score:.2f}")
            if treatment.treatment_types:
                print(f"   Treatment Types: {[t.value for t in treatment.treatment_types]}")
            print(f"   Explanation: {treatment.explanation}")
    else:
        print("\nNo cases found in this opinion.")

if __name__ == "__main__":
    # Example usage
    slug = 'littlejohn-v-state-7'
    
    print("Agentic Legal Case Treatment Analysis")
    print("=" * 50)
    
    result = extract_negative_treatments_agentic(slug)
    
    if result:
        print_results(result)
        
        # Export to JSON
        with open(f"treatment_analysis_{slug}.json", "w") as f:
            json.dump(result.dict(), f, indent=2, default=str)
        print(f"\nResults saved to treatment_analysis_{slug}.json")
    else:
        print("Analysis failed")
