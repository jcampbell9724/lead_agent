from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")

# Initialize the LLM for CrewAI (using langchain integration)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
    logger.info("Successfully initialized ChatGoogleGenerativeAI")
except Exception as e:
    logger.error(f"Error initializing ChatGoogleGenerativeAI: {str(e)}")
    raise

# Configure standard Gemini for backup/testing
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize search tool
try:
    search_tool = DuckDuckGoSearchRun()
    logger.info("Successfully initialized DuckDuckGoSearchRun")
except Exception as e:
    logger.error(f"Error initializing DuckDuckGoSearchRun: {str(e)}")
    # Fallback to a basic search if the tool fails
    search_tool = None

class LeadGenerationCrew:
    def __init__(self, business_description):
        self.business_description = business_description
        logger.info("LeadGenerationCrew initialized with business description")
        
    def create_agents(self):
        logger.info("Creating agents...")
        tools_list = [search_tool] if search_tool else []
        
        # Lead Qualification Expert
        lead_qualifier = Agent(
            role='Lead Qualification Expert',
            goal='Define ideal customer profile and qualification criteria',
            backstory="""You are an expert in lead qualification and customer profiling.
            Your expertise lies in understanding businesses and determining what makes
            a lead qualified for their specific needs.""",
            tools=tools_list,
            verbose=True,
            llm=llm  # Use Langchain-wrapped Gemini model
        )
        logger.info("Lead Qualification Expert agent created")

        # Lead Researcher
        lead_researcher = Agent(
            role='Lead Researcher',
            goal='Find and validate qualified leads based on criteria',
            backstory="""You are a skilled researcher specializing in finding business
            leads that match specific criteria. You're excellent at validating
            contact information and ensuring leads are current and accurate.""",
            tools=tools_list,
            verbose=True,
            llm=llm  # Use Langchain-wrapped Gemini model
        )
        logger.info("Lead Researcher agent created")

        # Email Campaign Specialist
        email_specialist = Agent(
            role='Email Campaign Specialist',
            goal='Create personalized email sequences and track potential outcomes',
            backstory="""You are an expert in crafting compelling email campaigns
            that convert. You specialize in personalization and understanding
            the psychology of email marketing.""",
            tools=tools_list,
            verbose=True,
            llm=llm  # Use Langchain-wrapped Gemini model
        )
        logger.info("Email Campaign Specialist agent created")

        return lead_qualifier, lead_researcher, email_specialist

    def create_tasks(self, lead_qualifier, lead_researcher, email_specialist):
        logger.info("Creating tasks...")
        # Task 1: Define qualification criteria
        define_criteria = Task(
            description=f"""
            Analyze the following business and determine detailed qualification criteria:
            {self.business_description}
            
            Create a comprehensive ideal customer profile including:
            1. Company size
            2. Industry
            3. Budget range
            4. Pain points
            5. Decision maker profiles
            6. Technology stack requirements
            7. Geographic location preferences
            
            Provide the criteria in a structured format.
            """,
            agent=lead_qualifier
        )
        logger.info("Define criteria task created")

        # Task 2: Find qualified leads
        find_leads = Task(
            description="""
            Using the qualification criteria provided, find 5 highly qualified leads.
            For each lead, provide:
            1. Company name
            2. Key decision maker's name and role
            3. Company size and industry
            4. Relevant technology stack or business practices
            5. Recent company news or developments
            6. LinkedIn profile or company website
            7. Qualification score (1-10) with justification
            """,
            agent=lead_researcher,
            dependencies=[define_criteria]
        )
        logger.info("Find leads task created")

        # Task 3: Create email sequences
        create_emails = Task(
            description="""
            For each qualified lead, create:
            1. Initial cold email
            2. Three follow-up emails
            3. Sales opportunity summary
            
            Each email should be personalized based on:
            - Company-specific information
            - Recent news or developments
            - Specific pain points
            - Relevant use cases
            
            Include subject lines and follow-up timing recommendations.
            """,
            agent=email_specialist,
            dependencies=[find_leads]
        )
        logger.info("Create emails task created")

        return [define_criteria, find_leads, create_emails]

    def run(self):
        logger.info("Starting the Lead Generation Crew process...")
        try:
            # Create agents
            lead_qualifier, lead_researcher, email_specialist = self.create_agents()
            
            # Create tasks
            tasks = self.create_tasks(lead_qualifier, lead_researcher, email_specialist)
            
            # Create crew
            crew = Crew(
                agents=[lead_qualifier, lead_researcher, email_specialist],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            logger.info("Crew created successfully")
            
            # Execute crew tasks
            logger.info("Kicking off the crew...")
            result = crew.kickoff()
            logger.info("Crew execution completed successfully")
            
            return result
        except Exception as e:
            logger.error(f"Error running Lead Generation Crew: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    business_description = """
    Example Business:
    A B2B SaaS company providing AI-powered inventory management software
    for medium to large retail businesses. The solution integrates with
    major e-commerce platforms and helps predict stock levels, automate
    reordering, and optimize warehouse operations.
    """
    
    crew = LeadGenerationCrew(business_description)
    result = crew.run()
    print("\nFinal Results:")
    print(result) 