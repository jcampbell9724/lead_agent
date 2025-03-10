from lead_generation_crew import LeadGenerationCrew
import traceback
import sys

def main():
    # Example business description
    business_description = """
    Company Name: TechFlow Analytics
    Product/Service: AI-Powered Business Intelligence Platform

    Core Offerings:
    - Real-time data analytics dashboard
    - Predictive sales forecasting
    - Customer behavior analysis
    - Automated reporting system

    Target Market:
    - Industry: E-commerce, Retail, SaaS companies
    - Company Size: 50-1000 employees
    - Geographic Focus: North America and Europe

    Value Proposition:
    - Reduce data analysis time by 75%
    - Increase forecast accuracy by 40%
    - Automate 90% of reporting tasks
    - ROI within 3 months

    Price Range:
    - Entry level: $2,000/month
    - Average deal size: $5,000/month
    - Enterprise: Custom pricing, typically $15,000+/month

    Current Customer Profile:
    - Mid-sized e-commerce companies
    - Data-driven SaaS businesses
    - Success stories with major retail chains
    """

    try:
        # Create and run the crew
        print("Initializing Lead Generation Crew...")
        crew = LeadGenerationCrew(business_description)
        
        print("Running the crew...")
        result = crew.run()

        # Print the results
        print("\nFinal Results:")
        print(result)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("- Check that all required packages are installed")
        print("- Verify API keys are correctly set")
        print("- Ensure the CrewAI version is compatible with the code")

if __name__ == "__main__":
    main()