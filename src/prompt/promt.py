from langchain_core.prompts import ChatPromptTemplate

format_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Role: You are an expert at extracting key information about projects from the "experience" section of an OCR'd resume.

Instruction:
You are provided with a dictionary containing information from the user's resume by an OCR model. It may have misspellings or wrong entries.
Please correct the spelling of each field.
Move the content of the fields to more appropriate fields if necessary.
You must not fabricate information and create new information.

Output must be in JSON format following the same structure as the input.
""",
        ),
        ("human", "{user_input}"),
    ]
)
matching_jd_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""
Role: Expert resume analyzer matching qualifications with the following job requirements:
{job_description}

Tasks:
1. Analyze resume experience and skills against job requirements:
- Match skills/experience to job requirements
- Score relevance (0-100%)
- Identify key matching qualifications
2. Provide concise reasoning explaining the score based on:
  - Key matching skills and experiences
  - Notable gaps
  - Level of experience alignment
Output JSON format:
Output JSON:
{{
   "score": float,  # Overall match percentage 
   "reasoning": str  # Clear 2-3 sentence explanation of the score
}}
""",
        ),
        ("human", "{resume_input}"),
    ]
)
