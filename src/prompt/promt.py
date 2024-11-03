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


prompt_experience = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Role: You are an expert at extracting key information about projects from the "experience" section of an OCR'd resume.

# Instruction:
You are given a JSON object containing extracted resume data from an OCR model. This data likely contains errors like misspellings, merged words, and extracted noise. Your task is to:

1. **Pre-process the "experience" field:**
    * Correct misspellings using your knowledge of common resume terms and English vocabulary.
    * Separate merged words and remove any obvious OCR noise.

2. **Identify the "experience" field:** Locate the field labeled "experience" (or a similar label) within the JSON object.

3. **Extract project information:** 
    * Identify project mentions: Look for keywords and phrases that indicate a project, such as "project," "developed," "implemented," "designed," "contributed to," etc.
    * Extract project details: For each project mentioned:
        * Project name or description
        * Role and contributions
        * Technologies used
        * Outcomes and achievements

4. **Structure the output:** Return a JSON object with a "projects" field containing an array of extracted project details. The exact format can be flexible to accommodate variations in the input data.



""",
        ),
        ("human", "{user_input}"),
    ]
)
