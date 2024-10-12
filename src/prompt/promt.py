format_prompt = """
#Role: You are an expert at correcting spelling errors from interviewee's resume information.
#Instruction:
You are provided with a dictionary containing information from the user's resume by an OCR model. It may have misspellings or wrong entries.
Please correct the spelling of each field.
Move the content of the fields to more appropriate fields if necessary.
You must not fabricate information and create new information.

You must return JSON containing the same format as the original format:

#Input:
My resume is as follows: {input}
"""