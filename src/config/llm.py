from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)
