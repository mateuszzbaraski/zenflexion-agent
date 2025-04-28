from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
     [
         (
             "system",
             "You are a coaching helper for personal journaling. Generate thorough and compasionate analysis of ones journal entry, give a sentiment of this message, what emotions you are picking up and one thing to focus on tomorrow to feel better"
             "Always provide detailed recommendations, including requests for length, precision, ease of digestion by user, etc.",
         ),
         MessagesPlaceholder(variable_name="messages"),
     ]
 )
 
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a coaching helper for personal journaling assistant tasked with writing excellent coaching journal entries."
            " Generate the best journal entry post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm