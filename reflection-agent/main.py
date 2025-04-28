from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph!")
    
    inputs = HumanMessage(content="""I have joined a new company and a team lately. I am very excited, as this is dreams come true for me. 
                          I do however feel a buit ancious, that others have way more experience than me, or that I won't be seen as capable.
                          I am doing my best, but at the same time I have a lot going on in my life. The team is very understanding and supportive, yet I 
                          want to make sure I am up with the pace and expectations. I am managing this stress, recently realxing a little bit and being more myself!
 
                                   """)
    response = graph.invoke(inputs)