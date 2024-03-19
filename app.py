#from langchain.agents import Tool
#from langchain.agents import load_tools

from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama

#from langchain.tools import DuckDuckGoSearchRun
#from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

#wrapper = DuckDuckGoSearchAPIWrapper(region="pt-br", time="d", max_results=2)

"""
1. Create agent who works for you as a researcher. Researcher will access information availble over internet to about the topic.
2. Create one more agent who writes about the the content provided by the researcher. 
3. Crete one more agent as review who will review the content and improvise it . Define agents for resardefine agents that are going to research latest  tools and write a blog about it 

"""

ollama_llm = Ollama(model="llama2:13b")

search_tool = DuckDuckGoSearchRun(region="pt-br")

researcher = Agent(
  role='Research',
  goal='Search the internet for the information requested',
  backstory="""
  You are a brasilian attorney. Using the information in the task, you find out some of the most popular 
  facts in brasil about the topic along with some of the trending aspects.
  Be detailed, inform the number and the date of brasilian federal laws, brasilian federal jurisprudence, brasilian federal processes and names of the cases.
  You have a knack for dissecting complex data and presenting actionable insights.
  These keywords must never be translated and transformed:
    - Action:
    - Thought:
    - Action Input:
    because they are part of the thinking process instead of the output.
  """,
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ollama_llm
)
researcher1 = Agent(
  role='Research',
  goal='Search the internet for brasilian federal laws, brasilian federal regulations and brasilian federal jurisprudence',
  backstory="""
  You are a brasilian attorney. Using the information in the task, you find out some of the most popular facts in brasil 
  about the topic along with some of the trending aspects.
  Be detailed, inform the number and the date of laws, jurisprudence, processes, 
  REsp and names of the cases.
  You have a knack for dissecting complex data and presenting actionable insights.
  These keywords must never be translated and transformed:
    - Action:
    - Thought:
    - Action Input:
    because they are part of the thinking process instead of the output.
  """,
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ollama_llm
)

writer = Agent(
  role='Attorney Content Strategist',
  goal='Craft compelling content on a set of information provided by the researcher.',
  backstory="""You are a researcher that like detailed explanation way of explaining. 
  You transform complex concepts into compelling narratives.
  These keywords must never be translated and transformed:
    - Action:
    - Thought:
    - Action Input:
    because they are part of the thinking process instead of the output.""",
  verbose=True,
  allow_delegation=True,
  llm=ollama_llm
)

task1 = Task(
  description="""Research cases about appeals in administrative proceedings regulated by Law No. 9,784/99 according to the Superior Court of Justice and the Federal Council of Medicine in Brazil""",
  agent=researcher,
  expected_output='A refined finalized version of the blog post in markdown format'
)

task2 = Task(
  description="""Using insights provided, government an engaging blog
  post that highlights Brazilian federal justice cases, legal process case, Brazilian federal law,
  Your post should be informative attorney adviser
  Make it sound precise, and never invent facts.
  Your final answer MUST be the full blog post of at least 30 paragraphs.
  The target word count for the blog post should be between 5,500 and 20,500 words, with a sweet spot at around 10,450 words.""",
  agent=writer,
  expected_output='A refined finalized version of the blog post in markdown format'
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher,researcher1,  writer],
  tasks=[task1, task1, task2],
  verbose=1, # You can set it to 1 or 2 for different logging levels
)

# Get your crew to work!
result = crew.kickoff()
print("------------result-------------", result)
