from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def main():
    print("Hello from langchain-course!")
    information = """Chandrasekaran Joseph Vijay (born 22 June 1974) is an Indian politician and former actor who has served as the ninth Chief Minister of Tamil Nadu since May 2026. He is the founder and president of the political party Tamilaga Vettri Kazhagam (TVK). Prior to entering politics, he was a leading actor in Tamil cinema and among the highest-paid actors in India.

Following a career as a child actor in the 1980s, Vijay made his debut as a leading man in Naalaiya Theerpu (1992), directed by his father S. A. Chandrasekaran. He rose to fame with romance films such as Poove Unakkaga (1996), Love Today (1997), Kadhalukku Mariyadhai (1997) and Thullatha Manamum Thullum (1999), before transitioning into an action star with Thirumalai (2003), Ghilli (2004) and Pokkiri (2007). From the 2010s onward, he starred in major commercial successes including Thuppakki (2012), Kaththi (2014), Mersal (2017), Sarkar (2018), Master (2021), Leo (2023), and The Greatest of All Time (2024), which rank among the highest-grossing Tamil films.

In February 2024, Vijay announced his retirement from films and his entry into politics. He founded the TVK, which secured 108 seats in the 2026 Tamil Nadu Legislative Assembly election. Emerging as the single-largest party, the TVK broke the half-century-long DMK–AIADMK duopoly in Tamil Nadu politics.[1][2] With the support of several smaller parties, Vijay became chief minister on 10 May 2026.
"""
    summary_template = """
             You are a helpful assistant that summarizes information about a person. information: {information}
             1. Provide a concise summary.
             3. Provide two interesting facts about the person."""
    print(summary_template)
    summary_template_prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    llm = ChatOllama(temperature=0, model="qwen3:latest")

    chain = summary_template_prompt | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
