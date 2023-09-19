from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

# import chainlit as cl
import streamlit as st

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        # model="meta-llama/Llama-2-7b-chat-hf",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.5,
    )


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response


def process_answer(instruction):
    print("\n********************************* START ******************************\n")
    response = ""
    instruction = instruction
    qa = qa_bot()
    generated_text = qa(instruction)
    answer = generated_text["result"]
    return answer, generated_text


def main():
    # Ask the user for a query
    question = input("What is your query? (Type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == "exit":
        print("Exiting the program.")
        return

    # Process the user's query
    answer, generated_text = process_answer(question)
    # Display the result
    print("\n\nResult:\n", answer)

    index = 1
    for document_string in generated_text["source_documents"]:
        print(f"\n\nMore Info {index} :\n")
        index += 1
        document_string = str(document_string)
        content_start = document_string.find("page_content='") + len("page_content='")
        content_end = document_string.find("metadata=") - len("metadata=")
        page_content = document_string[content_start:content_end].replace("\\n", "\n")
        print(page_content)
    print("\n********************************* END ******************************\n")


# # chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to PDF Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)


# @cl.on_message
# async def main(message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message, callbacks=[cb])
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\nSources: \n" + str(sources)
#     else:
#         answer += "\nNo sources found"

#     await cl.Message(content=answer).send()


# # streamlit code


# def main():
#     st.title("Search Your PDF 🐦📄")
#     with st.expander("About the App"):
#         st.markdown(
#             """
#             This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
#             """
#         )
#     question = st.text_area("Enter your Question")
#     if st.button("Ask"):
#         st.info("Your Question: " + question)
#         st.info("Your Answer")
#         answer, metadata = process_answer(question)
#         st.write(answer)
#         st.write(metadata)


if __name__ == "__main__":
    main()
