"""Streamlit chat app for the HR Policy Assistant.

Run with:  basicragenv\\Scripts\\python.exe -m streamlit run app.py
"""

import streamlit as st

from hr_assistant.pipeline import ask, build_hr_assistant


st.set_page_config(page_title="HR Policy Assistant", page_icon="HR")
st.title("HR Policy Assistant")
st.caption("Ask me anything about the company HR policy document.")


@st.cache_resource(show_spinner="Setting up the assistant...")
def get_agent():
    return build_hr_assistant()


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

agent = get_agent()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about HR policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask(agent, question)
            except Exception as error:
                answer = f"Sorry, something went wrong: {error}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
