from agents.ollama_agent import stream_task_agent
from agents.ollama_agent import run_task_agent
import streamlit as st


def render_main_view(model: str, temperature: float, mode: str):
    st.markdown("<h1>✨ AI Task Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Manage your tasks with local AI</p>", unsafe_allow_html=True)
    
    # Modern layout for metrics
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 2])
        col1.metric("Current Model", model)
        col2.metric("Creativity", f"{temperature:.1f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    import uuid

    # Initialize chat history and thread ID
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask your agent (e.g., What should I do today?)"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display AI response in chat message container
        with st.chat_message("ai"):
            with st.spinner("Agent is thinking...", show_time=True):
                if mode == "Streaming":
                    placeholder = st.empty()
                    thinking = ""
                    response = ""
                    for chunk in stream_task_agent(prompt, model, temperature, st.session_state.thread_id):
                        if chunk.type == "reasoning":
                            thinking += chunk.reasoning
                        elif chunk.type == "text":
                            response += chunk.content
                            
                        display_text = ""
                        if thinking:
                            display_text += f"🤔 **Thinking:**\n```text\n{thinking}\n```\n\n"
                        if response:
                            display_text += f"💡 **Response:**\n{response}"
                            
                        placeholder.markdown(display_text)
                else:
                    response = run_task_agent(prompt, model, temperature, st.session_state.thread_id)
                    st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "ai", "content": response})
