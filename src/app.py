from ui.main_view import render_main_view
from ui.sidebar import render_sidebar
import streamlit as st

def main():
	st.set_page_config(
		page_title="Task Agent",
		page_icon="📋",
		layout="wide",
		initial_sidebar_state="expanded",
	)

	
	config = render_sidebar()	

	render_main_view(
		model=config["model"],
		temperature=config["temperature"],
		mode=config["mode"]
	)



if __name__ == "__main__":
    main()
