import streamlit as st
from PIL import Image
from predict import output_sr
from streamlit_option_menu import option_menu

def main():
    # main menu
    main_choice = main_option_menu()
    # for side-menu
    models_list = ['fsrcnn', 'edsr']
    image_width = 500

    if main_choice == "Home":
        text = "Hello World!!"
        size = 60
        color = "Blue"
        st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',unsafe_allow_html=True)
    elif main_choice == "SR-Test":
        st.sidebar.header('Model Selection')
        # get selected value
        choice = st.sidebar.selectbox('Select SR Model', models_list)
        # define saved model
        saved_model = "./sample_saved/%s/%s.ckpt" %(choice, choice)
        # input image
        image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'],key=main_choice) # key for refreshing

        if image is not None:
            # set two columns
            col1, col2 = st.columns(2)
            # read input image
            image_ = image.read()
            with col1:
                # show input image
                st.image(image_, caption='original', width=image_width)
            # predict sr_image    
            sr_Image = output_sr(image_, choice, saved_model)
            with col2:
                # show sr_image
                st.image(image_, caption=choice, width=image_width)
    elif main_choice == "Compare":
        image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'],key=main_choice) # key for refreshing
        if image is not None:
            image_ = image.read()
            # set columns
            cols = st.columns(len(models_list)+1)
            # read input image
            with cols[0]:
                # show input image
                st.image(image_, caption='original',width=image_width)

            for i, m in enumerate(models_list,1):
                saved_model = "./saved/%s/%s.ckpt" %(m, m)
                with cols[i]:
                    # show input image
                    st.image(image_, caption=m, width=image_width)

def main_option_menu():
    # CSS style definitions
    selected = option_menu(None, ["Home", "SR-Test", "Compare"], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )
    return selected

if __name__ == "__main__":
    st.set_page_config(layout="wide") # set wide mode as default 
    main()
