# # #importing required libraries

# # import pickle
# # import warnings
# # from urllib.parse import quote_plus

# # import numpy as np
# # import pandas as pd
# # from flask import Flask, render_template, request
# # from sklearn import metrics

# # warnings.filterwarnings('ignore')
# # from feature import FeatureExtraction

# # file = open("pickle/model.pkl","rb")
# # gbc = pickle.load(file)
# # file.close()


# # app = Flask(__name__)

# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     if request.method == "POST":

# #         url = request.form["url"]
# #         obj = FeatureExtraction(url)
# #         x = np.array(obj.getFeaturesList()).reshape(1,30)

# #         y_pred =gbc.predict(x)[0]
# #         #1 is safe
# #         #-1 is unsafe
# #         y_pro_phishing = gbc.predict_proba(x)[0,0]
# #         y_pro_non_phishing = gbc.predict_proba(x)[0,1]
# #         # if(y_pred ==1 ):
# #         pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
# #         return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
# #     return render_template("index.html", xx =-1)


# # if __name__ == "__main__":
# #     app.run(debug=True)





# import pickle
# import warnings

# import numpy as np
# import pandas as pd
# import streamlit as st

# from feature import FeatureExtraction

# warnings.filterwarnings('ignore')

# # Load the trained model
# file = open("pickle/model.pkl", "rb")
# gbc = pickle.load(file)
# file.close()

# # Define the main function
# def main():
#     st.title("Phishing URL Detector")

#     # Get URL input from the user
#     url = st.text_input("Enter the URL:")
    
#     # Add a button to initiate the URL check
#     if st.button("Check URL"):
#         if url:
#             obj = FeatureExtraction(url)
#             x = np.array(obj.getFeaturesList()).reshape(1, 30)

#             # Make predictions
#             y_pred = gbc.predict(x)[0]
#             y_pro_phishing = gbc.predict_proba(x)[0, 0]
#             y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

#             # Display the result
#             if y_pred == 1:
#                 st.success("It is {:.2f}% safe to go".format(y_pro_non_phishing * 100))
#             else:
#                 st.error("It is {:.2f}% unsafe to go".format(y_pro_phishing * 100))

# # Run the main function
# if __name__ == "__main__":
#     main()









# import pickle
# import warnings
# from pathlib import Path

# from feature import FeatureExtraction

# warnings.filterwarnings('ignore')
# import matplotlib.pyplot as plt
# import numpy as np
# import requests as re
# import streamlit as st
# from bs4 import BeautifulSoup

# # Import the FeatureExtraction class from feature.py
# from feature import FeatureExtraction

# st.set_page_config(page_title='Phishing Website Detection Using Machine Learning', page_icon='./static/favicon.png')
# # Add the CSS rule using st.markdown
# st.markdown(
#     """
#     <style>
#     .css-wjbhl0.e1fqkh3o9 {
#         display: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# # --- PATH SETTINGS ---
# current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# css_file = current_dir / "styles" / "main.css"
# phishing_account_pic = current_dir / "static" / "Phishing-account.gif"

# # Load the trained model
# file = open("pickle/model.pkl", "rb")
# gbc = pickle.load(file)
# file.close()


# def applicationRun():

#     # Add content for the Home page here
#     # Set page title and description
#     st.markdown("<h1 style='color:#c8a808'>Phishr</h1>", unsafe_allow_html=True)
#     st.markdown("<h3 style='color:#4d6cc1'>Phish the Phisher before they phish you!!!</h3>", unsafe_allow_html=True)

#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Understanding Phishing Attack</h4>", unsafe_allow_html=True)
#     st.write('Phishing attacks are a common type of cyber attack where malicious actors attempt to deceive individuals or organizations into revealing '
#              'sensitive information such as usernames, passwords, credit card numbers, or other personal or financial data. These attacks typically '
#              'involve impersonating a trusted entity, such as a bank, a government agency, a company, or even a colleague or friend.')

#     # Load the GIF
#     phishing_acc = "static/Phishing-account.gif"

#     # Display the GIF
#     st.image(phishing_acc, caption='PHISHr', use_column_width=True)

#     st.markdown("<hr>", unsafe_allow_html=True)
#     with st.expander('EXAMPLE PHISHING URLs:'):
#         st.write('_https://rtyu38.godaddysites.com/_')
#         st.write('_https://karafuru.invite-mint.com/_')
#         st.write('_https://defi-ned.top/h5/#/_')
#         st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)

#     url = st.text_input('Enter the URL', key='url_input')
#     if st.button("Check URL"):
#         if url:
#             obj = FeatureExtraction(url)
#             x = np.array(obj.getFeaturesList()).reshape(1, 30)

#             # Make predictions
#             y_pred = gbc.predict(x)[0]
#             y_pro_phishing = gbc.predict_proba(x)[0, 0]
#             y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
            
#             # Display the result
#             if y_pred == 1:
#                 st.success("It is {:.2f}% safe to go".format(y_pro_non_phishing * 100))
#             else:
#                 st.error("It is {:.2f}% unsafe to go".format(y_pro_phishing * 100))

#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown("<h4 style='color:#4d6cc1'>Mitigating Phishing Risks</h4>", unsafe_allow_html=True)

#     st.write(
#         'Phishing attacks pose significant risks to individuals, businesses, and organizations. They can lead to identity theft, financial loss, data breaches, '
#         'and reputational damage. To protect against phishing attacks, it\'s essential to stay vigilant, be cautious of unsolicited emails or messages, '
#         'verify the authenticity of websites and communications, and regularly update security measures such as antivirus software and firewalls. '
#         'Additionally, education and awareness training for employees and users are crucial in preventing successful phishing attacks.')
#     # Add a horizontal line
#     st.markdown("<hr>", unsafe_allow_html=True)

# from menu import streamlit_menu
# from pages import FAQ, Blog, Contact_Us, Project_Description

# selected = streamlit_menu()

# if selected == "Home":
#     applicationRun()
# if selected == "Project_Description":
#     Project_Description.show()
# elif selected == "Contact_Us":
#     Contact_Us.show()
# elif selected == "FAQ":
#     FAQ.show()
# elif selected == "Blog":
#     Blog.show()


import pickle
import warnings
from pathlib import Path

from feature import FeatureExtraction

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import requests as re
import streamlit as st
from bs4 import BeautifulSoup

# Import the FeatureExtraction class from feature.py
from feature import FeatureExtraction
from menu import streamlit_menu
from pages import FAQ, Blog, Contact_Us, Project_Description
from utils import footer

# Set page config
st.set_page_config(page_title='Phishing Website Detection Using Machine Learning', page_icon='./static/favicon.png')
# Add the CSS rule using st.markdown
st.markdown(
    """
    <style>
    .css-wjbhl0.e1fqkh3o9 {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
phishing_account_pic = current_dir / "static" / "Phishing-account.gif"


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        # Combine predictions (e.g., by averaging or voting)
        # For simplicity, let's use majority voting
        return np.mean(predictions, axis=0) > 0.5  # Use threshold as needed


# Load the trained model
file = open("pickle/ensemble_model.pkl", "rb")
ensemble_model = pickle.load(file)
file.close()


def applicationRun():

    # Add content for the Home page here
    # Set page title and description
    st.markdown("<h1 style='color:#c8a808'>Phishr</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#4d6cc1'>Phish the Phisher before they phish you!!!</h3>", unsafe_allow_html=True)

    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Understanding Phishing Attack</h4>", unsafe_allow_html=True)
    st.write('Phishing attacks are a common type of cyber attack where malicious actors attempt to deceive individuals or organizations into revealing '
             'sensitive information such as usernames, passwords, credit card numbers, or other personal or financial data. These attacks typically '
             'involve impersonating a trusted entity, such as a bank, a government agency, a company, or even a colleague or friend.')

    # Load the GIF
    phishing_acc = "static/Phishing-account.gif"

    # Display the GIF
    st.image(phishing_acc, caption='PHISHr', use_column_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander('EXAMPLE PHISHING URLs:'):
        st.write('_https://rtyu38.godaddysites.com/_')
        st.write('_https://karafuru.invite-mint.com/_')
        st.write('_https://defi-ned.top/h5/#/_')
        st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    url = st.text_input('Enter the URL', key='url_input')
    if st.button("Check URL"):
        if url:
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)

            # Make predictions using the ensemble model
            is_phishing = ensemble_model.predict(x)

            # Display the result
            if is_phishing:
                st.error("This URL is likely unsafe.")
            else:
                st.success("This URL is likely safe.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Mitigating Phishing Risks</h4>", unsafe_allow_html=True)

    st.write(
        'Phishing attacks pose significant risks to individuals, businesses, and organizations. They can lead to identity theft, financial loss, data breaches, '
        'and reputational damage. To protect against phishing attacks, it\'s essential to stay vigilant, be cautious of unsolicited emails or messages, '
        'verify the authenticity of websites and communications, and regularly update security measures such as antivirus software and firewalls. '
        'Additionally, education and awareness training for employees and users are crucial in preventing successful phishing attacks.')
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    # Call the footer function to display the footer
    footer()


selected = streamlit_menu()

if selected == "Home":
    applicationRun()
if selected == "Project_Description":
    Project_Description.show()
elif selected == "Contact_Us":
    Contact_Us.show()
elif selected == "FAQ":
    FAQ.show()
elif selected == "Blog":
    Blog.show()
    footer()
