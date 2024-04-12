
import matplotlib.pyplot as plt
import requests as re
import streamlit as st
from bs4 import BeautifulSoup

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

def show():
    st.markdown("<h1 style='color:#c8a808'>Project Details</h1>", unsafe_allow_html=True)
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h4 style='color:#4d6cc1'>Approach</h4>", unsafe_allow_html=True)
    # st.subheader('Approach')
    st.write('We used _supervised learning_ to classify phishing and legitimate websites. '
             'We benefit from content-based approach and focus on html of the websites. '
             'Also, We used scikit-learn for the ML models.'
             )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Conceptual Diagram</h4>", unsafe_allow_html=True)
    # Load the image
    phising_architect = "static/phising_architecture.jpg"

    # Display the image
    st.image(phising_architect, caption='Conceptual Diagram', use_column_width=True)

    st.write('For this educational project, '
             'We created my own data set and defined features, some from the literature and some based on manual analysis. '
             'We used requests library to collect data, BeautifulSoup module to parse and extract features. ')
    st.write('The source code and data sets are available in the below Github link:')
    st.write('_https://github.com/AdarshVajpayee19/Phishing-Website-Detection-ML_')
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Data set </h4>", unsafe_allow_html=True)

    st.write('We used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')

    # ----- FOR THE PIE CHART ----- #
    labels = 'phishing', 'legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))


    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Features</h4>", unsafe_allow_html=True)

    st.write('We used only content-based features.We didn\'t use url-based faetures like length of url etc.'
             'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

    st.markdown("<h4 style='color:#4d6cc1'>Results</h4>", unsafe_allow_html=True)

    st.write('We used 7 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
             'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
             'Comparison table is below:')
    st.table(ml.df_results)
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours')
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Phishing Demo.")
    # st.image("static\phising.gif", use_column_width=True)
    # st.markdown(
    #     '<img src="./app/static/phising.gif">',
    #     unsafe_allow_html=True,
    # )

    # Load the GIF
    phising = "static/phising.gif"

    # Display the GIF
    st.image(phising, caption='Animated Phishing Example GIF', use_column_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)





    # Title and Description
    st.title("End-to-End Machine Learning Project: Phishing Website Detection")
    st.write("This is an End-to-End Machine Learning Project which focuses on phishing websites to classify phishing and legitimate ones. Particularly, We focused on content-based features like html tag based features. You can find feature extraction, data collection, preparation process here. Also, building ML models, evaluating them are available here.")

    # Inputs
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Inputs</h4>", unsafe_allow_html=True)

    st.write("CSV files of phishing and legitimate URLs:")
    st.write("- verified_online.csv: phishing websites URLs from phishtank.org")
    st.write("- tranco_list.csv: legitimate websites URLs from tranco-list.eu")

    # General Flow
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>General Flow</h4>", unsafe_allow_html=True)

    st.write("1. Use csv file to get URLs")
    st.write("2. Send a request to each URL and receive a response by requests library of python")
    st.write("3. Use the content of response and parse it by BeautifulSoup module")
    st.write("4. Extract features and create a vector which contains numerical values for each feature")
    st.write("5. Repeat feature extraction process for all content\websites and create a structured dataframe")
    st.write("6. Add label at the end to the dataframes | 1 for phishing 0 for legitimate")
    st.write("7. Save the dataframe as csv and structured_data files are ready!")
    st.write("8. Check 'structured_data_legitimate.csv' and 'structured_data_phishing.csv' files.")
    st.write("9. After obtaining structured data, you can use combine them and use them as train and test data")
    st.write("10. You can split data as train and test like in the machine_learning.py first part, or you can implement K-fold cross-validation like in the second part of the same file. We implemented K-fold as K=5.")
    st.write("11. Then We implemented five different ML models:")
    st.write("   - Support Vector Machine")
    st.write("   - Gaussian Naive Bayes")
    st.write("   - Decision Tree")
    st.write("   - Random Forest")
    st.write("   - AdaBoost")
    st.write("12. You can obtain the confusion matrix, and performance measures: accuracy, precision, recall")
    st.write("13. Finally, We visualized the performance measures for all models.")
    st.write("14. Naive Bayes is the best for my case.")

    # Important Notes
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Important Notes</h4>", unsafe_allow_html=True)

    st.write("Features are content-based and need BeautifulSoup module's methods and fields etc So, you should install it.")

    # Dataset
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#4d6cc1'>Data set</h4>", unsafe_allow_html=True)

    st.write("With your URL list, you can create your own dataset by using data_collector python file.")

    st.markdown("<hr>", unsafe_allow_html=True)
