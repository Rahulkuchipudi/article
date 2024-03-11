import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sqlite3
import bcrypt



# Initialize SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup(username, password):
    # Check if user already exists
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False
    else:
        # Hash the password and store the user
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True
   
        

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(password, user[1]):
        return True
    else:
        return False

df = pd.read_csv('Article (3).csv')
df = df.dropna()
df['combined_text'] = df['title'] + ' ' + df['text'] + ' ' + df['summary'] + ' ' + df['keywords'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim, df=df):
    try:
        idx = df[df['title'] == title].index[0]  
    except IndexError:
        print(f"Error: Title '{title}' not found in the dataset.")
        return pd.Series([])  

    sim_scores = list(enumerate(cosine_sim[idx]))  
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    sim_scores = sim_scores[1:11] 
    title_indices = [i[0] for i in sim_scores]  
    return df[['title','Links']].iloc[title_indices]


if __name__=='__main__':
    st.title('News Recommendation App')


    auth_status = st.session_state.get('auth_status', None)
    if auth_status == "logged_in":
        st.success(f"Welcome {st.session_state.username}!")
        st.header('Select a News for Recommendation')

        selected_News = st.selectbox('Choose a News', df['title'].unique())

        if st.button('Get Recommendations'):
            st.subheader('Recommended Newss:')
            recommendations = get_recommendations(selected_News)
            # st.write(recommendations)

            # Display recommendations with clickable links
            for index, row in recommendations.iterrows():
                st.markdown(f"[{row['title']}]({row['Links']})")


    elif auth_status == "login_failed":
        st.error("Login failed. Please check your username and password.")
        auth_status = None
    elif auth_status == "signup_failed":
        st.error("Signup failed. Username already exists.")
        auth_status = None
    # Login/Signup form
    if auth_status is None or auth_status == "logged_out":
        form_type = st.radio("Choose form type:", ["Login", "Signup"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if form_type == "Login":
            if st.button("Login"):
                if login(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "login_failed"
                    st.rerun()
        else:  # Signup
            if st.button("Signup"):
                if signup(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "signup_failed"
                    st.rerun()

    # Logout button
    if auth_status == "logged_in":
        if st.button("Logout"):
            st.session_state.auth_status = "logged_out"
            del st.session_state.username
            st.rerun()
