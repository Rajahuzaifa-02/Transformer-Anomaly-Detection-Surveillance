import streamlit as st
import requests

def user_login():
        # Centered page title with custom styling
    st.markdown("""
        <h2 style='text-align: center; color: #333;'>Transformer Based Anomaly Detection</h2>
    """, unsafe_allow_html=True)

    # CSS for the centered container and styled inputs/buttons
    st.markdown("""
    <style>
    body {
        background-color: #f2f2f2;
    }

    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;
    }

    .login-box {
        background-color: blue;
        padding: 40px 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        width: 50%;
        max-width: 400px;
        text-align: center;
    }

    .login-box h2 {
        margin-bottom: 30px;
        color: #333;
    }

    .stTextInput>div>input, .stPasswordInput>div>input {
        padding: 12px;
        width: 100% !important;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        padding: 12px;
        border-radius: 6px;
        width: 100%;
        margin-top: 10px;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .login-footer {
        margin-top: 15px;
        text-align: center;
        font-size: 14px;
    }

    .login-footer a {
        color: #007bff;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)



    # Login form inside a container for better layout
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # User input fields
        username = st.text_input("Username", placeholder="Enter your username", key="username")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="password")

        # Optional: Remember me checkbox
        remember_me = st.checkbox("Remember me", key="remember_me")

        # Login button with handler
        if st.button("Login", use_container_width=True):
            if username and password:
                # Replace with your FastAPI backend URL for authentication
                response = requests.post('http://127.0.0.1:8000/token', data={'username': username, 'password': password})

                if response.status_code == 200:
                    # Extract the JWT token from the response
                    token = response.json().get("access_token")
                    if token:
                        st.session_state['user_logged_in'] = True
                        st.session_state['access_token'] = token  # Save token in session state
                        st.success("Login Successful! Redirecting to Dashboard...")
                        
                        # Use session_state to trigger a rerun
                        st.session_state.login_success = True
                        st.rerun()  # Force re-run and redirect to the dashboard

                    else:
                        st.error("❌ No token received, login failed.")
                else:
                    st.error("❌ Invalid credentials. Please try again.")

            else:
                st.error("❌ Please enter both username and password.")

        # Optional: Add forgot password or sign-up link
        st.markdown("""
            <div style='text-align:center; margin-top: 10px'>
                <a href='#' style="font-size: 14px; color: #007bff; text-decoration: none;">Forgot Password?</a>
                <p>Don't have an account? <a href='#' style="font-size: 14px; color: #007bff; text-decoration: none;">Sign up</a></p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # End of the container

# Check if login_success exists in session_state, and rerun if necessary
if "login_success" in st.session_state and st.session_state.login_success:
    # After login success, simulate redirect by setting the flag to False
    st.session_state.login_success = False
    st.rerun()  # Rerun the app to display the dashboard or redirect to another page
# Function to handle API requests with authentication
def get_data_from_backend():
    token = st.session_state.get('access_token')
    
    if token:
        headers = {
            'Authorization': f'Bearer {token}'  # Add the token as Bearer token
        }
        
        # Make your request with the header containing the token
        response = requests.get('http://127.0.0.1:8000/predict/', headers=headers)

        if response.status_code == 200:
            data = response.json()
            # Do something with the data
            return data
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.error("❌ No valid token found.")