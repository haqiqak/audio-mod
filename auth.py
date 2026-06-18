"""
auth.py — Login / Register screen for the standalone profiling app.

Call  require_auth()  at the top of app.py (after set_page_config).
Sets on success:
  st.session_state.authenticated = True
  st.session_state.current_user  = "<username>"
"""

import streamlit as st
from user_store import list_users, register_user, verify_user

_AUTH_CSS = """
<style>
/* stop native form controls (inputs, tabs) from picking up the browser's
   dark mode — this was the source of the black input/tab boxes that only
   became readable on hover/focus */
html { color-scheme: light only; }

/* kill Streamlit's default top padding + wide-layout max-width so the
   login screen doesn't sit under a huge empty gap and feel "long" */
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 2.2rem !important;
    max-width: 480px !important;
}
[data-testid="stHeader"] { background: transparent; }

html, body, [data-testid="stAppViewContainer"] { background: #f7f8fb; }

.auth-root {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1rem 0 0;
}
.auth-card {
    background: #fff;
    border: 1.5px solid #e2e6ee;
    border-radius: 18px;
    padding: 2.2rem 2.4rem 1.8rem;
    width: 100%;
    box-shadow: 0 4px 24px rgba(20,30,60,.06);
}
.auth-logo { text-align: center; margin-bottom: 1.2rem; }
.auth-logo h1 { font-size: 1.8rem; color: #11172a; margin: 0 0 .15rem; font-weight: 700; }
.auth-logo h1 span { color: #2f6fed; }
.auth-logo p  { font-size: .8rem; color: #5b6478; margin: 0; }

/* text inputs */
[data-testid="stTextInput"] input {
    background: #ffffff !important; color: #1b2433 !important;
    border: 1px solid #d8dde8 !important; border-radius: 9px !important;
}
[data-testid="stTextInput"] label p { color: #1b2433 !important; font-weight: 500; }

/* tabs (Login / Register) */
[data-testid="stTabs"] button[role="tab"] { color: #5b6478 !important; font-weight: 600; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #11172a !important; border-bottom-color: #2f6fed !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #2f6fed !important; }

/* buttons */
.stButton > button {
    background: #2f6fed !important; color: #ffffff !important;
    border: 1px solid #2f6fed !important; border-radius: 9px !important; font-weight: 600;
}
.stButton > button:hover { background: #2660d4 !important; }

/* alerts (error / success messages) */
[data-testid="stAlert"] p { color: inherit !important; }
</style>
"""


def require_auth() -> None:
    if st.session_state.get("authenticated"):
        return

    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-root"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-logo">'
        '<h1>Audio <span>Mod</span></h1>'
        '<p>Audio Module for Speech AI — disfluency detection &amp; profiling</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    tab_login, tab_reg = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in", use_container_width=True, key="btn_login"):
            ok, msg = verify_user(username, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.current_user = username.strip().lower()
                st.rerun()
            else:
                st.error(msg)

    with tab_reg:
        new_user = st.text_input("Choose a username", key="reg_user")
        new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        if st.button("Create account", use_container_width=True, key="btn_reg"):
            ok, msg = register_user(new_user, new_pass)
            if ok:
                st.success("Account created — please sign in.")
            else:
                st.error(msg)

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()
