import streamlit as st
import pandas as pd
import sklearn
import pickle
import xgboost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

st.write("""
#Project Sporting Bets - The App!
         
This is our streamlit app for the Datascientest project.
Below you can fill in the parameters to see on who to bet on! :)
""")

streamlit_model = pickle.load(open('streamlit_xgb.pkl', 'rb'))

if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

playerlist = {
    'Anderson K.':['Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Cecchinato M.':['Anderson K.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Cilic M.':['Anderson K.','Cecchinato M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Coric B.':['Anderson K.','Cecchinato M.','Cilic M.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Del Potro J.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Dimitrov G.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Djokovic N.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Edmund K.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Federer R.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Fgonini B.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Isner J.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Khachanov K.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Raonic M.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Schwartzman D.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Thiem D.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Tsitsipas S.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Medvedev D.','Nadal R.','Nishikori K.','Zverev A.'],
    'Medvedev D.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Nadal R.','Nishikori K.','Zverev A.'],
    'Nadal R.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nishikori K.','Zverev A.'],
    'Nishikori K.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Zverev A.'],
    'Zverev A.':['Anderson K.','Cecchinato M.','Cilic M.','Coric B.','Del Potro J.','Dimitrov G.','Djokovic N.','Edmund K.','Federer R.','Fgonini B.','Isner J.','Khachanov K.','Raonic M.','Schwartzman D.','Thiem D.','Tsitsipas S.','Medvedev D.','Nadal R.','Nishikori K.'],
}


if __name__ == '__main__':
    player1 = st.selectbox('Player 1', options=['-Select-']+list(playerlist.keys()))
    if player1 != '-Select-':
        player2 = st.selectbox('Player 2', options=['-Select-']+list(playerlist[player1]))
        if player2 != '-Select-':
            tournament = st.selectbox('Tournament',('-Select-','Wimbledon','Australian Open','French Open','US Open'))

    st.write(f'The predicted winner is:'streamlit_model(__name__))



# with col2:
#     text_input = st.text_input(
#         "Enter ATP Rank of Player1",
#         label_visibility=st.session_state.visibility,
#         disabled=st.session_state.disabled,
#         placeholder=st.session_state.placeholder,
#     )

# xgb_model_loaded.predict(test2)


# THIS IS THE IMPORTANT STEP