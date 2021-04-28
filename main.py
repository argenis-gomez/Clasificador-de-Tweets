import streamlit as st
from utils import *


def main():
    st.title('Clasificador de Tweets.')

    st.write('Introduce el Tweet:')

    tweet = st.text_input('Ingrese su tweet en inglés', '', max_chars=100)

    if st.button('Clasificar'):
        clasificacion = classify(tweet, MODEL)
        st.success(f'Sentimiento: {clasificacion[0]:2.2f}% {clasificacion[1].title()}')


if __name__ == '__main__':
    MODEL = load_model()
    print(classify('good morning', MODEL))
    #main()
