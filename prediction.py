# Importing ToolKits
import pandas as pd
import numpy as np
import plotly.express as px


import streamlit as st
import warnings


def creat_matrix_score_cards(card_image="", card_title="Card Title", card_value=None, percent=False):
    st.image(card_image,
             caption="", width=70)

    st.subheader(
        card_title)

    if percent:
        st.subheader(
            f"{card_value}%")

    else:
        st.subheader(
            f"{card_value}")


def create_comparison_df(y_actual, y_pred):
    predected_df = pd.DataFrame()
    predected_df["Actual Spent Values"] = y_actual
    predected_df.reset_index(
        drop=True, inplace=True)
    predected_df["Predicted Spent Value"] = y_pred

    return predected_df


def create_confusion_plot(cm):
    fig = px.imshow(cm, aspect=True, text_auto="0.0f", template="plotly_dark",
                    color_continuous_scale="greens", x=["Stay", "Left"], y=["Saty", "Left"], height=550)

    fig.update_traces(
        textfont={
            "size": 15,
            "family": "consolas"
        }
    )
    return fig
