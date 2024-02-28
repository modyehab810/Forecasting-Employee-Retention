# Importing ToolKits
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st
import warnings


def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={
            "font": {
                "size": title_size,
                "family": "tahoma"
            }
        },
        hoverlabel={
            "bgcolor": "#000",
            "font_size": hover_font_size,
            "font_family": "arial"
        }

    )


def box_plot(the_df, column):
    fig = px.box(
        data_frame=the_df,
        x=column,
        title=f'{column.title().replace("_", " ")} Distribution & 5-Summary',
        template="plotly_dark",
        labels={column: column.title().replace("_", " ")},
        height=600,
        color_discrete_sequence=['#17B794']

    )

    custome_layout(fig, showlegend=False)

    return fig


def bar_plot(the_df, column, orientation="v", top_10=False):
    dep = the_df[column].value_counts()
    if top_10:
        dep = the_df[column].value_counts().nlargest(10)

    fig = px.bar(data_frame=dep,
                 x=dep.index,
                 y=dep / sum(dep) * 100,
                 orientation=orientation,
                 color=dep.index.astype(str),
                 title=f'Observations Distribution Via {column.title().replace("_", " ")}',
                 color_discrete_sequence=["#17B794"],
                 labels={column: column.title().replace("_", " "),
                         "y": "Employees Frequency in PCT(%)"},
                 template="plotly_dark",
                 text=dep.apply(lambda x: f"{x / sum(dep) * 100:0.0f}%"),
                 height=650)

    fig.update_traces(
        textfont={
            "size": 20,
            "family": "consolas",
            "color": "#000"
        },
        hovertemplate="X Axis: %{x}<br>Y Axis: %{y:0.1f}%",
    )

    if orientation == "h":
        fig = px.bar(data_frame=dep,
                     y=dep.index[::-1],
                     x=dep[::-1] / sum(dep) * 100,
                     orientation=orientation,
                     color=dep.index.astype(str),
                     title=f'Observations Distribution Via {column.title().replace("_", " ")}',
                     color_discrete_sequence=["#17B394"],
                     labels={"y": column.title().replace("_", " "),
                             "x": "Employees Frequency in PCT(%)"},
                     template="plotly_dark",
                     text=dep[::-
                              1].apply(lambda x: f"{x / sum(dep) * 100:0.0f}%"),
                     height=650)

        fig.update_traces(
            textfont={
                "size": 20,
                "family": "consolas",
                "color": "#000"

            },
            hovertemplate="X Axis: %{y}<br>Y Axis: %{x:0.1f}%",
        )
    custome_layout(fig, title_size=28)

    return fig


def pie_chart(the_df, column):
    counts = the_df[column].value_counts()

    fig = px.pie(data_frame=counts,
                 names=counts.index,
                 values=counts,
                 title=f'Popularity of {column.title().replace("_", " ")}',
                 color_discrete_sequence=["#17B794", "#EEB76B", "#9C3D54"],
                 template="plotly_dark",
                 height=650
                 )

    custome_layout(fig, showlegend=True, title_size=28)
    pulls = np.zeros(len(counts))
    pulls[-1] = 0.1

    fig.update_traces(
        textfont={
            "size": 16,
            "family": "arial",
            "color": "#fff"
        },
        hovertemplate="Label:%{label}<br>Frequency: %{value:0.4s}<br>Percentage: %{percent}",
        marker=dict(line=dict(color='#000000', width=0.5)),
        pull=pulls,
    )

    return fig


# Main Vizualization Function
def create_vizualization(the_df, viz_type="box", data_type="number"):
    """
        This Function Take 3 Parameters [data_frame, viz_type, data_type]
        and return 3 
        1• [array of all created figures].
        2• df columns.
        3• target column Index according to dtype
    """
    figs = []
    num_columns = list(the_df.select_dtypes(include=data_type).columns)
    cols_index = []

    if viz_type == "box":

        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() > 10:
                figs.append(box_plot(
                    the_df, num_columns[i]))
                cols_index.append(i)

    if viz_type == "bar":

        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() < 8:
                figs.append(bar_plot(
                    the_df, num_columns[i]))
                cols_index.append(i)

            elif the_df[num_columns[i]].nunique() >= 8 and the_df[num_columns[i]].nunique() < 15:
                figs.append(bar_plot(
                    the_df, num_columns[i], "h"))
                cols_index.append(i)

            if the_df[num_columns[i]].nunique() >= 15:
                figs.append(bar_plot(
                    the_df, num_columns[i], "h", top_10=True))
                cols_index.append(i)

    if viz_type == "pie":
        num_columns = list(the_df.columns)
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() <= 4:
                figs.append(pie_chart(
                    the_df, num_columns[i]))
                cols_index.append(i)

    if len(cols_index) > 0:

        tabs = st.tabs(
            [str(num_columns[i]).title().replace("_", " ") for i in cols_index])

        for i in range(len(cols_index)):
            tabs[i].plotly_chart(figs[i], use_container_width=True)

    # return figs, num_columns, cols_index


def create_heat_map(the_df):
    correlation = the_df.corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        template="plotly_dark",
        text_auto="0.2f",
        aspect=1,
        color_continuous_scale="greens",
        title="Correlation Heatmap of Data",
        height=650,
    )
    fig.update_traces(
        textfont={
            "size": 16,
            "family": "consolas"
        },

    )
    fig.update_layout(
        title={
            "font": {
                "size": 30,
                "family": "tahoma"
            }
        },
        hoverlabel={
            "bgcolor": "#111",
            "font_size": 15,
            "font_family": "consolas"
        }
    )
    return fig
