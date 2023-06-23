import streamlit as st
import pandas as pd
import altair as alt
import typing
import numpy as np
import cv2
import tempfile
import os


def create_overall_result_chart(data: pd.DataFrame):
    data["predict_text"] = data["predict"].astype(str) + "%"
    chart = alt.Chart(data).encode(
        x=alt.X(
            "predict:Q",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(title="Real Level", labelFontSize=12, labelFontWeight="bold"),
        ),
        y=alt.Y(
            "model:N",
            axis=alt.Axis(title="Model Name", labelFontSize=12, labelFontWeight="bold"),
        ),
        text=alt.Text(
            "predict_text:N",
        ),
    )

    # Combine the bar chart and value labels
    chart = chart.mark_bar(
        size=20,
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#f2ce99", offset=0),
                alt.GradientStop(color="orange", offset=1),
            ],
            x1=1,
            x2=0,
            y1=0,
            y2=0,
        ),
    ) + chart.mark_text(
        align="left",
        dx=3,
        fontSize=14,  # Set the font size of the labels
        fontWeight="bold",  # Set the font weight of the labels
        color="black",
    )
    chart = chart.properties(width=500, height=250)
    return chart


def create_frame_detail_result_chart(data):
    max_frame_id = data["frame_id"].max()
    area_chart = alt.Chart(data).encode(
        x=alt.X(
            "frame_id",
            scale=alt.Scale(domain=[0, max_frame_id]),
            axis=alt.Axis(
                title="Frame number", labelFontSize=12, labelFontWeight="bold"
            ),
        ),
        y=alt.Y(
            "predict",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(title="Real level", labelFontSize=12, labelFontWeight="bold"),
        ),
    )
    area_chart = area_chart.mark_area(
        interpolate="monotone",
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#f2ce99", offset=0),
                alt.GradientStop(color="orange", offset=1),
            ],
            x1=1,
            x2=1,
            y1=1,
            y2=0,
        ),
    )
    area_chart = area_chart.properties(width=700, height=250)
    return area_chart
