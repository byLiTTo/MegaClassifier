import pandas as pd
import plotly.graph_objects as go


def create_pie_chart(
    data_labels,
    data_values,
    title,
    width=1000,
    height=600,
    show_values=False,
    text_position="inside",
) -> go.Figure:
    """
    Creates a pie chart using Plotly with customizable options such as dimensions, title, and value display.

    Args:
        data_labels (List[str]): A list of labels for the pie chart segments.
        data_values (List[float]): A list of numerical values corresponding to each label in the pie chart.
        title (str): The title of the pie chart.
        width (int): The width of the pie chart figure. Defaults to 1000.
        height (int): The height of the pie chart figure. Defaults to 600.
        show_values (bool): Whether to display the values inside the pie chart segments. Defaults to False.
        text_position (str): The position of the text within the pie chart. Options include "inside", "outside", etc.
            Defaults to "inside".

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object representing the pie chart.

    """
    figure = go.Figure(
        data=[
            go.Pie(
                labels=data_labels,
                values=data_values,
                hole=0.3,
                text=data_values if show_values else None,
                textposition=text_position,
            )
        ]
    )

    figure.update_layout(
        title=title,
        width=width,
        height=height,
        template="seaborn",
    )

    return figure


def create_bar_chart(
    data_labels,
    data_values,
    x_title="",
    y_title="",
    title="",
    width=1000,
    height=600,
    show_values=False,
    text_position="auto",
) -> go.Figure:
    """
    Creates a bar chart visualization using input data and plotly.

    This function generates a bar chart based on the provided labels and values. It allows for customization
    of chart aesthetics, including axis titles, overall title, dimensions, and textual representation of values
    on the bars.

    Args:
        data_labels: A list containing labels for the x-axis.
        data_values: A list containing numeric values corresponding to the data labels for the y-axis.
        x_title: A string representing the title of the x-axis.
        y_title: A string representing the title of the y-axis.
        title: A string for the chart title.
        width: An integer specifying the width of the chart in pixels.
        height: An integer specifying the height of the chart in pixels.
        show_values: A boolean flag to determine if values should be displayed on the bars.
        text_position: A string defining the positioning of the text on bars (e.g., 'outside', 'auto').

    Returns:
        plotly.graph_objects.Figure: A plotly figure object containing the bar chart visualization.

    Raises:
        ValueError: If lengths of `data_labels` and `data_values` do not match.
    """
    figure = go.Figure(
        data=[
            go.Bar(
                x=data_labels,
                y=data_values,
                text=data_values if show_values else None,
                textposition=text_position,
            )
        ],
    )

    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=width,
        height=height,
        template="seaborn",
    )

    return figure


def create_heatmap_chart(
    conf_matrix,
    x_data,
    y_data,
    conf_matrix_text,
    title="",
    x_title="",
    y_title="",
    width=600,
    height=600,
) -> go.Figure:
    """
    Creates a heatmap chart based on the provided data and configuration.

    This function generates a heatmap chart using Plotly's `go.Heatmap` based on the
    specified confusion matrix, axis data, and additional parameters. The chart is
    highly customizable, allowing users to define titles, axis labels, and layout
    dimensions.

    Args:
        conf_matrix: The 2D list or array representing the values to be displayed in the heatmap.
        x_data: A list of labels for the x-axis of the heatmap.
        y_data: A list of labels for the y-axis of the heatmap.
        conf_matrix_text: A 2D list of text annotations to be displayed over the heatmap cells.
        title: The title of the heatmap chart. Defaults to an empty string.
        x_title: The label for the x-axis. Defaults to an empty string.
        y_title: The label for the y-axis. Defaults to an empty string.
        width: The width of the heatmap chart in pixels. Defaults to 600.
        height: The height of the heatmap chart in pixels. Defaults to 600.

    Returns:
        go.Figure: A Plotly figure object containing the heatmap chart.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=x_data,
            y=y_data,
            text=conf_matrix_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=x_data),
        yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=y_data),
        template="seaborn",
        width=width,
        height=height,
    )

    return fig


def create_roc_curve_chart(model_name, fpr, tpr, roc_auc):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC = {roc_auc:.4f})"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier (AUC = 0.5000",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title=f"ROC Curve {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Curves",
        template="seaborn",
        width=700,
        height=500,
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=0.1,
        ),
    )

    return fig


def create_confusion_matrix_chart(conf_matrix, conf_matrix_text, model_name):
    return create_heatmap_chart(
        conf_matrix=conf_matrix,
        conf_matrix_text=conf_matrix_text,
        x_data=["Empty", "Animal"],
        y_data=["Empty", "Animal"],
        title=f"{model_name} (Subset: test)",
        x_title="Model",
        y_title="Dataset",
    )


def create_training_accuracy_chart(history_path, model_name):
    history_df = pd.read_csv(history_path, sep=";")

    fig_accu = go.Figure()
    fig_accu.add_trace(
        go.Scatter(
            x=list(range(1, len(history_df["accuracy"]) + 1)),
            y=history_df["accuracy"],
            mode="lines+markers",
            name="Training Accuracy",
            line=dict(width=2),
        )
    )

    fig_accu.add_trace(
        go.Scatter(
            x=list(range(1, len(history_df["val_accuracy"]) + 1)),
            y=history_df["val_accuracy"],
            mode="lines+markers",
            name="Validation Accuracy",
            line=dict(width=2),
        )
    )

    fig_accu.update_layout(
        title=f"Accuracy - {model_name}",
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
        template="seaborn",
        width=700,
        height=500,
    )

    return fig_accu


def create_training_loss_chart(history_path, model_name):
    history_df = pd.read_csv(history_path, sep=";")

    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, len(history_df["loss"]) + 1)),
            y=history_df["loss"],
            mode="lines+markers",
            name="Training Loss",
            line=dict(width=2),
        )
    )

    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, len(history_df["val_loss"]) + 1)),
            y=history_df["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(width=2),
        )
    )

    fig_loss.update_layout(
        title=f"Loss - {model_name}",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        template="seaborn",
        width=700,
        height=500,
    )

    return fig_loss
