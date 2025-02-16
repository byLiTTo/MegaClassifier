import plotly.graph_objects as go


def create_pie_chart(data_labels, data_values, title, width=1000, height=600, show_values=False,
                     text_position="inside") -> go.Figure:
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
    figure = go.Figure(data=[
        go.Pie(labels=data_labels, values=data_values, hole=0.3, text=data_values if show_values else None,
               textposition=text_position)])

    figure.update_layout(title=title, width=width, height=height, template="seaborn", )

    return figure


def create_bar_chart(data_labels, data_values, x_title="", y_title="", title="", width=1000, height=600,
                     show_values=False, text_position="auto") -> go.Figure:
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
        data=[go.Bar(x=data_labels, y=data_values, text=data_values if show_values else None,
                     textposition=text_position)], )

    figure.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height,
                         template="seaborn", )

    return figure
