#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################


def border_style(color="blue", direction="bottom", border=3, padding=4):
    assert direction in ["top", "bottom", "right", "left"]
    style = (
        f"border-{direction}: {border}px solid {color};"
        f"padding-{direction}: {padding}px;"
        "background-color:none;"
    )

    return style
