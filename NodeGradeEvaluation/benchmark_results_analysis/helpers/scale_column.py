def scale_column(df, column_name, new_min=0, new_max=5, old_min=None, old_max=None):
    """
    Scales the specified column in the DataFrame to a new range.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to scale.
        column_name (str): The name of the column to scale.
        new_min (float, optional): The minimum value of the new range. Defaults to 0.
        new_max (float, optional): The maximum value of the new range. Defaults to 5.
        old_min (float, optional): The minimum value of the original range. If None, uses the column's min. Defaults to None.
        old_max (float, optional): The maximum value of the original range. If None, uses the column's max. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with the scaled column.
    """
    if old_min is None:
        old_min = df[column_name].min()
    if old_max is None:
        old_max = df[column_name].max()

    if old_max == old_min:
        raise ValueError(f"Cannot scale column '{column_name}' because old_max ({old_max}) and old_min ({old_min}) are equal.")

    df[column_name] = (df[column_name] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return df
