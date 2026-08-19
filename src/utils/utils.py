import os

import pandas as pd
import errno


def save_dataframe(df, output_name, output_filepath):
    """
    Save a dataframe to a CSV file.
    """
    try:
        os.makedirs(output_filepath, exist_ok=True)
        full_output_path = os.path.join(output_filepath, output_name)
        df.to_csv(full_output_path, index=False)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
